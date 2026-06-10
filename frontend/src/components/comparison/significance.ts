// SPDX-License-Identifier: Apache-2.0

import type { ComparisonCard, ComparisonInstance } from "@/data/comparisons";
import { coveragePrecision, f1, formatCompactMagnitude, formatDurationDelta, formatSignedFixed, getComparisonPair } from "@/components/comparison/format";

type PairedValue = {
  baseline: number;
  treatment: number;
};

type PairedInstance = {
  baseline: ComparisonInstance;
  treatment: ComparisonInstance;
};

type SignificanceUnit = "number" | "percent-point" | "duration-ms" | "tokens" | "currency";

type MetricSpec = {
  key: string;
  test: "mcnemar" | "permutation";
  unit: SignificanceUnit;
  value: (instance: ComparisonInstance) => number | null;
  aggregateValue?: (instances: ComparisonInstance[]) => number | null;
  effectLabel?: string;
};

export type SignificanceGroupKey = "language" | "bench" | "repositorySize";

export type PairedSignificance = {
  metricKey: string;
  testName: string;
  n: number;
  effect: number;
  ciLow: number;
  ciHigh: number;
  pValue: number;
  qValue: number;
  significant: boolean;
  unit: SignificanceUnit;
  effectLabel?: string;
};

const SIGNIFICANCE_ALPHA = 0.05;
const RESAMPLE_COUNT = 2_000;
const significanceCache = new WeakMap<ComparisonCard, Map<string, PairedSignificance>>();
const groupedSignificanceCache = new WeakMap<ComparisonCard, Map<string, Map<string, PairedSignificance>>>();

const metricSpecs: MetricSpec[] = [
  {
    key: "completedRunRate",
    test: "mcnemar",
    unit: "percent-point",
    value: (instance) => instance.outcome.status ? (instance.outcome.status === "completed" ? 1 : 0) : null,
  },
  {
    key: "patchProductionRate",
    test: "mcnemar",
    unit: "percent-point",
    value: (instance) => instance.artifacts ? (instance.artifacts.hasModelPatch ? 1 : 0) : null,
  },
  {
    key: "validEvaluationRate",
    test: "mcnemar",
    unit: "percent-point",
    value: (instance) => instance.artifacts?.evaluationStatus ? (instance.artifacts.evaluationStatus === "valid" ? 1 : 0) : null,
  },
  {
    key: "officialPassAt1",
    test: "mcnemar",
    unit: "percent-point",
    value: (instance) => {
      const status = instance.artifacts?.resolutionStatus;
      if (status !== "resolved" && status !== "unresolved") return null;
      return status === "resolved" ? 1 : 0;
    },
  },
  {
    key: "contextF1",
    test: "permutation",
    unit: "number",
    value: (instance) => contextAggregate(instance, "f1"),
    aggregateValue: (instances) => contextAggregatePooled(instances, "f1"),
  },
  {
    key: "trajectoryGoldFound",
    test: "permutation",
    unit: "number",
    value: (instance) => trajectoryGoldFoundAggregate(instance),
    aggregateValue: (instances) => trajectoryGoldFoundAggregatePooled(instances),
  },
  {
    key: "contextRecall",
    test: "permutation",
    unit: "number",
    value: (instance) => contextAggregate(instance, "recall"),
    aggregateValue: (instances) => contextAggregatePooled(instances, "recall"),
  },
  {
    key: "contextPrecision",
    test: "permutation",
    unit: "number",
    value: (instance) => contextAggregate(instance, "precision"),
    aggregateValue: (instances) => contextAggregatePooled(instances, "precision"),
  },
  ...contextLevelSpecs("file", "file"),
  ...contextLevelSpecs("symbol", "symbol"),
  ...contextLevelSpecs("block", "span"),
  ...contextLevelSpecs("line", "line"),
  {
    key: "fileF1",
    test: "permutation",
    unit: "number",
    value: (instance) => contextLevelMetric(instance, "file", "f1"),
    aggregateValue: (instances) => contextLevelMetricPooled(instances, "file", "f1"),
  },
  {
    key: "symbolF1",
    test: "permutation",
    unit: "number",
    value: (instance) => contextLevelMetric(instance, "symbol", "f1"),
    aggregateValue: (instances) => contextLevelMetricPooled(instances, "symbol", "f1"),
  },
  {
    key: "spanF1",
    test: "permutation",
    unit: "number",
    value: (instance) => contextLevelMetric(instance, "span", "f1"),
    aggregateValue: (instances) => contextLevelMetricPooled(instances, "span", "f1"),
  },
  {
    key: "avgLineF1",
    test: "permutation",
    unit: "number",
    value: (instance) => contextLevelMetric(instance, "line", "f1"),
    aggregateValue: (instances) => contextLevelMetricPooled(instances, "line", "f1"),
  },
  {
    key: "fixOverlapVsGoldRecall",
    test: "permutation",
    unit: "percent-point",
    value: (instance) => fixOverlapMetric(instance, "recall"),
    aggregateValue: (instances) => fixOverlapMetricPooled(instances, "recall"),
  },
  {
    key: "fixOverlapVsGoldPrecision",
    test: "permutation",
    unit: "percent-point",
    value: (instance) => fixOverlapMetric(instance, "precision"),
    aggregateValue: (instances) => fixOverlapMetricPooled(instances, "precision"),
  },
  {
    key: "fixOverlapVsGoldF1",
    test: "permutation",
    unit: "percent-point",
    value: (instance) => fixOverlapMetric(instance, "f1"),
    aggregateValue: (instances) => fixOverlapMetricPooled(instances, "f1"),
  },
  {
    key: "averageSteps",
    test: "permutation",
    unit: "number",
    value: (instance) => finiteOrNull(instance.trajectory.steps),
  },
  {
    key: "averageDuration",
    test: "permutation",
    unit: "duration-ms",
    value: (instance) => positiveFiniteOrNull(instance.resources.durationMs),
  },
  {
    key: "totalTokens",
    test: "permutation",
    unit: "tokens",
    value: (instance) => finiteOrZero(instance.resources.totalTokens),
  },
  {
    key: "estimatedCost",
    test: "permutation",
    unit: "currency",
    value: (instance) => finiteOrNull(instance.resources.costUsd),
  },
  {
    key: "mcpToolCalls",
    test: "permutation",
    unit: "number",
    value: (instance) => finiteOrZero(instance.resources.mcpToolCalls),
  },
  {
    key: "toolCalls",
    test: "permutation",
    unit: "number",
    value: (instance) => finiteOrZero(instance.resources.toolCalls),
  },
  {
    key: "commandExecutions",
    test: "permutation",
    unit: "number",
    value: (instance) => finiteOrZero(instance.resources.commandExecutions),
  },
  {
    key: "rawTraceEvents",
    test: "permutation",
    unit: "number",
    value: (instance) => finiteOrZero(instance.resources.rawTraceEvents),
  },
  {
    key: "rawAgentActions",
    test: "permutation",
    unit: "number",
    value: (instance) => finiteOrZero(instance.resources.rawAgentActions),
  },
  {
    key: "readToolCalls",
    test: "permutation",
    unit: "number",
    value: (instance) => finiteOrZero(instance.resources.readToolCalls),
  },
  {
    key: "editToolCalls",
    test: "permutation",
    unit: "number",
    value: (instance) => finiteOrZero(instance.resources.editToolCalls),
  },
  {
    key: "skillInvocations",
    test: "permutation",
    unit: "number",
    value: (instance) => finiteOrZero(instance.skills?.totalInvocations),
  },
  {
    key: "toolInvocations",
    test: "permutation",
    unit: "number",
    value: (instance) => finiteOrZero(instance.tools?.totalInvocations),
  },
];

export function getMetricSignificance(comparison: ComparisonCard, metricKey: string): PairedSignificance | null {
  return buildSignificanceLookup(comparison).get(metricKey) ?? null;
}

export function getGroupedMetricSignificance(
  comparison: ComparisonCard,
  metricKey: string,
  groupKey: SignificanceGroupKey,
  groupValue: string | null | undefined,
): PairedSignificance | null {
  if (!groupValue) return null;
  return buildGroupedSignificanceLookup(comparison, metricKey, groupKey).get(groupValue) ?? null;
}

export function formatSignificanceEffect(stat: PairedSignificance): string {
  return formatSignificanceValue(stat.effect, stat.unit);
}

export function formatSignificanceInterval(stat: PairedSignificance): string {
  return `${formatSignificanceValue(stat.ciLow, stat.unit)} to ${formatSignificanceValue(stat.ciHigh, stat.unit)}`;
}

export function formatPValue(value: number): string {
  if (value < 0.001) return "<0.001";
  if (value < 0.01) return value.toFixed(3);
  return value.toFixed(2);
}

function buildSignificanceLookup(comparison: ComparisonCard): Map<string, PairedSignificance> {
  const cached = significanceCache.get(comparison);
  if (cached) return cached;

  const pair = getComparisonPair(comparison);
  if (!pair) {
    const empty = new Map<string, PairedSignificance>();
    significanceCache.set(comparison, empty);
    return empty;
  }

  const rawStats = metricSpecs
    .map((spec) => calculateSignificance(comparison, spec))
    .filter((stat): stat is PairedSignificance => stat !== null);
  applyBenjaminiHochberg(rawStats);
  const lookup = new Map(rawStats.map((stat) => [stat.metricKey, stat]));
  significanceCache.set(comparison, lookup);
  return lookup;
}

function calculateSignificance(comparison: ComparisonCard, spec: MetricSpec): PairedSignificance | null {
  if (spec.aggregateValue) {
    return calculateAggregateSignificanceFromPairs(comparison.id, spec, pairedInstances(comparison));
  }
  const values = pairedValues(comparison, spec);
  return calculateSignificanceFromValues(comparison.id, spec, values);
}

function buildGroupedSignificanceLookup(
  comparison: ComparisonCard,
  metricKey: string,
  groupKey: SignificanceGroupKey,
): Map<string, PairedSignificance> {
  const cachedComparison = groupedSignificanceCache.get(comparison);
  const cacheKey = `${metricKey}:${groupKey}`;
  const cached = cachedComparison?.get(cacheKey);
  if (cached) return cached;

  const spec = metricSpecs.find((candidate) => candidate.key === metricKey);
  const groupedCache = cachedComparison ?? new Map<string, Map<string, PairedSignificance>>();
  if (!cachedComparison) groupedSignificanceCache.set(comparison, groupedCache);

  if (!spec) {
    const empty = new Map<string, PairedSignificance>();
    groupedCache.set(cacheKey, empty);
    return empty;
  }

  if (spec.aggregateValue) {
    const pairsByGroup = groupedPairedInstances(comparison, groupKey);
    const rawAggregateStats = Array.from(pairsByGroup.entries())
      .map(([group, pairs]) => [
        group,
        calculateAggregateSignificanceFromPairs(`${comparison.id}:${groupKey}:${group}`, spec, pairs),
      ] as const)
      .filter((entry): entry is readonly [string, PairedSignificance] => entry[1] !== null);

    applyBenjaminiHochberg(rawAggregateStats.map(([, stat]) => stat));
    const aggregateLookup = new Map(rawAggregateStats.map(([group, stat]) => [group, stat]));
    groupedCache.set(cacheKey, aggregateLookup);
    return aggregateLookup;
  }

  const valuesByGroup = groupedPairedValues(comparison, spec, groupKey);
  const rawStats = Array.from(valuesByGroup.entries())
    .map(([group, values]) => [
      group,
      calculateSignificanceFromValues(`${comparison.id}:${groupKey}:${group}`, spec, values),
    ] as const)
    .filter((entry): entry is readonly [string, PairedSignificance] => entry[1] !== null);

  applyBenjaminiHochberg(rawStats.map(([, stat]) => stat));
  const lookup = new Map(rawStats.map(([group, stat]) => [group, stat]));
  groupedCache.set(cacheKey, lookup);
  return lookup;
}

function calculateAggregateSignificanceFromPairs(
  seedKey: string,
  spec: MetricSpec,
  pairs: PairedInstance[],
): PairedSignificance | null {
  if (!spec.aggregateValue) return null;

  const completePairs = completeCaseAggregatePairs(pairs, spec);
  if (completePairs.length < 2) return null;

  const baselineValue = spec.aggregateValue(completePairs.map((pair) => pair.baseline));
  const treatmentValue = spec.aggregateValue(completePairs.map((pair) => pair.treatment));
  if (baselineValue === null || treatmentValue === null) return null;

  const effect = treatmentValue - baselineValue;
  const interval = bootstrapAggregateInterval(completePairs, spec.aggregateValue, hashSeed(`${seedKey}:${spec.key}:bootstrap`));
  if (!interval) return null;
  const [ciLow, ciHigh] = interval;
  const pValue = signFlipAggregatePValue(completePairs, spec.aggregateValue, effect, hashSeed(`${seedKey}:${spec.key}:permutation`));

  return {
    metricKey: spec.key,
    testName: "Paired sign-flip permutation",
    n: completePairs.length,
    effect,
    ciLow,
    ciHigh,
    pValue,
    qValue: pValue,
    significant: false,
    unit: spec.unit,
    effectLabel: spec.effectLabel ?? "Aggregate effect on complete paired tasks",
  };
}

function completeCaseAggregatePairs(pairs: PairedInstance[], spec: MetricSpec): PairedInstance[] {
  return pairs.filter((pair) => safeMetricValue(spec, pair.baseline) !== null && safeMetricValue(spec, pair.treatment) !== null);
}

function calculateSignificanceFromValues(
  seedKey: string,
  spec: MetricSpec,
  values: PairedValue[],
): PairedSignificance | null {
  if (values.length < 2) return null;

  const deltas = values.map((value) => value.treatment - value.baseline);
  const effect = mean(deltas);
  const [ciLow, ciHigh] = bootstrapMeanInterval(deltas, hashSeed(`${seedKey}:${spec.key}:bootstrap`));
  const pValue = spec.test === "mcnemar"
    ? exactMcNemarPValue(values)
    : signFlipPValue(deltas, hashSeed(`${seedKey}:${spec.key}:permutation`));

  return {
    metricKey: spec.key,
    testName: spec.test === "mcnemar" ? "Exact McNemar" : "Paired sign-flip permutation",
    n: values.length,
    effect,
    ciLow,
    ciHigh,
    pValue,
    qValue: pValue,
    significant: false,
    unit: spec.unit,
    effectLabel: spec.effectLabel ?? (spec.test === "mcnemar" ? "Paired rate effect" : "Mean paired effect per task"),
  };
}

function groupedPairedValues(
  comparison: ComparisonCard,
  spec: MetricSpec,
  groupKey: SignificanceGroupKey,
): Map<string, PairedValue[]> {
  const pair = getComparisonPair(comparison);
  if (!pair) return new Map<string, PairedValue[]>();

  const baselineById = new Map((pair.baseline.instances ?? []).map((instance) => [instance.instanceId, instance]));
  const valuesByGroup = new Map<string, PairedValue[]>();
  for (const treatment of pair.treatment.instances ?? []) {
    const baseline = baselineById.get(treatment.instanceId);
    if (!baseline) continue;
    const group = significanceGroupValue(baseline, groupKey);
    if (!group) continue;
    const baselineValue = safeMetricValue(spec, baseline);
    const treatmentValue = safeMetricValue(spec, treatment);
    if (baselineValue === null || treatmentValue === null) continue;
    const values = valuesByGroup.get(group) ?? [];
    values.push({ baseline: baselineValue, treatment: treatmentValue });
    valuesByGroup.set(group, values);
  }
  return valuesByGroup;
}

function pairedInstances(comparison: ComparisonCard): PairedInstance[] {
  const pair = getComparisonPair(comparison);
  if (!pair) return [];

  const baselineById = new Map((pair.baseline.instances ?? []).map((instance) => [instance.instanceId, instance]));
  return (pair.treatment.instances ?? [])
    .map((treatment): PairedInstance | null => {
      const baseline = baselineById.get(treatment.instanceId);
      return baseline ? { baseline, treatment } : null;
    })
    .filter((value): value is PairedInstance => value !== null);
}

function groupedPairedInstances(
  comparison: ComparisonCard,
  groupKey: SignificanceGroupKey,
): Map<string, PairedInstance[]> {
  const pair = getComparisonPair(comparison);
  if (!pair) return new Map<string, PairedInstance[]>();

  const baselineById = new Map((pair.baseline.instances ?? []).map((instance) => [instance.instanceId, instance]));
  const pairsByGroup = new Map<string, PairedInstance[]>();
  for (const treatment of pair.treatment.instances ?? []) {
    const baseline = baselineById.get(treatment.instanceId);
    if (!baseline) continue;
    const group = significanceGroupValue(baseline, groupKey);
    if (!group) continue;
    const pairs = pairsByGroup.get(group) ?? [];
    pairs.push({ baseline, treatment });
    pairsByGroup.set(group, pairs);
  }
  return pairsByGroup;
}

function pairedValues(comparison: ComparisonCard, spec: MetricSpec): PairedValue[] {
  const pair = getComparisonPair(comparison);
  if (!pair) return [];

  const baselineById = new Map((pair.baseline.instances ?? []).map((instance) => [instance.instanceId, instance]));
  return (pair.treatment.instances ?? [])
    .map((treatment): PairedValue | null => {
      const baseline = baselineById.get(treatment.instanceId);
      if (!baseline) return null;
      const baselineValue = safeMetricValue(spec, baseline);
      const treatmentValue = safeMetricValue(spec, treatment);
      if (baselineValue === null || treatmentValue === null) return null;
      return { baseline: baselineValue, treatment: treatmentValue };
    })
    .filter((value): value is PairedValue => value !== null);
}

function safeMetricValue(spec: MetricSpec, instance: ComparisonInstance): number | null {
  try {
    return spec.value(instance);
  } catch {
    return null;
  }
}

const repositorySizeBuckets = [
  { key: "lt-1k", min: 0, max: 1_000 },
  { key: "1k-5k", min: 1_000, max: 5_000 },
  { key: "5k-20k", min: 5_000, max: 20_000 },
  { key: "20k-50k", min: 20_000, max: 50_000 },
  { key: "50k-plus", min: 50_000, max: Number.POSITIVE_INFINITY },
] as const;

function significanceGroupValue(instance: ComparisonInstance, groupKey: SignificanceGroupKey): string | null {
  if (groupKey === "language") return instance.language;
  if (groupKey === "bench") return instance.bench;

  const trackedFiles = instance.repositorySize?.trackedFiles;
  if (typeof trackedFiles !== "number") return null;
  return repositorySizeBuckets.find((bucket) => trackedFiles >= bucket.min && trackedFiles < bucket.max)?.key ?? null;
}

function applyBenjaminiHochberg(stats: PairedSignificance[]) {
  const ordered = [...stats].sort((left, right) => left.pValue - right.pValue);
  let nextQ = 1;
  for (let index = ordered.length - 1; index >= 0; index -= 1) {
    const rank = index + 1;
    const qValue = Math.min(nextQ, (ordered[index].pValue * ordered.length) / rank);
    ordered[index].qValue = qValue;
    ordered[index].significant = qValue < SIGNIFICANCE_ALPHA && intervalExcludesZero(ordered[index]);
    nextQ = qValue;
  }
}

function intervalExcludesZero(stat: PairedSignificance): boolean {
  return stat.ciLow > 0 || stat.ciHigh < 0;
}

function exactMcNemarPValue(values: PairedValue[]): number {
  const baselineLostTreatmentWon = values.filter((value) => value.baseline === 0 && value.treatment === 1).length;
  const baselineWonTreatmentLost = values.filter((value) => value.baseline === 1 && value.treatment === 0).length;
  const discordant = baselineLostTreatmentWon + baselineWonTreatmentLost;
  if (discordant === 0) return 1;

  const smaller = Math.min(baselineLostTreatmentWon, baselineWonTreatmentLost);
  const logProbabilities: number[] = [];
  for (let index = 0; index <= smaller; index += 1) {
    logProbabilities.push(logBinomialProbability(discordant, index));
  }
  return Math.min(1, Math.exp(logSumExp(logProbabilities)) * 2);
}

function signFlipPValue(deltas: number[], seed: number): number {
  const observed = Math.abs(mean(deltas));
  if (observed === 0) return 1;

  const random = seededRandom(seed);
  let extremeCount = 0;
  for (let iteration = 0; iteration < RESAMPLE_COUNT; iteration += 1) {
    const sampledMean = mean(deltas.map((delta) => random() < 0.5 ? delta : -delta));
    if (Math.abs(sampledMean) >= observed) extremeCount += 1;
  }
  return (extremeCount + 1) / (RESAMPLE_COUNT + 1);
}

function signFlipAggregatePValue(
  pairs: PairedInstance[],
  aggregateValue: (instances: ComparisonInstance[]) => number | null,
  observedEffect: number,
  seed: number,
): number {
  const observed = Math.abs(observedEffect);
  if (observed === 0) return 1;

  const random = seededRandom(seed);
  let validIterations = 0;
  let extremeCount = 0;
  for (let iteration = 0; iteration < RESAMPLE_COUNT; iteration += 1) {
    const sampledBaseline: ComparisonInstance[] = [];
    const sampledTreatment: ComparisonInstance[] = [];
    for (const pair of pairs) {
      if (random() < 0.5) {
        sampledBaseline.push(pair.baseline);
        sampledTreatment.push(pair.treatment);
      } else {
        sampledBaseline.push(pair.treatment);
        sampledTreatment.push(pair.baseline);
      }
    }
    const baselineValue = aggregateValue(sampledBaseline);
    const treatmentValue = aggregateValue(sampledTreatment);
    if (baselineValue === null || treatmentValue === null) continue;
    validIterations += 1;
    if (Math.abs(treatmentValue - baselineValue) >= observed) extremeCount += 1;
  }
  return (extremeCount + 1) / (validIterations + 1);
}

function bootstrapMeanInterval(deltas: number[], seed: number): [number, number] {
  const random = seededRandom(seed);
  const samples: number[] = [];
  for (let iteration = 0; iteration < RESAMPLE_COUNT; iteration += 1) {
    let total = 0;
    for (let index = 0; index < deltas.length; index += 1) {
      total += deltas[Math.floor(random() * deltas.length)];
    }
    samples.push(total / deltas.length);
  }
  samples.sort((left, right) => left - right);
  return [quantileSorted(samples, 0.025), quantileSorted(samples, 0.975)];
}

function bootstrapAggregateInterval(
  pairs: PairedInstance[],
  aggregateValue: (instances: ComparisonInstance[]) => number | null,
  seed: number,
): [number, number] | null {
  const random = seededRandom(seed);
  const samples: number[] = [];
  for (let iteration = 0; iteration < RESAMPLE_COUNT; iteration += 1) {
    const sampledBaseline: ComparisonInstance[] = [];
    const sampledTreatment: ComparisonInstance[] = [];
    for (let index = 0; index < pairs.length; index += 1) {
      const pair = pairs[Math.floor(random() * pairs.length)];
      sampledBaseline.push(pair.baseline);
      sampledTreatment.push(pair.treatment);
    }
    const baselineValue = aggregateValue(sampledBaseline);
    const treatmentValue = aggregateValue(sampledTreatment);
    if (baselineValue === null || treatmentValue === null) continue;
    samples.push(treatmentValue - baselineValue);
  }
  if (samples.length < 2) return null;
  samples.sort((left, right) => left - right);
  return [quantileSorted(samples, 0.025), quantileSorted(samples, 0.975)];
}

function quantileSorted(values: number[], quantile: number): number {
  if (values.length === 0) return 0;
  const index = Math.min(values.length - 1, Math.max(0, Math.floor(quantile * (values.length - 1))));
  return values[index];
}

const logFactorials = [0];

function logFactorial(value: number): number {
  for (let index = logFactorials.length; index <= value; index += 1) {
    logFactorials[index] = logFactorials[index - 1] + Math.log(index);
  }
  return logFactorials[value];
}

function logBinomialProbability(trials: number, successes: number): number {
  return logFactorial(trials) - logFactorial(successes) - logFactorial(trials - successes) - trials * Math.log(2);
}

function logSumExp(values: number[]): number {
  const max = Math.max(...values);
  if (!Number.isFinite(max)) return max;
  const total = values.reduce((sum, value) => sum + Math.exp(value - max), 0);
  return max + Math.log(total);
}

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function hashSeed(value: string): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function contextLevelSpecs(label: string, level: "file" | "symbol" | "span" | "line"): MetricSpec[] {
  return (["f1", "recall", "precision"] as const).map((measure) => ({
    key: `context.${label}.${measure}`,
    test: "permutation",
    unit: "number",
    value: (instance) => contextLevelMetric(instance, level, measure),
    aggregateValue: (instances) => contextLevelMetricPooled(instances, level, measure),
  }));
}

function contextAggregate(instance: ComparisonInstance, measure: "f1" | "recall" | "precision"): number | null {
  const file = contextLevelMetric(instance, "file", measure);
  const span = contextLevelMetric(instance, "span", measure);
  const line = contextLevelMetric(instance, "line", measure);
  const symbol = contextLevelMetric(instance, "symbol", measure);
  if (file === null || span === null || line === null || symbol === null) return null;
  return (file + span + line + symbol) / 4;
}

function contextLevelMetric(
  instance: ComparisonInstance,
  level: "file" | "symbol" | "span" | "line",
  measure: "f1" | "recall" | "precision",
): number | null {
  if (instance.artifacts?.evaluationStatus && instance.artifacts.evaluationStatus !== "valid") return null;
  const metric = instance.quality[level];
  const values = coveragePrecision(metric.predSize, metric.goldSize, metric.intersection);
  if (measure === "recall") return values.coverage;
  if (measure === "precision") return values.precision;
  return f1(values.coverage, values.precision);
}

function contextAggregatePooled(instances: ComparisonInstance[], measure: "f1" | "recall" | "precision"): number | null {
  if (instances.length === 0) return null;
  const values = instances
    .map((instance) => contextAggregate(instance, measure))
    .filter((value): value is number => value !== null);
  return values.length > 0 ? mean(values) : null;
}

function contextLevelMetricPooled(
  instances: ComparisonInstance[],
  level: "file" | "symbol" | "span" | "line",
  measure: "f1" | "recall" | "precision",
): number | null {
  if (instances.length === 0) return null;
  const values = instances
    .map((instance) => contextLevelMetric(instance, level, measure))
    .filter((value): value is number => value !== null);
  return values.length > 0 ? mean(values) : null;
}

function trajectoryLevelMetric(instance: ComparisonInstance, level: "file" | "symbol" | "span" | "line"): number | null {
  if (instance.artifacts?.evaluationStatus && instance.artifacts.evaluationStatus !== "valid") return null;
  if (!instance.evaluatedTrajectory) return null;
  const steps = (instance.evaluatedTrajectory?.steps ?? []).filter((step) => !step.isSkillRead);
  const values = steps
    .map((step) => step.coverage[level])
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  if (values.length === 0) return 0;
  return Math.min(Math.max(values[values.length - 1], 0), 1);
}

function trajectoryGoldFoundAggregate(instance: ComparisonInstance): number | null {
  const values = (["file", "span", "line", "symbol"] as const)
    .map((level) => trajectoryLevelMetric(instance, level))
    .filter((value): value is number => value !== null);
  return values.length > 0 ? mean(values) : null;
}

function trajectoryGoldFoundAggregatePooled(instances: ComparisonInstance[]): number | null {
  if (instances.length === 0) return null;
  const values = instances
    .map((instance) => trajectoryGoldFoundAggregate(instance))
    .filter((value): value is number => value !== null);
  return values.length > 0 ? mean(values) : null;
}

function fixOverlapMetric(instance: ComparisonInstance, measure: "recall" | "precision" | "f1"): number | null {
  const metric = instance.fixOverlap?.vsGold;
  if (!metric || metric.status !== "available") return null;
  return finiteOrNull(metric[measure]);
}

function fixOverlapMetricPooled(instances: ComparisonInstance[], measure: "recall" | "precision" | "f1"): number | null {
  const available = instances
    .map((instance) => instance.fixOverlap?.vsGold)
    .filter((metric): metric is NonNullable<NonNullable<ComparisonInstance["fixOverlap"]>["vsGold"]> => metric?.status === "available");
  if (available.length === 0) return null;

  const intersection = available.reduce((sum, metric) => sum + (metric.intersection ?? 0), 0);
  const goldSize = available.reduce((sum, metric) => sum + (metric.goldSize ?? 0), 0);
  const predSize = available.reduce((sum, metric) => sum + (metric.predSize ?? 0), 0);
  const recall = goldSize > 0 ? intersection / goldSize : 0;
  const precision = predSize > 0 ? intersection / predSize : 0;
  if (measure === "recall") return recall;
  if (measure === "precision") return precision;
  return f1(recall, precision);
}

function finiteOrNull(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function positiveFiniteOrNull(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : null;
}

function finiteOrZero(value: number | null | undefined): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function formatSignificanceValue(value: number, unit: SignificanceUnit): string {
  if (unit === "percent-point") return `${formatSignedFixed(value * 100, 1)} pts`;
  if (unit === "duration-ms") return formatDurationDelta(value / 1000);
  if (unit === "tokens") return `${value > 0 ? "+" : value < 0 ? "-" : ""}${formatCompactMagnitude(value)}`;
  if (unit === "currency") return `${value > 0 ? "+" : value < 0 ? "-" : ""}$${Math.abs(value).toFixed(2)}`;
  return formatSignedFixed(value, Math.abs(value) >= 10 ? 1 : 3);
}
