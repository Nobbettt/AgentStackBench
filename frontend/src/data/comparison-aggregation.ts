// Fork note: Modified by Norbert Laszlo on 2026-04-17 from upstream ContextBench.
// Summary of changes: aggregate fork-specific execution completion and integrity counters separately from official Pass@1.

import type { ComparisonCard, ComparisonInstance } from "@/data/comparisons";

export type ComparisonFilters = {
  benches: string[];
  languages: string[];
};

const BENCH_ORDER = ["Verified", "Pro", "Poly", "Multi"];

function uniqueSorted(values: Iterable<string>): string[] {
  return Array.from(new Set(values)).sort((left, right) => left.localeCompare(right));
}

function sortBenches(values: Iterable<string>): string[] {
  return Array.from(new Set(values)).sort((left, right) => {
    const leftIndex = BENCH_ORDER.indexOf(left);
    const rightIndex = BENCH_ORDER.indexOf(right);
    if (leftIndex >= 0 || rightIndex >= 0) {
      return (leftIndex >= 0 ? leftIndex : Number.MAX_SAFE_INTEGER) - (rightIndex >= 0 ? rightIndex : Number.MAX_SAFE_INTEGER);
    }
    return left.localeCompare(right);
  });
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatMetric(value: number): string {
  return value.toFixed(3);
}

function formatPatternMetric(value: number): string {
  return value.toFixed(2);
}

function formatDurationMs(value: number): string {
  const totalSeconds = Math.round(value / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;

  if (hours > 0) {
    return `${hours}h ${remainingMinutes.toString().padStart(2, "0")}m`;
  }
  return `${minutes}m ${seconds.toString().padStart(2, "0")}s`;
}

function formatTokens(value: number): string {
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(2)}M`;
  }
  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(0)}K`;
  }
  return String(value);
}

function formatCurrency(value: number): string {
  return `$${value.toFixed(2)}`;
}

function coveragePrecision(predSize: number, goldSize: number, intersection: number) {
  return {
    coverage: goldSize > 0 ? intersection / goldSize : 1,
    precision: predSize > 0 ? intersection / predSize : 1,
  };
}

function f1(coverage: number, precision: number): number {
  const denominator = coverage + precision;
  return denominator === 0 ? 0 : (2 * coverage * precision) / denominator;
}

function mean(values: number[]): number | null {
  return values.length > 0 ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

function variantInstances(variant: ComparisonCard["variants"][number]): ComparisonInstance[] {
  return variant.instances ?? [];
}

function countCompletedRuns(instances: ComparisonInstance[]): number {
  return instances.filter((instance) => instance.outcome.status === "completed").length;
}

function countPartialRuns(instances: ComparisonInstance[]): number {
  return instances.filter((instance) => instance.outcome.status === "partial").length;
}

function countFailures(instances: ComparisonInstance[]): number {
  return instances.filter((instance) => !["completed", "partial", "skipped"].includes(instance.outcome.status)).length;
}

function parseMetricValue(value: string | undefined): number | null {
  if (!value) {
    return null;
  }

  const parsed = Number.parseFloat(value.replace("%", "").replace("$", ""));
  return Number.isFinite(parsed) ? parsed : null;
}

function instanceMatches(instance: ComparisonInstance, filters: ComparisonFilters): boolean {
  return filters.languages.includes(instance.language) && filters.benches.includes(instance.bench);
}

function intersectInstanceIds(comparison: ComparisonCard, filters: ComparisonFilters): Set<string> | null {
  const idSets = comparison.variants
    .map((variant) =>
      new Set(variantInstances(variant).filter((instance) => instanceMatches(instance, filters)).map((instance) => instance.instanceId)),
    )
    .filter((idSet) => idSet.size > 0);

  if (idSets.length === 0) {
    return new Set<string>();
  }

  const [first, ...rest] = idSets;
  const intersection = new Set(first);
  for (const idSet of rest) {
    for (const instanceId of Array.from(intersection)) {
      if (!idSet.has(instanceId)) {
        intersection.delete(instanceId);
      }
    }
  }

  return intersection;
}

function aggregateVariant(
  variant: ComparisonCard["variants"][number],
  filteredInstances: ComparisonInstance[],
): ComparisonCard["variants"][number] {
  const taskCount = filteredInstances.length;
  const hasArtifactData = filteredInstances.some((instance) => instance.artifacts);
  const completedRuns = countCompletedRuns(filteredInstances);
  const partialRuns = countPartialRuns(filteredInstances);
  const failures = countFailures(filteredInstances);
  const patchProducingRuns = filteredInstances.filter((instance) => instance.artifacts?.hasModelPatch).length;
  const convertedPredictions = filteredInstances.filter((instance) => instance.artifacts?.hasPrediction).length;
  const validEvaluations = filteredInstances.filter((instance) => instance.artifacts?.evaluationStatus === "valid").length;
  const resolutionStatuses = filteredInstances.map((instance) => instance.artifacts?.resolutionStatus);
  const hasCompleteResolutionData =
    taskCount > 0 &&
    resolutionStatuses.every((status) => status === "resolved" || status === "unresolved");
  const resolvedTasks = filteredInstances.filter((instance) => instance.artifacts?.resolutionStatus === "resolved").length;

  const qualityTotals = {
    file: { intersection: 0, goldSize: 0, predSize: 0 },
    symbol: { intersection: 0, goldSize: 0, predSize: 0 },
    span: { intersection: 0, goldSize: 0, predSize: 0 },
    line: { intersection: 0, goldSize: 0, predSize: 0 },
  };

  for (const instance of filteredInstances) {
    for (const granularity of ["file", "symbol", "span", "line"] as const) {
      qualityTotals[granularity].intersection += instance.quality[granularity].intersection;
      qualityTotals[granularity].goldSize += instance.quality[granularity].goldSize;
      qualityTotals[granularity].predSize += instance.quality[granularity].predSize;
    }
  }

  const fileMetrics = coveragePrecision(
    qualityTotals.file.predSize,
    qualityTotals.file.goldSize,
    qualityTotals.file.intersection,
  );
  const symbolMetrics = coveragePrecision(
    qualityTotals.symbol.predSize,
    qualityTotals.symbol.goldSize,
    qualityTotals.symbol.intersection,
  );
  const spanMetrics = coveragePrecision(
    qualityTotals.span.predSize,
    qualityTotals.span.goldSize,
    qualityTotals.span.intersection,
  );
  const lineMetrics = coveragePrecision(
    qualityTotals.line.predSize,
    qualityTotals.line.goldSize,
    qualityTotals.line.intersection,
  );

  const fileF1 = f1(fileMetrics.coverage, fileMetrics.precision);
  const symbolF1 = f1(symbolMetrics.coverage, symbolMetrics.precision);
  const spanF1 = f1(spanMetrics.coverage, spanMetrics.precision);
  const lineF1 = f1(lineMetrics.coverage, lineMetrics.precision);
  const contextF1 = (fileF1 + symbolF1 + spanF1) / 3;

  const efficiency = mean(
    filteredInstances
      .map((instance) => instance.trajectory.efficiency)
      .filter((value): value is number => typeof value === "number"),
  );
  const redundancy = mean(
    filteredInstances
      .map((instance) => instance.trajectory.redundancy)
      .filter((value): value is number => typeof value === "number"),
  );
  const usageDrop = mean(
    filteredInstances
      .map((instance) => instance.trajectory.usageDrop)
      .filter((value): value is number => typeof value === "number"),
  );
  const averageSteps = mean(
    filteredInstances
      .map((instance) => instance.trajectory.steps)
      .filter((value): value is number => typeof value === "number"),
  );
  const totalWeightedLines = filteredInstances.reduce((sum, instance) => {
    if (typeof instance.trajectory.linesPerStep !== "number" || typeof instance.trajectory.steps !== "number") {
      return sum;
    }
    return sum + instance.trajectory.linesPerStep * instance.trajectory.steps;
  }, 0);
  const totalSteps = filteredInstances.reduce((sum, instance) => {
    return sum + (typeof instance.trajectory.steps === "number" ? instance.trajectory.steps : 0);
  }, 0);
  const avgLinesPerStep = totalSteps > 0 ? totalWeightedLines / totalSteps : null;

  const durationValues = filteredInstances
    .map((instance) => instance.resources.durationMs)
    .filter((value): value is number => typeof value === "number" && value > 0);
  const totalTokens = filteredInstances.reduce((sum, instance) => sum + (instance.resources.totalTokens ?? 0), 0);
  const toolCalls = filteredInstances.reduce((sum, instance) => sum + (instance.resources.toolCalls ?? 0), 0);
  const costValues = filteredInstances
    .map((instance) => instance.resources.costUsd)
    .filter((value): value is number => typeof value === "number");

  const totalSkillInvocations = filteredInstances.reduce(
    (sum, instance) => sum + (instance.skills?.totalInvocations ?? 0),
    0,
  );
  const skillCounts: Record<string, number> = {};
  for (const instance of filteredInstances) {
    for (const entry of instance.skills?.byType ?? []) {
      skillCounts[entry.name] = (skillCounts[entry.name] ?? 0) + entry.count;
    }
  }
  const totalToolInvocations = filteredInstances.reduce(
    (sum, instance) => sum + (instance.tools?.totalInvocations ?? 0),
    0,
  );
  const toolCounts: Record<string, number> = {};
  for (const instance of filteredInstances) {
    for (const entry of instance.tools?.byType ?? []) {
      toolCounts[entry.name] = (toolCounts[entry.name] ?? 0) + entry.count;
    }
  }

  return {
    ...variant,
    contextF1: taskCount > 0 ? formatMetric(contextF1) : undefined,
    score: taskCount > 0 ? formatMetric(contextF1) : undefined,
    results: {
      outcome: {
        completedRuns,
        partialRuns,
        failures,
        finishedRuns: completedRuns + partialRuns,
        expectedTasks: taskCount,
        attemptedTasks: taskCount,
        completedRunRate: taskCount > 0 ? formatPercent(completedRuns / taskCount) : "—",
        officialPassAt1: hasCompleteResolutionData ? formatPercent(resolvedTasks / taskCount) : null,
        metricType: "execution_status",
        comparableToOfficialLeaderboard: false,
      },
      integrity: {
        patchProducingRuns,
        convertedPredictions,
        validEvaluations,
        resolvedTasks,
        patchProductionRate: hasArtifactData && taskCount > 0 ? formatPercent(patchProducingRuns / taskCount) : "—",
        convertedPredictionRate: hasArtifactData && taskCount > 0 ? formatPercent(convertedPredictions / taskCount) : "—",
        validEvaluationRate: hasArtifactData && taskCount > 0 ? formatPercent(validEvaluations / taskCount) : "—",
      },
      quality: {
        contextF1: taskCount > 0 ? formatMetric(contextF1) : undefined,
        fileF1: taskCount > 0 ? formatMetric(fileF1) : undefined,
        symbolF1: taskCount > 0 ? formatMetric(symbolF1) : undefined,
        spanF1: taskCount > 0 ? formatMetric(spanF1) : undefined,
        avgLineF1: taskCount > 0 ? formatMetric(lineF1) : undefined,
        fixOverlapVsGold: aggregateFixOverlapVsGold(filteredInstances),
      },
      efficiency: {
        efficiency: efficiency !== null ? formatMetric(efficiency) : undefined,
        redundancy: redundancy !== null ? formatMetric(redundancy) : undefined,
        usageDrop: usageDrop !== null ? formatMetric(usageDrop) : undefined,
        averageDuration: durationValues.length > 0 ? formatDurationMs(durationValues.reduce((sum, value) => sum + value, 0) / durationValues.length) : undefined,
        averageSteps: averageSteps !== null ? formatPatternMetric(averageSteps) : undefined,
        avgLinesPerStep: avgLinesPerStep !== null ? formatPatternMetric(avgLinesPerStep) : undefined,
        totalTokens: totalTokens > 0 ? formatTokens(totalTokens) : undefined,
        toolCalls: String(toolCalls),
        cost: taskCount > 0 && costValues.length === taskCount ? formatCurrency(costValues.reduce((sum, value) => sum + value, 0) / costValues.length) : undefined,
      },
      skills: {
        totalInvocations: totalSkillInvocations,
        averageInvocationsPerRun: taskCount > 0 ? Number((totalSkillInvocations / taskCount).toFixed(2)) : 0,
        byType: Object.entries(skillCounts)
          .sort((left, right) => left[0].localeCompare(right[0]))
          .map(([name, count]) => ({
            name,
            averagePerRun: taskCount > 0 ? Number((count / taskCount).toFixed(2)) : 0,
          })),
      },
      tools: {
        totalInvocations: totalToolInvocations,
        averageInvocationsPerRun: taskCount > 0 ? Number((totalToolInvocations / taskCount).toFixed(2)) : 0,
        byType: Object.entries(toolCounts)
          .sort((left, right) => left[0].localeCompare(right[0]))
          .map(([name, count]) => ({
            name,
            averagePerRun: taskCount > 0 ? Number((count / taskCount).toFixed(2)) : 0,
          })),
      },
    },
    instances: filteredInstances,
  };
}

function aggregateFixOverlapVsGold(
  instances: ComparisonInstance[],
): ComparisonCard["variants"][number]["results"]["quality"]["fixOverlapVsGold"] {
  const metrics = instances
    .map((instance) => instance.fixOverlap?.vsGold)
    .filter((metric): metric is NonNullable<NonNullable<ComparisonInstance["fixOverlap"]>["vsGold"]> => Boolean(metric));
  if (metrics.length === 0) return undefined;

  const available = metrics.filter((metric) => metric?.status === "available");
  if (available.length === 0) {
    return {
      status: "unavailable",
      recall: null,
      precision: null,
      f1: null,
      availableInstances: 0,
      unavailableInstances: metrics.length,
    };
  }

  const intersection = available.reduce((sum, metric) => sum + (metric?.intersection ?? 0), 0);
  const goldSize = available.reduce((sum, metric) => sum + (metric?.goldSize ?? 0), 0);
  const predSize = available.reduce((sum, metric) => sum + (metric?.predSize ?? 0), 0);
  const recall = goldSize > 0 ? intersection / goldSize : 0;
  const precision = predSize > 0 ? intersection / predSize : 0;

  return {
    status: "available",
    recall: formatPercent(recall),
    precision: formatPercent(precision),
    f1: formatPercent(f1(recall, precision)),
    availableInstances: available.length,
    unavailableInstances: metrics.length - available.length,
  };
}

export function comparisonHasInstanceData(comparison: ComparisonCard): boolean {
  return comparison.variants.every((variant) => variantInstances(variant).length > 0);
}

export function getAvailableLanguages(comparison: ComparisonCard): string[] {
  return uniqueSorted(
    comparison.variants.flatMap((variant) => variantInstances(variant).map((instance) => instance.language).filter(Boolean)),
  );
}

export function getAvailableBenches(comparison: ComparisonCard): string[] {
  return sortBenches(
    comparison.variants.flatMap((variant) => variantInstances(variant).map((instance) => instance.bench).filter(Boolean)),
  );
}

export function buildFilteredComparison(comparison: ComparisonCard, filters: ComparisonFilters): ComparisonCard {
  if (!comparisonHasInstanceData(comparison)) {
    return comparison;
  }

  const selectedInstanceIds = intersectInstanceIds(comparison, filters);
  if (!selectedInstanceIds) {
    return comparison;
  }

  const filteredVariants = comparison.variants.map((variant) =>
    aggregateVariant(
      variant,
      variantInstances(variant).filter((instance) => selectedInstanceIds.has(instance.instanceId)),
    ),
  );

  const filteredTaskCount = selectedInstanceIds.size;
  const benchCounts = Object.fromEntries(
    Object.entries(
      filteredVariants[0]?.instances?.reduce<Record<string, number>>((counts, instance) => {
        counts[instance.bench] = (counts[instance.bench] ?? 0) + 1;
        return counts;
      }, {}) ?? {},
    ).sort((left, right) => left[0].localeCompare(right[0])),
  );

  const topContextF1 = filteredVariants
    .map((variant) => parseMetricValue(variant.contextF1 ?? variant.score))
    .filter((value): value is number => value !== null);

  return {
    ...comparison,
    tasks: filteredTaskCount,
    contextF1: topContextF1.length > 0 ? formatMetric(Math.max(...topContextF1)) : undefined,
    score: topContextF1.length > 0 ? formatMetric(Math.max(...topContextF1)) : undefined,
    fixOverlapBetweenVariants: filteredTaskCount === comparison.tasks ? comparison.fixOverlapBetweenVariants : undefined,
    taskSet: {
      count: filteredTaskCount,
      benchCounts,
      hash: filteredTaskCount === comparison.tasks ? comparison.taskSet?.hash : undefined,
      sourceDatasetCount: filteredTaskCount === comparison.tasks ? comparison.taskSet?.sourceDatasetCount : undefined,
      selectionKind: filteredTaskCount === comparison.tasks ? comparison.taskSet?.selectionKind : undefined,
    },
    variants: filteredVariants,
  };
}
