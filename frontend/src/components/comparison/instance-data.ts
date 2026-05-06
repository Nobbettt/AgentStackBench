
import type { ComparisonCard, ComparisonInstance } from "@/data/comparisons";
import { coveragePrecision, f1, formatCurrency, formatDurationMs, formatMetric, formatPatternMetric, formatPercent, formatTokens, sortBench } from "@/components/comparison/format";
import { getComparisonPair } from "@/components/comparison/format";
import type { ComparisonVariant, InstanceRow } from "@/components/comparison/types";

export function instanceContextF1(instance: ComparisonInstance | undefined): number | null {
  if (!instance || (instance.artifacts && instance.artifacts.evaluationStatus !== "valid")) return null;
  const fileMetrics = coveragePrecision(instance.quality.file.predSize, instance.quality.file.goldSize, instance.quality.file.intersection);
  const symbolMetrics = coveragePrecision(instance.quality.symbol.predSize, instance.quality.symbol.goldSize, instance.quality.symbol.intersection);
  const spanMetrics = coveragePrecision(instance.quality.span.predSize, instance.quality.span.goldSize, instance.quality.span.intersection);
  return (f1(fileMetrics.coverage, fileMetrics.precision) + f1(symbolMetrics.coverage, symbolMetrics.precision) + f1(spanMetrics.coverage, spanMetrics.precision)) / 3;
}

export function buildInstanceRows(comparison: ComparisonCard): InstanceRow[] {
  const comparisonPair = getComparisonPair(comparison);
  if (comparisonPair) {
    const baselineMap = new Map(comparisonPair.baseline.instances?.map((instance) => [instance.instanceId, instance]) ?? []);
    const treatmentMap = new Map(comparisonPair.treatment.instances?.map((instance) => [instance.instanceId, instance]) ?? []);
    return Array.from(new Set([...baselineMap.keys(), ...treatmentMap.keys()]))
      .map((instanceId): InstanceRow | null => {
        const baseline = baselineMap.get(instanceId);
        const treatment = treatmentMap.get(instanceId);
        const source = baseline ?? treatment;
        return source ? {
          instanceId,
          originalInstanceId: source.originalInstanceId,
          bench: source.bench,
          language: source.language,
          baseline,
          treatment,
        } : null;
      })
      .filter((row): row is InstanceRow => row !== null)
      .sort(sortInstanceRows);
  }

  return (comparison.variants[0]?.instances ?? [])
    .map((instance) => ({
      instanceId: instance.instanceId,
      originalInstanceId: instance.originalInstanceId,
      bench: instance.bench,
      language: instance.language,
      baseline: instance,
    }))
    .sort(sortInstanceRows);
}

function sortInstanceRows(left: InstanceRow, right: InstanceRow): number {
  const benchOrder = sortBench(left.bench, right.bench);
  if (benchOrder !== 0) return benchOrder;
  const languageOrder = left.language.localeCompare(right.language);
  if (languageOrder !== 0) return languageOrder;
  return left.instanceId.localeCompare(right.instanceId);
}

export function buildInstanceVariant(variant: ComparisonVariant, instance: ComparisonInstance | undefined): ComparisonVariant {
  const status = instance?.outcome.status ?? "";
  const completedRuns = status === "completed" ? 1 : 0;
  const partialRuns = status === "partial" ? 1 : 0;
  const failures = status && !["completed", "partial", "skipped"].includes(status) ? 1 : 0;
  const hasArtifactData = Boolean(instance?.artifacts);
  const hasValidEvaluation = instance?.artifacts ? instance.artifacts.evaluationStatus === "valid" : true;
  const fileF1 = metricF1(instance?.quality.file);
  const symbolF1 = metricF1(instance?.quality.symbol);
  const spanF1 = metricF1(instance?.quality.span);
  const lineF1 = metricF1(instance?.quality.line);
  const contextF1 = instanceContextF1(instance);

  return {
    ...variant,
    contextF1: contextF1 !== null ? formatMetric(contextF1) : undefined,
    score: contextF1 !== null ? formatMetric(contextF1) : undefined,
    parameters: [],
    results: {
      outcome: {
        completedRuns,
        partialRuns,
        failures,
        finishedRuns: completedRuns + partialRuns,
        expectedTasks: status ? 1 : 0,
        attemptedTasks: status ? 1 : 0,
        completedRunRate: status ? formatPercent(completedRuns) : "—",
        officialPassAt1:
          instance?.artifacts?.resolutionStatus === "resolved" || instance?.artifacts?.resolutionStatus === "unresolved"
            ? formatPercent(instance.artifacts.resolutionStatus === "resolved" ? 1 : 0)
            : null,
      },
      integrity: {
        patchProducingRuns: instance?.artifacts?.hasModelPatch ? 1 : 0,
        convertedPredictions: instance?.artifacts?.hasPrediction ? 1 : 0,
        validEvaluations: hasValidEvaluation ? 1 : 0,
        resolvedTasks: instance?.artifacts?.resolutionStatus === "resolved" ? 1 : 0,
        patchProductionRate: hasArtifactData && status ? formatPercent(instance?.artifacts?.hasModelPatch ? 1 : 0) : "—",
        convertedPredictionRate: hasArtifactData && status ? formatPercent(instance?.artifacts?.hasPrediction ? 1 : 0) : "—",
        validEvaluationRate: hasArtifactData && status ? formatPercent(hasValidEvaluation ? 1 : 0) : "—",
      },
      quality: {
        contextF1: contextF1 !== null ? formatMetric(contextF1) : undefined,
        fileF1: hasValidEvaluation && instance ? formatMetric(fileF1) : undefined,
        symbolF1: hasValidEvaluation && instance ? formatMetric(symbolF1) : undefined,
        spanF1: hasValidEvaluation && instance ? formatMetric(spanF1) : undefined,
        avgLineF1: hasValidEvaluation && instance ? formatMetric(lineF1) : undefined,
        fixOverlapVsGold: fixOverlapSummaryFromInstance(instance),
      },
      efficiency: {
        efficiency: typeof instance?.trajectory.efficiency === "number" ? formatMetric(instance.trajectory.efficiency) : undefined,
        redundancy: typeof instance?.trajectory.redundancy === "number" ? formatMetric(instance.trajectory.redundancy) : undefined,
        usageDrop: typeof instance?.trajectory.usageDrop === "number" ? formatMetric(instance.trajectory.usageDrop) : undefined,
        averageDuration: typeof instance?.resources.durationMs === "number" && instance.resources.durationMs > 0 ? formatDurationMs(instance.resources.durationMs) : undefined,
        averageSteps: typeof instance?.trajectory.steps === "number" ? formatPatternMetric(instance.trajectory.steps) : undefined,
        avgLinesPerStep: typeof instance?.trajectory.linesPerStep === "number" ? formatPatternMetric(instance.trajectory.linesPerStep) : undefined,
        totalTokens: typeof instance?.resources.totalTokens === "number" ? formatTokens(instance.resources.totalTokens) : undefined,
        toolCalls: typeof instance?.resources.toolCalls === "number" ? String(instance.resources.toolCalls) : undefined,
        cost: typeof instance?.resources.costUsd === "number" ? formatCurrency(instance.resources.costUsd) : undefined,
      },
      skills: {
        totalInvocations: instance?.skills?.totalInvocations ?? 0,
        averageInvocationsPerRun: instance?.skills?.totalInvocations ?? 0,
        byType: (instance?.skills?.byType ?? []).map((entry) => ({ name: entry.name, averagePerRun: entry.count })),
      },
      tools: {
        totalInvocations: instance?.tools?.totalInvocations ?? 0,
        averageInvocationsPerRun: instance?.tools?.totalInvocations ?? 0,
        byType: (instance?.tools?.byType ?? []).map((entry) => ({ name: entry.name, averagePerRun: entry.count })),
      },
    },
    instances: instance ? [instance] : [],
  };
}

function fixOverlapSummaryFromInstance(instance: ComparisonInstance | undefined): ComparisonVariant["results"]["quality"]["fixOverlapVsGold"] {
  const metric = instance?.fixOverlap?.vsGold;
  if (!metric) return undefined;
  const available = metric.status === "available";
  return {
    status: metric.status,
    reason: metric.reason,
    recall: available && typeof metric.recall === "number" ? formatPercent(metric.recall) : null,
    precision: available && typeof metric.precision === "number" ? formatPercent(metric.precision) : null,
    f1: available && typeof metric.f1 === "number" ? formatPercent(metric.f1) : null,
    availableInstances: available ? 1 : 0,
    unavailableInstances: available ? 0 : 1,
  };
}

function metricF1(metric: ComparisonInstance["quality"]["file"] | undefined): number {
  const values = coveragePrecision(metric?.predSize ?? 0, metric?.goldSize ?? 0, metric?.intersection ?? 0);
  return f1(values.coverage, values.precision);
}

export function buildInstanceComparison(comparison: ComparisonCard, row: InstanceRow): ComparisonCard {
  const comparisonPair = getComparisonPair(comparison);
  const variants = comparisonPair
    ? [buildInstanceVariant(comparisonPair.baseline, row.baseline), buildInstanceVariant(comparisonPair.treatment, row.treatment)]
    : comparison.variants.slice(0, 1).map((variant) => buildInstanceVariant(variant, row.baseline));
  return {
    ...comparison,
    id: row.instanceId,
    summary: row.originalInstanceId ? `Original issue: ${row.originalInstanceId}` : comparison.summary,
    taskSet: { count: 1, benchCounts: { [row.bench]: 1 } },
    tasks: 1,
    contextF1: variants.map((variant) => variant.contextF1).find(Boolean),
    score: variants.map((variant) => variant.score).find(Boolean),
    fixOverlapBetweenVariants: undefined,
    variants,
  };
}
