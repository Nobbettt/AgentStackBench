
import type { ComparisonCard, ComparisonInstance } from "@/data/comparisons";
import { formatCurrency, formatDurationMs, formatMetric, formatPatternMetric, formatPercent, formatTokens, sortBench } from "@/components/comparison/format";
import { getComparisonPair } from "@/components/comparison/format";
import {
  type ContextLevel,
  coveragePrecisionOrNull,
  f1,
  instanceContextAggregate,
  instanceTrajectoryGoldFound as sharedInstanceTrajectoryGoldFound,
  terminalTrajectoryCoverage as sharedTerminalTrajectoryCoverage,
} from "@/data/instance-metrics";
import type { ComparisonVariant, InstanceRow } from "@/components/comparison/types";

export function instanceContextF1(instance: ComparisonInstance | undefined): number | null {
  if (!instance) return null;
  return instanceContextAggregate(instance, "f1");
}

function instanceContextRecallPrecision(instance: ComparisonInstance | undefined): { recall: number; precision: number } | null {
  if (!instance) return null;
  const recall = instanceContextAggregate(instance, "recall");
  const precision = instanceContextAggregate(instance, "precision");
  if (recall === null || precision === null) return null;
  return { recall, precision };
}

function instanceTrajectoryGoldFound(instance: ComparisonInstance | undefined): number | null {
  return instance ? sharedInstanceTrajectoryGoldFound(instance) : null;
}

function formatTrajectoryLevel(instance: ComparisonInstance | undefined, level: ContextLevel) {
  const value = instance ? sharedTerminalTrajectoryCoverage(instance, level) : null;
  return {
    goldFound: value !== null ? formatMetric(value) : undefined,
  };
}

function formatContextLevelMetrics(metric: ComparisonInstance["quality"]["file"] | undefined) {
  const values = coveragePrecisionOrNull(metric?.predSize ?? 0, metric?.goldSize ?? 0, metric?.intersection ?? 0);
  if (!values) {
    return { recall: "—", precision: "—", f1: "—" };
  }
  return {
    recall: formatMetric(values.coverage),
    precision: formatMetric(values.precision),
    f1: formatMetric(f1(values.coverage, values.precision)),
  };
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
  const hasValidEvaluation = instance ? (instance.artifacts ? instance.artifacts.evaluationStatus === "valid" : true) : false;
  const fileF1 = metricF1(instance?.quality.file);
  const symbolF1 = metricF1(instance?.quality.symbol);
  const spanF1 = metricF1(instance?.quality.span);
  const lineF1 = metricF1(instance?.quality.line);
  const contextF1 = instanceContextF1(instance);
  const contextRecallPrecision = instanceContextRecallPrecision(instance);
  const trajectoryGoldFound = instanceTrajectoryGoldFound(instance);

  return {
    ...variant,
    contextF1: contextF1 !== null ? formatMetric(contextF1) : undefined,
    score: contextF1 !== null ? formatMetric(contextF1) : undefined,
    parameters: variant.parameters ?? [],
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
        contextRecall: contextRecallPrecision ? formatMetric(contextRecallPrecision.recall) : undefined,
        contextPrecision: contextRecallPrecision ? formatMetric(contextRecallPrecision.precision) : undefined,
        trajectoryGoldFound: trajectoryGoldFound !== null ? formatMetric(trajectoryGoldFound) : undefined,
        fileF1: hasValidEvaluation && instance && fileF1 !== null ? formatMetric(fileF1) : undefined,
        symbolF1: hasValidEvaluation && instance && symbolF1 !== null ? formatMetric(symbolF1) : undefined,
        spanF1: hasValidEvaluation && instance && spanF1 !== null ? formatMetric(spanF1) : undefined,
        avgLineF1: hasValidEvaluation && instance && lineF1 !== null ? formatMetric(lineF1) : undefined,
        contextLevels: hasValidEvaluation && instance
          ? {
              file: formatContextLevelMetrics(instance.quality.file),
              symbol: formatContextLevelMetrics(instance.quality.symbol),
              block: formatContextLevelMetrics(instance.quality.span),
              line: formatContextLevelMetrics(instance.quality.line),
            }
          : undefined,
        trajectoryContextLevels: hasValidEvaluation && instance
          ? {
              file: formatTrajectoryLevel(instance, "file"),
              symbol: formatTrajectoryLevel(instance, "symbol"),
              block: formatTrajectoryLevel(instance, "span"),
              line: formatTrajectoryLevel(instance, "line"),
            }
          : undefined,
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
        mcpToolCalls: typeof instance?.resources.mcpToolCalls === "number" ? String(instance.resources.mcpToolCalls) : undefined,
        successfulMcpToolCalls: typeof instance?.resources.successfulMcpToolCalls === "number" ? String(instance.resources.successfulMcpToolCalls) : undefined,
        commandExecutions: typeof instance?.resources.commandExecutions === "number" ? String(instance.resources.commandExecutions) : undefined,
        readToolCalls: typeof instance?.resources.readToolCalls === "number" ? String(instance.resources.readToolCalls) : undefined,
        editToolCalls: typeof instance?.resources.editToolCalls === "number" ? String(instance.resources.editToolCalls) : undefined,
        rawTraceEvents: typeof instance?.resources.rawTraceEvents === "number" ? String(instance.resources.rawTraceEvents) : undefined,
        rawAgentActions: typeof instance?.resources.rawAgentActions === "number" ? String(instance.resources.rawAgentActions) : undefined,
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

function metricF1(metric: ComparisonInstance["quality"]["file"] | undefined): number | null {
  const values = coveragePrecisionOrNull(metric?.predSize ?? 0, metric?.goldSize ?? 0, metric?.intersection ?? 0);
  return values ? f1(values.coverage, values.precision) : null;
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
