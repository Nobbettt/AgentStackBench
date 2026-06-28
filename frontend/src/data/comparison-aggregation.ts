// Fork note: Modified by Norbert Laszlo on 2026-04-17 from upstream ContextBench.
// Summary of changes: aggregate fork-specific execution completion and integrity counters separately from official Pass@1.

import type { ComparisonCard, ComparisonInstance } from "@/data/comparisons";
import {
  type ContextLevel,
  CONTEXT_LEVELS,
  contextLevelMetric,
  f1,
  hasValidEvaluation,
  mean,
  terminalTrajectoryCoverage,
} from "@/data/instance-metrics";

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
  return String(Math.round(value));
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

function formatContextLevelMetrics(metrics: { coverage: number; precision: number; f1?: number; n?: number }) {
  return {
    recall: formatMetric(metrics.coverage),
    precision: formatMetric(metrics.precision),
    // Macro levels carry their own mean-of-f1s; recomputing from mean
    // coverage/precision (F1-of-means) is only correct for pooled totals.
    f1: formatMetric(metrics.f1 ?? f1(metrics.coverage, metrics.precision)),
    ...(metrics.n !== undefined ? { n: metrics.n } : {}),
  };
}

function macroContextLevelMetrics(instances: ComparisonInstance[], level: ContextLevel) {
  const values = (measure: "recall" | "precision" | "f1") =>
    instances
      .map((instance) => contextLevelMetric(instance, level, measure))
      .filter((value): value is number => value !== null);
  const f1Values = values("f1");
  return {
    coverage: mean(values("recall")) ?? 0,
    precision: mean(values("precision")) ?? 0,
    f1: mean(f1Values) ?? 0,
    n: f1Values.length,
  };
}

function macroTrajectoryContextLevel(instances: ComparisonInstance[], level: ContextLevel): number | null {
  return mean(
    instances
      .map((instance) => terminalTrajectoryCoverage(instance, level))
      .filter((value): value is number => value !== null),
  );
}

function formatTrajectoryContextLevel(value: number | null) {
  return {
    goldFound: value !== null ? formatMetric(value) : undefined,
  };
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
  const idSets = comparison.variants.map((variant) =>
    new Set(variantInstances(variant).filter((instance) => instanceMatches(instance, filters)).map((instance) => instance.instanceId)),
  );

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
  options: { preserveExistingTrajectoryMetrics?: boolean } = {},
): ComparisonCard["variants"][number] {
  const taskCount = filteredInstances.length;
  const qualityInstances = filteredInstances.filter(hasValidEvaluation);
  const qualityCount = qualityInstances.length;
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

  for (const instance of qualityInstances) {
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

  const macroLevels = {
    file: macroContextLevelMetrics(qualityInstances, "file"),
    symbol: macroContextLevelMetrics(qualityInstances, "symbol"),
    span: macroContextLevelMetrics(qualityInstances, "span"),
    line: macroContextLevelMetrics(qualityInstances, "line"),
  };
  const fileF1 = macroLevels.file.f1;
  const symbolF1 = macroLevels.symbol.f1;
  const spanF1 = macroLevels.span.f1;
  const lineF1 = macroLevels.line.f1;
  const contextRecall = mean(CONTEXT_LEVELS.map((level) => macroLevels[level].coverage)) ?? 0;
  const contextPrecision = mean(CONTEXT_LEVELS.map((level) => macroLevels[level].precision)) ?? 0;
  const contextF1 = mean(CONTEXT_LEVELS.map((level) => macroLevels[level].f1)) ?? 0;
  const trajectoryLevels = {
    file: macroTrajectoryContextLevel(qualityInstances, "file"),
    symbol: macroTrajectoryContextLevel(qualityInstances, "symbol"),
    span: macroTrajectoryContextLevel(qualityInstances, "span"),
    line: macroTrajectoryContextLevel(qualityInstances, "line"),
  };
  const trajectoryGoldFound = mean(
    CONTEXT_LEVELS
      .map((level) => trajectoryLevels[level])
      .filter((value): value is number => value !== null),
  );
  const hasTrajectoryCoverage = CONTEXT_LEVELS.some((level) => trajectoryLevels[level] !== null);
  const trajectoryGoldFoundValue = trajectoryGoldFound !== null
    ? formatMetric(trajectoryGoldFound)
    : options.preserveExistingTrajectoryMetrics
      ? variant.results.quality.trajectoryGoldFound
      : undefined;
  const trajectoryContextLevelsValue = hasTrajectoryCoverage
    ? {
        file: formatTrajectoryContextLevel(trajectoryLevels.file),
        symbol: formatTrajectoryContextLevel(trajectoryLevels.symbol),
        block: formatTrajectoryContextLevel(trajectoryLevels.span),
        line: formatTrajectoryContextLevel(trajectoryLevels.line),
      }
    : options.preserveExistingTrajectoryMetrics
      ? variant.results.quality.trajectoryContextLevels
      : undefined;

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
  const excludedDurationValues = filteredInstances.filter((instance) => instance.resources.durationStatus === "unavailable").length;
  const totalTokens = filteredInstances.reduce((sum, instance) => sum + (instance.resources.totalTokens ?? 0), 0);
  const inputTokens = filteredInstances.reduce((sum, instance) => sum + (instance.resources.inputTokens ?? 0), 0);
  const outputTokens = filteredInstances.reduce((sum, instance) => sum + (instance.resources.outputTokens ?? 0), 0);
  const cachedInputTokens = filteredInstances.reduce((sum, instance) => sum + (instance.resources.cachedInputTokens ?? 0), 0);
  const nonCachedInputTokens = filteredInstances.reduce((sum, instance) => sum + (instance.resources.nonCachedInputTokens ?? 0), 0);
  const averageTotalTokens = taskCount > 0 ? totalTokens / taskCount : 0;
  const averageInputTokens = taskCount > 0 ? inputTokens / taskCount : 0;
  const averageOutputTokens = taskCount > 0 ? outputTokens / taskCount : 0;
  const averageCachedInputTokens = taskCount > 0 ? cachedInputTokens / taskCount : 0;
  const averageNonCachedInputTokens = taskCount > 0 ? nonCachedInputTokens / taskCount : 0;
  const toolCalls = filteredInstances.reduce((sum, instance) => sum + (instance.resources.toolCalls ?? 0), 0);
  const mcpToolCalls = filteredInstances.reduce((sum, instance) => sum + (instance.resources.mcpToolCalls ?? 0), 0);
  const successfulMcpToolCalls = filteredInstances.reduce((sum, instance) => sum + (instance.resources.successfulMcpToolCalls ?? 0), 0);
  const commandExecutions = filteredInstances.reduce((sum, instance) => sum + (instance.resources.commandExecutions ?? 0), 0);
  const readToolCalls = filteredInstances.reduce((sum, instance) => sum + (instance.resources.readToolCalls ?? 0), 0);
  const editToolCalls = filteredInstances.reduce((sum, instance) => sum + (instance.resources.editToolCalls ?? 0), 0);
  const rawTraceEvents = filteredInstances.reduce((sum, instance) => sum + (instance.resources.rawTraceEvents ?? 0), 0);
  const rawAgentActions = filteredInstances.reduce((sum, instance) => sum + (instance.resources.rawAgentActions ?? 0), 0);
  const averageRawTraceEvents = taskCount > 0 ? rawTraceEvents / taskCount : 0;
  const averageRawAgentActions = taskCount > 0 ? rawAgentActions / taskCount : 0;
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
  const mcpUsage = aggregateMcpUsage(filteredInstances);

  return {
    ...variant,
    contextF1: qualityCount > 0 ? formatMetric(contextF1) : undefined,
    score: qualityCount > 0 ? formatMetric(contextF1) : undefined,
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
        contextF1: qualityCount > 0 ? formatMetric(contextF1) : undefined,
        contextRecall: qualityCount > 0 ? formatMetric(contextRecall) : undefined,
        contextPrecision: qualityCount > 0 ? formatMetric(contextPrecision) : undefined,
        trajectoryGoldFound: qualityCount > 0 ? trajectoryGoldFoundValue : undefined,
        fileF1: qualityCount > 0 ? formatMetric(fileF1) : undefined,
        symbolF1: qualityCount > 0 ? formatMetric(symbolF1) : undefined,
        spanF1: qualityCount > 0 ? formatMetric(spanF1) : undefined,
        avgLineF1: qualityCount > 0 ? formatMetric(lineF1) : undefined,
        contextLevels: qualityCount > 0
          ? {
              file: formatContextLevelMetrics(macroLevels.file),
              symbol: formatContextLevelMetrics(macroLevels.symbol),
              block: formatContextLevelMetrics(macroLevels.span),
              line: formatContextLevelMetrics(macroLevels.line),
            }
          : undefined,
        pooledContextLevels: qualityCount > 0
          ? {
              file: {
                ...formatContextLevelMetrics(fileMetrics),
                ...qualityTotals.file,
              },
              symbol: {
                ...formatContextLevelMetrics(symbolMetrics),
                ...qualityTotals.symbol,
              },
              block: {
                ...formatContextLevelMetrics(spanMetrics),
                ...qualityTotals.span,
              },
              line: {
                ...formatContextLevelMetrics(lineMetrics),
                ...qualityTotals.line,
              },
            }
          : undefined,
        trajectoryContextLevels: qualityCount > 0 ? trajectoryContextLevelsValue : undefined,
        fixOverlapVsGold: aggregateFixOverlapVsGold(filteredInstances),
      },
      efficiency: {
        efficiency: efficiency !== null ? formatMetric(efficiency) : undefined,
        redundancy: redundancy !== null ? formatMetric(redundancy) : undefined,
        usageDrop: usageDrop !== null ? formatMetric(usageDrop) : undefined,
        averageDuration: durationValues.length > 0 ? formatDurationMs(durationValues.reduce((sum, value) => sum + value, 0) / durationValues.length) : undefined,
        excludedDurationValues,
        averageSteps: averageSteps !== null ? formatPatternMetric(averageSteps) : undefined,
        avgLinesPerStep: avgLinesPerStep !== null ? formatPatternMetric(avgLinesPerStep) : undefined,
        totalTokens: averageTotalTokens > 0 ? formatTokens(averageTotalTokens) : undefined,
        inputTokens: averageInputTokens > 0 ? formatTokens(averageInputTokens) : undefined,
        outputTokens: averageOutputTokens > 0 ? formatTokens(averageOutputTokens) : undefined,
        cachedInputTokens: averageCachedInputTokens > 0 ? formatTokens(averageCachedInputTokens) : undefined,
        nonCachedInputTokens: averageNonCachedInputTokens > 0 ? formatTokens(averageNonCachedInputTokens) : undefined,
        cachedInputShare: inputTokens > 0 ? formatPercent(cachedInputTokens / inputTokens) : null,
        toolCalls: String(toolCalls),
        mcpToolCalls: String(mcpToolCalls),
        successfulMcpToolCalls: String(successfulMcpToolCalls),
        commandExecutions: String(commandExecutions),
        readToolCalls: String(readToolCalls),
        editToolCalls: String(editToolCalls),
        rawTraceEvents: formatPatternMetric(averageRawTraceEvents),
        rawAgentActions: formatPatternMetric(averageRawAgentActions),
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
      mcp: mcpUsage,
    },
    instances: filteredInstances,
  };
}

function aggregateMcpUsage(instances: ComparisonInstance[]): NonNullable<ComparisonCard["variants"][number]["results"]["mcp"]> {
  const availableTools = new Set<string>();
  const byTool: Record<string, { calls: number; successfulCalls: number }> = {};
  const totals = {
    toolCalls: 0,
    successfulToolCalls: 0,
    callsWithResults: 0,
    meaningfulCalls: 0,
    callsWithFinalContextOverlap: 0,
    callsWithPatchOverlap: 0,
    callsWithFollowupOnReturnedPath: 0,
    returnedPathCount: 0,
  };
  let instancesWithMcpCalls = 0;
  let instancesWithMeaningfulMcpUse = 0;

  for (const instance of instances) {
    const mcp = instance.mcp;
    if (!mcp) continue;
    for (const tool of mcp.availableTools ?? []) {
      if (tool.trim()) availableTools.add(tool.trim());
    }
    for (const key of Object.keys(totals) as Array<keyof typeof totals>) {
      totals[key] += mcp[key] ?? 0;
    }
    if ((mcp.toolCalls ?? 0) > 0) instancesWithMcpCalls += 1;
    if ((mcp.meaningfulCalls ?? 0) > 0) instancesWithMeaningfulMcpUse += 1;
    for (const entry of mcp.byTool ?? []) {
      if (!entry.name.trim()) continue;
      const aggregate = byTool[entry.name] ?? { calls: 0, successfulCalls: 0 };
      aggregate.calls += entry.calls ?? 0;
      aggregate.successfulCalls += entry.successfulCalls ?? 0;
      byTool[entry.name] = aggregate;
    }
  }

  return {
    availableTools: Array.from(availableTools).sort((left, right) => left.localeCompare(right)),
    ...totals,
    instancesWithMcpCalls,
    instancesWithMeaningfulMcpUse,
    byTool: Object.entries(byTool)
      .sort((left, right) => left[0].localeCompare(right[0]))
      .map(([name, values]) => ({
        name,
        calls: values.calls,
        successfulCalls: values.successfulCalls,
      })),
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
      { preserveExistingTrajectoryMetrics: selectedInstanceIds.size === comparison.tasks },
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
