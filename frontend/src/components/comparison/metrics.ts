
import { completedRunRateForOutcome, completedRunsForOutcome, partialRunsForOutcome } from "@/data/comparisons";
import {
  deltaTone,
  formatAbsoluteMetricDelta,
  formatPercentDelta,
  parseCompactNumberMetric,
  parseCurrencyMetric,
  parseDurationMetric,
  parseFloatMetric,
  parsePercent,
} from "@/components/comparison/format";
import type {
  ComparisonVariant,
  DeltaDisplayMode,
  MetricDefinition,
  MetricDelta,
  OutcomeMetricName,
} from "@/components/comparison/types";

export const executionMetricDefinitions: MetricDefinition[] = [
  {
    key: "completedRunRate",
    label: "Completed Run Rate",
    explanation: "Fork-specific execution completion rate based on run records with status completed.",
    direction: "higher",
    value: (variant) => completedRunRateForOutcome(variant.results.outcome) ?? "—",
    parse: parsePercent,
  },
  {
    key: "patchProductionRate",
    label: "Patch Production Rate",
    explanation: "Rate of attempted tasks that produced a non-empty model patch.",
    direction: "higher",
    value: (variant) => variant.results.integrity?.patchProductionRate ?? "—",
    parse: parsePercent,
  },
  {
    key: "validEvaluationRate",
    label: "Valid Evaluation Rate",
    explanation: "Rate of attempted tasks with a valid evaluation row from the evaluator.",
    direction: "higher",
    value: (variant) => variant.results.integrity?.validEvaluationRate ?? "—",
    parse: parsePercent,
  },
];

export const resolutionMetricDefinitions: MetricDefinition[] = [
  {
    key: "officialPassAt1",
    label: "Pass@1",
    explanation: "Task-resolution rate computed through the benchmark-specific resolution harnesses.",
    direction: "higher",
    value: (variant) => variant.results.outcome.officialPassAt1 ?? "—",
    parse: parsePercent,
  },
  {
    key: "fixOverlapVsGoldF1",
    label: "Fix overlap F1",
    explanation: "Balanced overlap between model patch edit locations and gold patch edit locations.",
    direction: "higher",
    value: (variant) => variant.results.quality.fixOverlapVsGold?.f1 ?? "—",
    parse: parsePercent,
  },
  {
    key: "fixOverlapVsGoldRecall",
    label: "Fix overlap recall",
    explanation: "Share of gold patch edit locations covered by the model patch.",
    direction: "higher",
    value: (variant) => variant.results.quality.fixOverlapVsGold?.recall ?? "—",
    parse: parsePercent,
  },
  {
    key: "fixOverlapVsGoldPrecision",
    label: "Fix overlap precision",
    explanation: "Share of model patch edit locations that overlap gold patch edit locations.",
    direction: "higher",
    value: (variant) => variant.results.quality.fixOverlapVsGold?.precision ?? "—",
    parse: parsePercent,
  },
];

export const contextRetrievalMetricDefinitions: MetricDefinition[] = [
  {
    key: "contextF1",
    label: "Macro Context F1",
    explanation: "Macro-average final-context file/block/line/symbol F1 score, balancing final retained gold context and final viewed-context relevance.",
    direction: "higher",
    value: (variant) => variant.contextF1 ?? variant.score ?? "—",
    parse: parseFloatMetric,
  },
  {
    key: "trajectoryGoldFound",
    label: "Gold Found",
    explanation: "Macro-average terminal cumulative trajectory coverage across file/block/line/symbol levels.",
    direction: "higher",
    value: (variant) => variant.results.quality.trajectoryGoldFound ?? "—",
    parse: parseFloatMetric,
  },
  {
    key: "contextRecall",
    label: "Final Retained",
    explanation: "Macro-average file/block/line/symbol final-context recall: of the gold context, how much remained in the final retrieved context.",
    direction: "higher",
    value: (variant) => variant.results.quality.contextRecall ?? "—",
    parse: parseFloatMetric,
  },
  {
    key: "contextPrecision",
    label: "Viewed Gold",
    explanation: "Macro-average file/block/line/symbol retrieval precision: of the viewed context, how much was gold.",
    direction: "higher",
    value: (variant) => variant.results.quality.contextPrecision ?? "—",
    parse: parseFloatMetric,
  },
  {
    key: "fileF1",
    label: "File F1",
    explanation: "File-level retrieval F1.",
    direction: "higher",
    value: (variant) => variant.results.quality.fileF1 ?? variant.results.quality.fileCoverage ?? "—",
    parse: parseFloatMetric,
  },
  {
    key: "symbolF1",
    label: "Symbol F1",
    explanation: "Symbol-level retrieval F1.",
    direction: "higher",
    value: (variant) => variant.results.quality.symbolF1 ?? "—",
    parse: parseFloatMetric,
  },
  {
    key: "spanF1",
    label: "Span F1",
    explanation: "Span-level retrieval F1.",
    direction: "higher",
    value: (variant) => variant.results.quality.spanF1 ?? variant.results.quality.precision ?? "—",
    parse: parseFloatMetric,
  },
  {
    key: "avgLineF1",
    label: "Avg. Line F1",
    explanation: "Line-level retrieval F1.",
    direction: "higher",
    value: (variant) => variant.results.quality.avgLineF1 ?? variant.results.quality.editSuccess ?? "—",
    parse: parseFloatMetric,
  },
];

export const resourceMetricDefinitions: MetricDefinition[] = [
  {
    key: "averageSteps",
    label: "Average Scored Retrieval Steps",
    explanation: "Average number of inferred context retrieval steps scored by the evaluator per run.",
    direction: "lower",
    value: (variant) => variant.results.efficiency.averageSteps ?? "—",
    parse: parseFloatMetric,
  },
  {
    key: "averageDuration",
    label: "Average Scored Task Duration",
    explanation: "Average scored agent task runtime per run; excludes setup, bootstrap, validation, timed-out rows, and duration values that exceed the configured agent timeout.",
    direction: "lower",
    value: (variant) => variant.results.efficiency.averageDuration ?? variant.results.efficiency.avgDuration ?? "—",
    parse: parseDurationMetric,
  },
  {
    key: "totalTokens",
    label: "Avg Tokens / Run",
    explanation: "Average provider-reported tokens consumed per included run.",
    direction: "lower",
    value: (variant) => variant.results.efficiency.totalTokens ?? "—",
    parse: parseCompactNumberMetric,
  },
  {
    key: "rawTraceEvents",
    label: "Avg Raw Trace Events",
    explanation: "Average normalized trace rows exported to the per-instance Trace timeline per included run, including commands, tool uses, file changes, assistant messages, and todo updates.",
    direction: "neutral",
    value: (variant) => variant.results.efficiency.rawTraceEvents ?? "—",
    parse: parseFloatMetric,
  },
  {
    key: "rawAgentActions",
    label: "Avg Raw Agent Actions",
    explanation: "Average concrete agent actions per included run in the normalized trace: command executions, tool invocations, and file changes. Assistant messages, todo updates, and tool-result-only rows are excluded.",
    direction: "neutral",
    value: (variant) => variant.results.efficiency.rawAgentActions ?? "—",
    parse: parseFloatMetric,
  },
  {
    key: "mcpToolCalls",
    label: "MCP Calls",
    explanation: "Total MCP tool calls across included runs.",
    direction: "neutral",
    value: (variant) => variant.results.efficiency.mcpToolCalls ?? variant.results.efficiency.toolCalls ?? "—",
    parse: parseFloatMetric,
  },
  {
    key: "commandExecutions",
    label: "Shell Commands",
    explanation: "Total shell command executions observed in the agent trace.",
    direction: "neutral",
    value: (variant) => variant.results.efficiency.commandExecutions ?? "—",
    parse: parseFloatMetric,
  },
  {
    key: "estimatedCost",
    label: "Execution Cost",
    explanation: "Average per-run inference cost when metadata is available.",
    direction: "lower",
    value: (variant) => variant.results.efficiency.cost ?? "—",
    parse: parseCurrencyMetric,
  },
];

export function metricDelta(
  metric: MetricDefinition,
  baseline: ComparisonVariant,
  treatment: ComparisonVariant,
  displayMode: DeltaDisplayMode,
): MetricDelta | null {
  const baselineValue = metric.parse(metric.value(baseline));
  const treatmentValue = metric.parse(metric.value(treatment));
  if (baselineValue === null || treatmentValue === null) return null;

  const delta = treatmentValue - baselineValue;
  const percentDelta = baselineValue === 0 ? (delta === 0 ? 0 : null) : (delta / Math.abs(baselineValue)) * 100;
  return {
    delta,
    label: displayMode === "percent" ? (percentDelta === null ? "n/a" : formatPercentDelta(percentDelta)) : formatAbsoluteMetricDelta(metric, delta),
    tone: deltaTone(metric.direction, delta),
  };
}

export function outcomeDelta(
  metricName: OutcomeMetricName,
  baseline: ComparisonVariant,
  treatment: ComparisonVariant,
  displayMode: DeltaDisplayMode,
): MetricDelta {
  const baselineValue = metricName === "Completed"
    ? completedRunsForOutcome(baseline.results.outcome)
    : metricName === "Failures"
      ? baseline.results.outcome.failures
      : partialRunsForOutcome(baseline.results.outcome);
  const treatmentValue = metricName === "Completed"
    ? completedRunsForOutcome(treatment.results.outcome)
    : metricName === "Failures"
      ? treatment.results.outcome.failures
      : partialRunsForOutcome(treatment.results.outcome);
  const delta = treatmentValue - baselineValue;
  const percentDelta = baselineValue === 0 ? (delta === 0 ? 0 : null) : (delta / Math.abs(baselineValue)) * 100;
  return {
    delta,
    label: displayMode === "percent" ? (percentDelta === null ? "n/a" : formatPercentDelta(percentDelta)) : String(delta > 0 ? `+${delta}` : delta),
    tone: deltaTone(metricName === "Failures" ? "lower" : metricName === "Completed" ? "higher" : "neutral", delta),
  };
}
