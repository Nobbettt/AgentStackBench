// SPDX-License-Identifier: Apache-2.0

import type { ComparisonInstance } from "@/data/comparisons";
import {
  type ContextLevel,
  contextLevelMetric,
  instanceContextAggregate,
  instanceTrajectoryGoldFound,
} from "@/data/instance-metrics";

export type CorrelationMetricGroup = "Context Quality" | "Trajectory" | "Resource Usage" | "Task Properties";

export const CORRELATION_METRIC_GROUPS: CorrelationMetricGroup[] = [
  "Context Quality",
  "Trajectory",
  "Resource Usage",
  "Task Properties",
];

export type CorrelationMetric = {
  id: string;
  label: string;
  group: CorrelationMetricGroup;
  explanation: string;
  // Heavily right-skewed metrics (tokens, duration, cost, repo size) read
  // better on a log axis, so they opt in as the default scale.
  preferLog: boolean;
  format: (value: number) => string;
  extract: (instance: ComparisonInstance) => number | null;
};

// Restrict dynamic resource lookups to numeric fields so a typo or a
// status/boolean field name fails to compile instead of silently extracting nulls.
type NumericResourceField = {
  [K in keyof ComparisonInstance["resources"]]-?: NonNullable<ComparisonInstance["resources"][K]> extends number ? K : never;
}[keyof ComparisonInstance["resources"]];

function finiteOrNull(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function formatRatio(value: number): string {
  return value.toFixed(3);
}

function formatCount(value: number): string {
  return Math.round(value).toLocaleString("en-US");
}

function formatCompact(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return formatCount(value);
}

function formatDuration(value: number): string {
  const totalSeconds = Math.round(value / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  if (hours > 0) return `${hours}h ${remainingMinutes.toString().padStart(2, "0")}m`;
  if (minutes > 0) return `${minutes}m ${seconds.toString().padStart(2, "0")}s`;
  return `${seconds}s`;
}

function formatUsd(value: number): string {
  return `$${value.toFixed(2)}`;
}

function formatBytes(value: number): string {
  if (value >= 1_073_741_824) return `${(value / 1_073_741_824).toFixed(2)} GB`;
  if (value >= 1_048_576) return `${(value / 1_048_576).toFixed(1)} MB`;
  if (value >= 1_024) return `${(value / 1_024).toFixed(1)} KB`;
  return `${formatCount(value)} B`;
}

function contextLevelF1Metric(level: ContextLevel, label: string): CorrelationMetric {
  return {
    id: `${level}-f1`,
    label: `${label} F1`,
    group: "Context Quality",
    explanation: `Harmonic mean of recall and precision for retrieved context at the ${label.toLowerCase()} granularity.`,
    preferLog: false,
    format: formatRatio,
    extract: (instance) => contextLevelMetric(instance, level, "f1"),
  };
}

function resourceCountMetric(
  id: string,
  label: string,
  explanation: string,
  field: NumericResourceField,
): CorrelationMetric {
  return {
    id,
    label,
    group: "Resource Usage",
    explanation,
    preferLog: false,
    format: formatCount,
    extract: (instance) => finiteOrNull(instance.resources[field]),
  };
}

function tokenMetric(
  id: string,
  label: string,
  explanation: string,
  field: NumericResourceField,
): CorrelationMetric {
  return {
    id,
    label,
    group: "Resource Usage",
    explanation,
    preferLog: true,
    format: formatCompact,
    extract: (instance) => finiteOrNull(instance.resources[field]),
  };
}

const contextQualityMetrics: CorrelationMetric[] = [
  {
    id: "context-f1",
    label: "Context F1 (macro)",
    group: "Context Quality",
    explanation: "Mean F1 of retrieved context across the file, symbol, span, and line granularities defined for the instance.",
    preferLog: false,
    format: formatRatio,
    extract: (instance) => instanceContextAggregate(instance, "f1"),
  },
  {
    id: "context-recall",
    label: "Context Recall (macro)",
    group: "Context Quality",
    explanation: "Mean share of gold context recovered across the granularities defined for the instance.",
    preferLog: false,
    format: formatRatio,
    extract: (instance) => instanceContextAggregate(instance, "recall"),
  },
  {
    id: "context-precision",
    label: "Context Precision (macro)",
    group: "Context Quality",
    explanation: "Mean share of retrieved context that is gold across the granularities defined for the instance.",
    preferLog: false,
    format: formatRatio,
    extract: (instance) => instanceContextAggregate(instance, "precision"),
  },
  contextLevelF1Metric("file", "File"),
  contextLevelF1Metric("symbol", "Symbol"),
  contextLevelF1Metric("span", "Span"),
  contextLevelF1Metric("line", "Line"),
  {
    id: "trajectory-gold-found",
    label: "Trajectory Gold Found",
    group: "Context Quality",
    explanation: "Terminal cumulative gold-context coverage along the trajectory, averaged across granularities.",
    preferLog: false,
    format: formatRatio,
    extract: (instance) => instanceTrajectoryGoldFound(instance),
  },
  {
    id: "fix-overlap-f1",
    label: "Patch Overlap vs Gold F1",
    group: "Context Quality",
    explanation: "F1 overlap between the produced patch and the gold patch, when both are available.",
    preferLog: false,
    format: formatRatio,
    extract: (instance) => {
      const overlap = instance.fixOverlap?.vsGold;
      if (!overlap || overlap.status !== "available") return null;
      return finiteOrNull(overlap.f1);
    },
  },
];

const trajectoryMetrics: CorrelationMetric[] = [
  {
    id: "trajectory-efficiency",
    label: "Efficiency",
    group: "Trajectory",
    explanation: "Trajectory efficiency: how directly the agent converged on the relevant context.",
    preferLog: false,
    format: formatRatio,
    extract: (instance) => finiteOrNull(instance.trajectory.efficiency),
  },
  {
    id: "trajectory-redundancy",
    label: "Redundancy",
    group: "Trajectory",
    explanation: "Share of trajectory work that revisited already-seen context.",
    preferLog: false,
    format: formatRatio,
    extract: (instance) => finiteOrNull(instance.trajectory.redundancy),
  },
  {
    id: "trajectory-usage-drop",
    label: "Usage Drop",
    group: "Trajectory",
    explanation: "Share of explored context that was dropped from the final retained set.",
    preferLog: false,
    format: formatRatio,
    extract: (instance) => finiteOrNull(instance.trajectory.usageDrop),
  },
  {
    id: "trajectory-steps",
    label: "Steps",
    group: "Trajectory",
    explanation: "Number of trajectory steps the agent took on the task.",
    preferLog: false,
    format: formatCount,
    extract: (instance) => finiteOrNull(instance.trajectory.steps),
  },
  {
    id: "trajectory-lines-per-step",
    label: "Lines per Step",
    group: "Trajectory",
    explanation: "Average number of context lines touched per trajectory step.",
    preferLog: false,
    format: (value) => value.toFixed(2),
    extract: (instance) => finiteOrNull(instance.trajectory.linesPerStep),
  },
];

const resourceMetrics: CorrelationMetric[] = [
  {
    id: "duration",
    label: "Duration",
    group: "Resource Usage",
    explanation: "Wall-clock run duration. Instances with unavailable durations (e.g. timeouts) are excluded.",
    preferLog: true,
    format: formatDuration,
    extract: (instance) => {
      if (instance.resources.durationStatus === "unavailable") return null;
      const duration = finiteOrNull(instance.resources.durationMs);
      return duration !== null && duration > 0 ? duration : null;
    },
  },
  tokenMetric("total-tokens", "Total Tokens", "Total tokens consumed by the run (input + output, including cache).", "totalTokens"),
  tokenMetric("input-tokens", "Input Tokens", "Input tokens consumed by the run, including cached input.", "inputTokens"),
  tokenMetric("output-tokens", "Output Tokens", "Output tokens produced by the run.", "outputTokens"),
  tokenMetric("cached-input-tokens", "Cached Input Tokens", "Input tokens served from prompt cache.", "cachedInputTokens"),
  tokenMetric("non-cached-input-tokens", "Non-cached Input Tokens", "Input tokens not served from prompt cache.", "nonCachedInputTokens"),
  {
    id: "cost",
    label: "Cost (USD)",
    group: "Resource Usage",
    explanation: "Estimated API cost of the run in USD.",
    preferLog: true,
    format: formatUsd,
    extract: (instance) => finiteOrNull(instance.resources.costUsd),
  },
  resourceCountMetric("tool-calls", "Tool Calls", "Total tool calls made during the run.", "toolCalls"),
  resourceCountMetric("mcp-tool-calls", "MCP Tool Calls", "Tool calls routed to MCP servers during the run.", "mcpToolCalls"),
  resourceCountMetric("command-executions", "Command Executions", "Shell commands executed during the run.", "commandExecutions"),
  resourceCountMetric("read-tool-calls", "Read Tool Calls", "File-read tool calls made during the run.", "readToolCalls"),
  resourceCountMetric("edit-tool-calls", "Edit Tool Calls", "File-edit tool calls made during the run.", "editToolCalls"),
  {
    id: "skill-invocations",
    label: "Skill Invocations",
    group: "Resource Usage",
    explanation: "Total skill invocations recorded for the run.",
    preferLog: false,
    format: formatCount,
    extract: (instance) => finiteOrNull(instance.skills?.totalInvocations),
  },
];

const taskPropertyMetrics: CorrelationMetric[] = [
  {
    id: "repo-tracked-files",
    label: "Repo Tracked Files",
    group: "Task Properties",
    explanation: "Number of git-tracked files in the task repository at the task commit.",
    preferLog: true,
    format: formatCompact,
    extract: (instance) => {
      if (instance.repositorySize?.status === "unavailable") return null;
      return finiteOrNull(instance.repositorySize?.trackedFiles);
    },
  },
  {
    id: "repo-tracked-lines",
    label: "Repo Tracked Lines",
    group: "Task Properties",
    explanation: "Number of text lines across git-tracked files in the task repository.",
    preferLog: true,
    format: formatCompact,
    extract: (instance) => {
      if (instance.repositorySize?.status === "unavailable") return null;
      if (instance.repositorySize?.lineCountStatus === "unavailable") return null;
      return finiteOrNull(instance.repositorySize?.trackedTextLines);
    },
  },
  {
    id: "repo-tracked-bytes",
    label: "Repo Tracked Bytes",
    group: "Task Properties",
    explanation: "Total size in bytes of git-tracked files in the task repository.",
    preferLog: true,
    format: formatBytes,
    extract: (instance) => {
      if (instance.repositorySize?.status === "unavailable") return null;
      return finiteOrNull(instance.repositorySize?.trackedBytes);
    },
  },
];

export const CORRELATION_METRICS: CorrelationMetric[] = [
  ...contextQualityMetrics,
  ...trajectoryMetrics,
  ...resourceMetrics,
  ...taskPropertyMetrics,
];

export const DEFAULT_X_METRIC_ID = "total-tokens";
export const DEFAULT_Y_METRIC_ID = "context-f1";

export function getCorrelationMetric(id: string): CorrelationMetric | null {
  return CORRELATION_METRICS.find((metric) => metric.id === id) ?? null;
}
