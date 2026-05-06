
import type { ComparisonCard, ComparisonInstance, ComparisonInstanceDetail } from "@/data/comparisons";

export type OutcomeMetricName = "Completed" | "Partial" | "Failures";
export type MetricDirection = "higher" | "lower" | "neutral";
export type ComparisonResultsViewMode = "side-by-side" | "treatment-delta";
export type DeltaDisplayMode = "absolute" | "percent";
export type ComparisonVariant = ComparisonCard["variants"][number];
export type DeltaTone = "success" | "danger" | "neutral";

export type InstanceRow = {
  instanceId: string;
  originalInstanceId?: string | null;
  bench: string;
  language: string;
  baseline?: ComparisonInstance;
  treatment?: ComparisonInstance;
};

export type DetailVariant = ComparisonInstanceDetail["variants"][number];

export type MetricDefinition = {
  key: string;
  label: string;
  explanation: string;
  direction: MetricDirection;
  value: (variant: ComparisonVariant) => string;
  parse: (value: string) => number | null;
};

export type MetricDelta = {
  delta: number;
  label: string;
  tone: DeltaTone;
};
