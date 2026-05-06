
import type { ComparisonCard } from "@/data/comparisons";
import type { ComparisonVariant, DeltaTone, MetricDefinition, MetricDirection } from "@/components/comparison/types";

const BENCH_ORDER = ["Verified", "Pro", "Poly", "Multi"];

export function parsePercent(value: string): number | null {
  const parsed = Number.parseFloat(value.replace("%", ""));
  return Number.isFinite(parsed) ? parsed : null;
}

export function parseFloatMetric(value: string): number | null {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function parseCurrencyMetric(value: string): number | null {
  const parsed = Number.parseFloat(value.replace("$", ""));
  return Number.isFinite(parsed) ? parsed : null;
}

export function parseCompactNumberMetric(value: string): number | null {
  const normalized = value.trim().toUpperCase();
  if (!normalized) return null;
  if (normalized.endsWith("M")) {
    const parsed = Number.parseFloat(normalized.slice(0, -1));
    return Number.isFinite(parsed) ? parsed * 1_000_000 : null;
  }
  if (normalized.endsWith("K")) {
    const parsed = Number.parseFloat(normalized.slice(0, -1));
    return Number.isFinite(parsed) ? parsed * 1_000 : null;
  }
  return parseFloatMetric(normalized);
}

export function parseDurationMetric(value: string): number | null {
  let totalSeconds = 0;
  for (const part of value.split(" ")) {
    if (part.endsWith("h")) totalSeconds += Number.parseInt(part.slice(0, -1), 10) * 3600;
    if (part.endsWith("m")) totalSeconds += Number.parseInt(part.slice(0, -1), 10) * 60;
    if (part.endsWith("s")) totalSeconds += Number.parseInt(part.slice(0, -1), 10);
  }
  return totalSeconds > 0 ? totalSeconds : null;
}

export function getComparisonPair(
  comparison: ComparisonCard,
): { baseline: ComparisonVariant; treatment: ComparisonVariant } | null {
  if (comparison.variants.length < 2) return null;
  return { baseline: comparison.variants[0], treatment: comparison.variants[1] };
}

export function deltaTone(direction: MetricDirection, delta: number): DeltaTone {
  if (direction === "neutral" || delta === 0) return "neutral";
  const improved = direction === "higher" ? delta > 0 : delta < 0;
  return improved ? "success" : "danger";
}

export function formatSignedFixed(value: number, decimals: number): string {
  if (value === 0) return value.toFixed(decimals);
  return `${value > 0 ? "+" : "-"}${Math.abs(value).toFixed(decimals)}`;
}

export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function formatOptionalOverlapPercent(value: string | number | null | undefined): string {
  if (typeof value === "number" && Number.isFinite(value)) return formatPercent(value);
  if (typeof value === "string" && value.trim()) return value;
  return "—";
}

export function formatMetric(value: number): string {
  return value.toFixed(3);
}

export function formatPatternMetric(value: number): string {
  return value.toFixed(2);
}

export function formatDurationMs(value: number): string {
  const totalSeconds = Math.round(value / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  if (hours > 0) return `${hours}h ${remainingMinutes.toString().padStart(2, "0")}m`;
  return `${minutes}m ${seconds.toString().padStart(2, "0")}s`;
}

export function formatDurationDelta(value: number): string {
  const absoluteSeconds = Math.round(Math.abs(value));
  const minutes = Math.floor(absoluteSeconds / 60);
  const seconds = absoluteSeconds % 60;
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  const prefix = value > 0 ? "+" : value < 0 ? "-" : "";
  if (hours > 0) return `${prefix}${hours}h ${remainingMinutes.toString().padStart(2, "0")}m`;
  if (minutes > 0) return `${prefix}${minutes}m ${seconds.toString().padStart(2, "0")}s`;
  return `${prefix}${seconds}s`;
}

export function formatTokens(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(0)}K`;
  return String(value);
}

export function formatCurrency(value: number): string {
  return `$${value.toFixed(2)}`;
}

export function formatPercentDelta(value: number): string {
  return `${value > 0 ? "+" : value < 0 ? "-" : ""}${Math.abs(value).toFixed(1)}%`;
}

export function formatAbsoluteMetricDelta(metric: MetricDefinition, delta: number): string {
  if ([
    "completedRunRate",
    "officialPassAt1",
    "patchProductionRate",
    "convertedPredictionRate",
    "validEvaluationRate",
    "fixOverlapVsGoldRecall",
    "fixOverlapVsGoldPrecision",
    "fixOverlapVsGoldF1",
  ].includes(metric.key)) {
    return `${formatSignedFixed(delta, 1)} pts`;
  }
  if (metric.key === "averageDuration") return formatDurationDelta(delta);
  if (metric.key === "estimatedCost") return `${delta > 0 ? "+" : delta < 0 ? "-" : ""}$${Math.abs(delta).toFixed(2)}`;
  if (metric.key === "totalTokens") return `${delta > 0 ? "+" : delta < 0 ? "-" : ""}${formatCompactMagnitude(delta)}`;
  if (["toolCalls", "skillInvocations"].includes(metric.key)) {
    return Number.isInteger(delta) ? formatSignedFixed(delta, 0) : formatSignedFixed(delta, 2);
  }
  if (["averageSteps", "avgLinesPerStep"].includes(metric.key)) return formatSignedFixed(delta, 2);
  return formatSignedFixed(delta, 3);
}

export function formatCompactMagnitude(value: number): string {
  const absoluteValue = Math.abs(value);
  if (absoluteValue >= 1_000_000) return `${(absoluteValue / 1_000_000).toFixed(2)}M`;
  if (absoluteValue >= 1_000) return `${(absoluteValue / 1_000).toFixed(2)}K`;
  return Number.isInteger(absoluteValue) ? absoluteValue.toFixed(0) : absoluteValue.toFixed(2);
}

export function coveragePrecision(predSize: number, goldSize: number, intersection: number) {
  return {
    coverage: goldSize > 0 ? intersection / goldSize : 1,
    precision: predSize > 0 ? intersection / predSize : 1,
  };
}

export function f1(coverage: number, precision: number): number {
  const denominator = coverage + precision;
  return denominator === 0 ? 0 : (2 * coverage * precision) / denominator;
}

export function sortBench(left: string, right: string): number {
  const leftIndex = BENCH_ORDER.indexOf(left);
  const rightIndex = BENCH_ORDER.indexOf(right);
  if (leftIndex >= 0 || rightIndex >= 0) {
    return (leftIndex >= 0 ? leftIndex : Number.MAX_SAFE_INTEGER) - (rightIndex >= 0 ? rightIndex : Number.MAX_SAFE_INTEGER);
  }
  return left.localeCompare(right);
}

export function formatLanguageLabel(language: string): string {
  const normalized = language.trim().toLowerCase();
  const known: Record<string, string> = {
    javascript: "JavaScript",
    typescript: "TypeScript",
    python: "Python",
    java: "Java",
    cpp: "C++",
    "c++": "C++",
    c: "C",
    go: "Go",
    rust: "Rust",
  };
  return known[normalized] ?? (normalized ? normalized.charAt(0).toUpperCase() + normalized.slice(1) : language);
}

export function formatInstanceMetric(value: number | null): string {
  return value === null ? "—" : value.toFixed(3);
}

export function formatResolutionStatus(status: string | undefined): string {
  const normalized = (status ?? "").trim().toLowerCase();
  if (!normalized) return "—";
  if (normalized === "resolved") return "Resolved";
  if (normalized === "unresolved") return "Unresolved";
  if (normalized === "error") return "Error";
  if (normalized === "missing") return "Missing";
  return normalized.charAt(0).toUpperCase() + normalized.slice(1);
}

export function resolutionStatusClassName(status: string | undefined): string {
  const normalized = (status ?? "").trim().toLowerCase();
  if (normalized === "resolved") return "text-emerald-700";
  if (normalized === "unresolved") return "text-rose-700";
  if (normalized === "error") return "text-amber-700";
  if (normalized === "missing") return "text-amber-700";
  return "text-muted-foreground";
}

export function deltaIndicatorClassName(tone: DeltaTone): string {
  if (tone === "success") return "text-emerald-700";
  if (tone === "danger") return "text-rose-700";
  return "text-slate-600";
}
