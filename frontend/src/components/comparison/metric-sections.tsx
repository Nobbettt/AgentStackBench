// SPDX-License-Identifier: Apache-2.0
// Fork note: Modified by Norbert Laszlo on 2026-05-28 from upstream ContextBench.
// Summary of changes: add fork-specific comparison metrics and resource distribution charts.

import { Fragment } from "react";

import { completedRunsForOutcome, partialRunsForOutcome, type ComparisonCard } from "@/data/comparisons";
import {
  deltaTone,
  formatOptionalOverlapPercent,
  formatPercentDelta,
  formatResolutionStatus,
  formatSignedFixed,
  getComparisonPair,
  parseFloatMetric,
  resolutionStatusClassName,
} from "@/components/comparison/format";
import { LanguageMetricsSection } from "@/components/comparison/language-metrics-section";
import { MetricSection, MetricVersusValues, type TreatmentDeltaDisplay } from "@/components/comparison/metric-display";
import { ResourceUsageMetricSection } from "@/components/comparison/resource-usage-metric-section";
import {
  executionMetricDefinitions,
  metricDelta,
  outcomeDelta,
  resolutionMetricDefinitions,
} from "@/components/comparison/metrics";
import { ComparisonSectionShell, DeltaIndicator, HelpIcon, MetricDirectionBadge } from "@/components/comparison/shared";
import type { ComparisonResultsViewMode, DeltaDisplayMode, MetricDefinition } from "@/components/comparison/types";
import { cn } from "@/lib/utils";

export { LanguageMetricsSection } from "@/components/comparison/language-metrics-section";
export { ResourceUsageMetricSection } from "@/components/comparison/resource-usage-metric-section";
function isFullyCompletedVariant(variant: ComparisonCard["variants"][number], comparison: ComparisonCard): boolean {
  const totalTasks = variant.instances?.length ?? comparison.tasks;
  return (
    totalTasks > 0 &&
    completedRunsForOutcome(variant.results.outcome) === totalTasks &&
    partialRunsForOutcome(variant.results.outcome) === 0 &&
    variant.results.outcome.failures === 0
  );
}

export function ExecutionMetricSection(props: Omit<Parameters<typeof MetricSection>[0], "title" | "metrics">) {
  return <MetricSection {...props} title="Execution Metrics" metrics={executionMetricDefinitions} />;
}

export function ResolutionMetricSection(props: Omit<Parameters<typeof MetricSection>[0], "title" | "metrics">) {
  return (
    <>
      <PassAt1ResolutionSection {...props} />
      <FixOverlapVsGoldSection {...props} />
    </>
  );
}

export function ContextRetrievalMetricSection(props: Omit<Parameters<typeof MetricSection>[0], "title" | "metrics">) {
  return <ContextBenchContextMetricSection {...props} />;
}

function PassAt1ResolutionSection({
  comparison,
  viewMode,
  deltaDisplayMode,
  treatmentDeltaDisplay = "delta",
  nonGraphDisplay = false,
  collapsible,
  defaultOpen = true,
}: Omit<Parameters<typeof MetricSection>[0], "title" | "metrics">) {
  const passAt1Metric = resolutionMetricDefinitions.find((metric) => metric.key === "officialPassAt1");
  if (!passAt1Metric) return null;

  const hasData = comparison.variants.some((variant) => passAt1Metric.value(variant) !== "—");
  if (!hasData) return null;

  const comparisonPair = getComparisonPair(comparison);
  const showDeltas = viewMode === "treatment-delta" && comparisonPair;
  const variants = comparison.variants;
  const showVersus = showDeltas && treatmentDeltaDisplay === "versus";
  const delta = showDeltas && !showVersus ? metricDelta(passAt1Metric, comparisonPair.baseline, comparisonPair.treatment, deltaDisplayMode) : null;

  if (nonGraphDisplay) {
    return (
      <ComparisonSectionShell
        title="Resolution Metrics"
        collapsible={collapsible}
        defaultOpen={defaultOpen}
        headerInline={<HelpIcon label="Resolution Metrics" explanation={passAt1Metric.explanation} />}
      >
        <div className="rounded-lg bg-background p-5">
          {showDeltas && comparisonPair ? (
            <ResolutionStatusVersusCard baseline={comparisonPair.baseline} treatment={comparisonPair.treatment} metric={passAt1Metric} />
          ) : (
            <div className="grid gap-3 md:grid-cols-2">
              {variants.map((variant) => (
                <ResolutionStatusCard key={variant.label} variant={variant} metric={passAt1Metric} />
              ))}
            </div>
          )}
        </div>
      </ComparisonSectionShell>
    );
  }

  return (
    <ComparisonSectionShell
      title="Resolution Metrics"
      collapsible={collapsible}
      defaultOpen={defaultOpen}
      headerInline={<HelpIcon label="Resolution Metrics" explanation={passAt1Metric.explanation} />}
    >
      <div className="rounded-lg bg-background p-5">
        <div className="mb-5 flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="inline-flex items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground">
              <span>{passAt1Metric.label}</span>
              <MetricDirectionBadge direction={passAt1Metric.direction} />
            </div>
            <div className="mt-1 text-sm text-muted-foreground">Task-resolution rate on evaluated benchmark tasks.</div>
          </div>
          {showVersus ? (
            <div className="w-full sm:w-80">
              <MetricVersusValues
                baselineValue={passAt1Metric.value(comparisonPair.baseline)}
                treatmentValue={passAt1Metric.value(comparisonPair.treatment)}
                direction={passAt1Metric.direction}
                baselineNumericValue={passAt1Metric.parse(passAt1Metric.value(comparisonPair.baseline))}
                treatmentNumericValue={passAt1Metric.parse(passAt1Metric.value(comparisonPair.treatment))}
              />
            </div>
          ) : delta ? (
            <div className="flex flex-col items-end gap-1">
              <div className="text-xs uppercase tracking-wide text-muted-foreground">Treatment delta</div>
              <DeltaIndicator label={delta.label} delta={delta.delta} tone={delta.tone} />
            </div>
          ) : null}
        </div>
        <div className="space-y-4">
          {variants.map((variant, index) => {
            const rawValue = passAt1Metric.value(variant);
            const numericValue = passAt1Metric.parse(rawValue);
            const percentValue = numericValue === null ? 0 : Math.min(Math.max(numericValue, 0), 100);
            const isTreatment = comparisonPair ? variant.label === comparisonPair.treatment.label : index > 0;
            return (
              <div key={variant.label} className="grid gap-2 md:grid-cols-[12rem_1fr_4.5rem] md:items-center">
                <div className="text-sm font-medium text-muted-foreground">{variant.name}</div>
                <div className="relative h-8 overflow-hidden rounded-md bg-muted">
                  <div
                    className={`h-full rounded-md ${isTreatment ? "bg-primary" : "bg-muted-foreground/45"}`}
                    style={{ width: `${percentValue}%` }}
                  />
                </div>
                <div className="text-right text-sm font-medium tabular-nums">{rawValue}</div>
              </div>
            );
          })}
        </div>
      </div>
    </ComparisonSectionShell>
  );
}

function ResolutionStatusCard({
  variant,
  metric,
}: {
  variant: ComparisonCard["variants"][number];
  metric: MetricDefinition;
}) {
  const status = variant.instances?.[0]?.artifacts?.resolutionStatus;
  return (
    <div className="rounded-md border p-4">
      <div className="mb-3 text-sm font-medium text-muted-foreground">{variant.name}</div>
      <div className="flex flex-wrap items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground">
        <span>{metric.label}</span>
        <MetricDirectionBadge direction={metric.direction} />
      </div>
      <div className={cn("mt-3 text-sm font-medium", resolutionStatusClassName(status))}>{formatResolutionStatus(status)}</div>
    </div>
  );
}

function ResolutionStatusVersusCard({
  baseline,
  treatment,
  metric,
}: {
  baseline: ComparisonCard["variants"][number];
  treatment: ComparisonCard["variants"][number];
  metric: MetricDefinition;
}) {
  const baselineStatus = baseline.instances?.[0]?.artifacts?.resolutionStatus;
  const treatmentStatus = treatment.instances?.[0]?.artifacts?.resolutionStatus;
  const baselineValue = formatResolutionStatus(baselineStatus);
  const treatmentValue = formatResolutionStatus(treatmentStatus);
  return (
    <div className="rounded-md border p-4">
      <div className="flex flex-wrap items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground">
        <span>{metric.label}</span>
        <MetricDirectionBadge direction={metric.direction} />
        <HelpIcon label={metric.label} explanation={metric.explanation} />
      </div>
      <MetricVersusValues
        baselineValue={baselineValue}
        treatmentValue={treatmentValue}
        direction={metric.direction}
        baselineNumericValue={resolutionStatusScore(baselineStatus)}
        treatmentNumericValue={resolutionStatusScore(treatmentStatus)}
        baselineClassName={resolutionStatusClassName(baselineStatus)}
        treatmentClassName={resolutionStatusClassName(treatmentStatus)}
      />
    </div>
  );
}

function resolutionStatusScore(status: string | undefined): number | null {
  const normalized = (status ?? "").trim().toLowerCase();
  if (normalized === "resolved") return 1;
  if (["unresolved", "error", "missing"].includes(normalized)) return 0;
  return null;
}

export function OutcomeBreakdownSection({
  comparison,
  viewMode,
  deltaDisplayMode,
  collapsible = false,
}: {
  comparison: ComparisonCard;
  viewMode: ComparisonResultsViewMode;
  deltaDisplayMode: DeltaDisplayMode;
  collapsible?: boolean;
}) {
  if (comparison.variants.length > 0 && comparison.variants.every((variant) => isFullyCompletedVariant(variant, comparison))) {
    return null;
  }

  const comparisonPair = getComparisonPair(comparison);
  const showDeltas = viewMode === "treatment-delta" && comparisonPair;
  const variants = showDeltas ? [comparisonPair.treatment] : comparison.variants;
  return (
    <ComparisonSectionShell title="Execution Outcomes" collapsible={collapsible}>
      <div className="rounded-lg bg-background p-5">
        <div className="grid gap-5 md:grid-cols-2">
          {variants.map((variant) => {
            const items = [
              { name: "Completed" as const, value: completedRunsForOutcome(variant.results.outcome) },
              { name: "Partial" as const, value: partialRunsForOutcome(variant.results.outcome) },
              { name: "Failures" as const, value: variant.results.outcome.failures },
            ];
            return (
              <div key={variant.label}>
                <div className="mb-4 text-sm font-medium text-muted-foreground">{variant.name}</div>
                <div className="grid gap-3 sm:grid-cols-3">
                  {items.map((item) => {
                    const delta = showDeltas ? outcomeDelta(item.name, comparisonPair.baseline, comparisonPair.treatment, deltaDisplayMode) : null;
                    return (
                      <div key={item.name} className="rounded-md border p-4">
                        <div className="text-xs uppercase tracking-wide text-muted-foreground">{item.name}</div>
                        <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
                          <div className="text-lg font-medium">{item.value}</div>
                          {delta ? <DeltaIndicator label={delta.label} delta={delta.delta} tone={delta.tone} /> : null}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </ComparisonSectionShell>
  );
}

type FixOverlapMeasureKey = "recall" | "precision" | "f1";

const fixOverlapMeasures: Array<{ key: FixOverlapMeasureKey; label: string; shortLabel: string }> = [
  { key: "recall", label: "Recall", shortLabel: "Rec." },
  { key: "precision", label: "Precision", shortLabel: "Pre." },
  { key: "f1", label: "F1", shortLabel: "F1" },
];

const fixOverlapExplanation = "Overlap between model patch edit locations and gold patch edit locations. Recall is gold coverage, precision is model-patch accuracy, and F1 balances both.";

function fixOverlapMeasureValue(
  variant: ComparisonCard["variants"][number],
  measure: FixOverlapMeasureKey,
): string {
  return variant.results.quality.fixOverlapVsGold?.[measure] ?? "—";
}

function fixOverlapMeasureDelta(
  baseline: ComparisonCard["variants"][number],
  treatment: ComparisonCard["variants"][number],
  measure: FixOverlapMeasureKey,
  displayMode: DeltaDisplayMode,
) {
  const baselineValue = parseFloatMetric(fixOverlapMeasureValue(baseline, measure));
  const treatmentValue = parseFloatMetric(fixOverlapMeasureValue(treatment, measure));
  if (baselineValue === null || treatmentValue === null) return null;

  const delta = treatmentValue - baselineValue;
  const percentDelta = baselineValue === 0 ? (delta === 0 ? 0 : null) : (delta / Math.abs(baselineValue)) * 100;
  return {
    delta,
    label: displayMode === "percent" ? (percentDelta === null ? "n/a" : formatPercentDelta(percentDelta)) : `${formatSignedFixed(delta, 1)} pts`,
    tone: deltaTone("higher", delta),
  };
}

function FixOverlapVsGoldSection({
  comparison,
  viewMode,
  deltaDisplayMode,
  treatmentDeltaDisplay = "delta",
  collapsible,
  defaultOpen = true,
}: Omit<Parameters<typeof MetricSection>[0], "title" | "metrics">) {
  const hasData = comparison.variants.some((variant) =>
    fixOverlapMeasures.some((measure) => fixOverlapMeasureValue(variant, measure.key) !== "—"),
  );
  if (!hasData) return null;

  const comparisonPair = getComparisonPair(comparison);
  const showDeltas = viewMode === "treatment-delta" && comparisonPair;
  const showVersus = showDeltas && treatmentDeltaDisplay === "versus";
  const primaryVariant = showDeltas ? comparisonPair.treatment : comparison.variants[0];

  return (
    <ComparisonSectionShell
      title="Fix Overlap vs Gold"
      collapsible={collapsible}
      defaultOpen={defaultOpen}
      headerInline={<HelpIcon label="Fix Overlap vs Gold" explanation={fixOverlapExplanation} />}
    >
      <div className="space-y-6 rounded-lg bg-background p-5">
        {showDeltas ? (
          <FixOverlapComparisonStrip
            baseline={comparisonPair.baseline}
            treatment={primaryVariant}
            deltaDisplayMode={deltaDisplayMode}
            treatmentDeltaDisplay={treatmentDeltaDisplay}
          />
        ) : (
          <FixOverlapVariantGrid variants={comparison.variants} />
        )}
        {showDeltas && !showVersus ? (
          <FixOverlapDumbbellChart baseline={comparisonPair.baseline} treatment={comparisonPair.treatment} />
        ) : null}
      </div>
    </ComparisonSectionShell>
  );
}

function FixOverlapVariantGrid({ variants }: { variants: ComparisonCard["variants"] }) {
  return (
    <div className="grid gap-3 md:grid-cols-2">
      {variants.map((variant) => (
        <div key={variant.label} className="rounded-md border p-4">
          <div className="mb-3 text-sm font-medium text-muted-foreground">{variant.name}</div>
          <div className="grid gap-3 sm:grid-cols-3">
            {fixOverlapMeasures.map((measure) => (
              <div key={measure.key}>
                <div className="text-xs uppercase tracking-wide text-muted-foreground">{measure.label}</div>
                <div className="mt-1 font-medium tabular-nums">{fixOverlapMeasureValue(variant, measure.key)}</div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function FixOverlapComparisonStrip({
  baseline,
  treatment,
  deltaDisplayMode,
  treatmentDeltaDisplay = "delta",
}: {
  baseline?: ComparisonCard["variants"][number];
  treatment: ComparisonCard["variants"][number];
  deltaDisplayMode: DeltaDisplayMode;
  treatmentDeltaDisplay?: TreatmentDeltaDisplay;
}) {
  const showVersus = treatmentDeltaDisplay === "versus" && baseline;
  return (
    <div>
      <div className="grid gap-3 rounded-md bg-muted/40 p-4 md:grid-cols-3">
        {fixOverlapMeasures.map((measure) => {
          const delta = baseline ? fixOverlapMeasureDelta(baseline, treatment, measure.key, deltaDisplayMode) : null;
          const baselineValue = baseline ? fixOverlapMeasureValue(baseline, measure.key) : "—";
          const treatmentValue = fixOverlapMeasureValue(treatment, measure.key);
          return (
            <div key={measure.key} className={showVersus ? "space-y-3" : "flex flex-wrap items-center justify-between gap-3"}>
              <div>
                <div className="text-sm font-medium">{measure.label}</div>
                {baseline && !showVersus ? (
                  <div className="text-xs text-muted-foreground">Baseline {baselineValue}</div>
                ) : null}
              </div>
              {showVersus ? (
                <MetricVersusValues
                  baselineValue={baselineValue}
                  treatmentValue={treatmentValue}
                  direction="higher"
                  baselineNumericValue={parseFloatMetric(baselineValue)}
                  treatmentNumericValue={parseFloatMetric(treatmentValue)}
                />
              ) : (
                <div className="flex items-center gap-2">
                  <span className="font-medium tabular-nums">{treatmentValue}</span>
                  {delta ? <DeltaIndicator label={delta.label} delta={delta.delta} tone={delta.tone} /> : null}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function FixOverlapDumbbellChart({
  baseline,
  treatment,
}: {
  baseline: ComparisonCard["variants"][number];
  treatment: ComparisonCard["variants"][number];
}) {
  return (
    <div>
      <div className="space-y-4 rounded-md border p-4">
        <div className="flex items-center justify-end gap-4 text-xs text-muted-foreground">
          <span className="inline-flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-muted-foreground" />{baseline.name}</span>
          <span className="inline-flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-primary" />{treatment.name}</span>
        </div>
        {fixOverlapMeasures.map((measure) => {
          const baselineValue = parseFloatMetric(fixOverlapMeasureValue(baseline, measure.key));
          const treatmentValue = parseFloatMetric(fixOverlapMeasureValue(treatment, measure.key));
          if (baselineValue === null || treatmentValue === null) return null;
          const left = Math.min(Math.max(baselineValue, 0), 100);
          const right = Math.min(Math.max(treatmentValue, 0), 100);
          const start = Math.min(left, right);
          const width = Math.abs(right - left);
          return (
            <div key={measure.key} className="grid gap-3 md:grid-cols-[7rem_1fr_8rem] md:items-center">
              <div className="text-sm font-medium">{measure.label}</div>
              <div className="relative h-8">
                <div className="absolute left-0 right-0 top-1/2 h-1 -translate-y-1/2 rounded-full bg-muted" />
                <div
                  className="absolute top-1/2 h-1 -translate-y-1/2 rounded-full bg-primary/40"
                  style={{ left: `${start}%`, width: `${width}%` }}
                />
                <span
                  className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-muted-foreground ring-2 ring-background"
                  style={{ left: `${left}%` }}
                />
                <span
                  className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-primary ring-2 ring-background"
                  style={{ left: `${right}%` }}
                />
              </div>
              <div className="text-right text-sm tabular-nums text-muted-foreground">
                {fixOverlapMeasureValue(baseline, measure.key)} → <span className="font-medium text-foreground">{fixOverlapMeasureValue(treatment, measure.key)}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function ComparisonMetricSections({
  comparison,
  viewMode,
  deltaDisplayMode,
  collapsible = false,
  showExecutionMetrics = true,
}: {
  comparison: ComparisonCard;
  viewMode: ComparisonResultsViewMode;
  deltaDisplayMode: DeltaDisplayMode;
  collapsible?: boolean;
  showExecutionMetrics?: boolean;
}) {
  return (
    <>
      {showExecutionMetrics ? (
        <ExecutionMetricSection
          comparison={comparison}
          viewMode={viewMode}
          deltaDisplayMode={deltaDisplayMode}
          collapsible={collapsible}
          defaultOpen={false}
        />
      ) : null}
      <ResolutionMetricSection comparison={comparison} viewMode={viewMode} deltaDisplayMode={deltaDisplayMode} collapsible={collapsible} />
      <ContextRetrievalMetricSection comparison={comparison} viewMode={viewMode} deltaDisplayMode={deltaDisplayMode} collapsible={collapsible} />
      <PatchOverlapBetweenVariantsSection overlap={comparison.fixOverlapBetweenVariants} collapsible={collapsible} />
      <ResourceUsageMetricSection comparison={comparison} viewMode={viewMode} deltaDisplayMode={deltaDisplayMode} collapsible={collapsible} />
      <LanguageMetricsSection comparison={comparison} viewMode={viewMode} deltaDisplayMode={deltaDisplayMode} />
    </>
  );
}

export function PatchOverlapBetweenVariantsSection({
  overlap,
  collapsible = false,
}: {
  overlap?: ComparisonCard["fixOverlapBetweenVariants"];
  collapsible?: boolean;
}) {
  if (!overlap) return null;

  const leftLabel = overlap.leftLabel || "A";
  const rightLabel = overlap.rightLabel || "B";
  const items = [
    {
      label: `${leftLabel} covered by ${rightLabel}`,
      value: formatOptionalOverlapPercent(overlap.leftCoveredByRight),
    },
    {
      label: `${rightLabel} covered by ${leftLabel}`,
      value: formatOptionalOverlapPercent(overlap.rightCoveredByLeft),
    },
    {
      label: "Overlap F1",
      value: formatOptionalOverlapPercent(overlap.f1),
    },
  ];
  const hasAvailabilityCounts =
    typeof overlap.availableInstances === "number" || typeof overlap.unavailableInstances === "number";
  const footer = overlap.status === "available"
    ? hasAvailabilityCounts
      ? `${overlap.availableInstances ?? 0} available / ${overlap.unavailableInstances ?? 0} unavailable`
      : `${overlap.intersection ?? 0} overlap / ${overlap.leftSize ?? 0} ${leftLabel} / ${overlap.rightSize ?? 0} ${rightLabel}`
    : `Unavailable${overlap.reason ? `: ${overlap.reason}` : ""}`;
  const explanation = `Overlap between ${leftLabel} and ${rightLabel} model patch edit locations.`;

  return (
    <ComparisonSectionShell
      title="Patch Overlap"
      collapsible={collapsible}
      defaultOpen={false}
      headerInline={<HelpIcon label="Patch Overlap" explanation={explanation} />}
    >
      <div className="rounded-lg bg-background p-5">
        <div className="grid gap-3 sm:grid-cols-3">
          {items.map((item) => (
            <div key={item.label} className="rounded-md border p-4">
              <div className="text-xs uppercase tracking-wide text-muted-foreground">{item.label}</div>
              <div className="mt-3 text-sm font-medium tabular-nums">{overlap.status === "available" ? item.value : "—"}</div>
            </div>
          ))}
        </div>
        <div className="mt-4 text-xs text-muted-foreground">{footer}</div>
      </div>
    </ComparisonSectionShell>
  );
}

type ContextLevelKey = "file" | "block" | "line" | "symbol";
type ContextMeasureKey = "recall" | "precision" | "f1";

const contextSummaryMeasures: Array<{ key: ContextMeasureKey; label: string; explanation: string }> = [
  {
    key: "f1",
    label: "Context F1",
    explanation: "Aggregate context retrieval F1 across file, symbol, and block levels.",
  },
  {
    key: "recall",
    label: "Context Recall",
    explanation: "Aggregate context retrieval recall across file, symbol, and block levels.",
  },
  {
    key: "precision",
    label: "Context Precision",
    explanation: "Aggregate context retrieval precision across file, symbol, and block levels.",
  },
];

const contextLevelRows: Array<{ key: ContextLevelKey; label: string; explanation: string }> = [
  {
    key: "file",
    label: "File Level",
    explanation: "Recall, precision, and F1 over files in the final retrieved context.",
  },
  {
    key: "block",
    label: "Block Level",
    explanation: "Recall, precision, and F1 over final retrieved code spans. This is the local span metric reported with ContextBench block-level terminology.",
  },
  {
    key: "line",
    label: "Line Level",
    explanation: "Recall, precision, and F1 over final retrieved line intervals.",
  },
  {
    key: "symbol",
    label: "Symbol Level",
    explanation: "Supplemental fork metric over retrieved functions, classes, methods, or other named code entities.",
  },
];

function contextMeasureValue(
  variant: ComparisonCard["variants"][number],
  level: ContextLevelKey,
  measure: ContextMeasureKey,
): string {
  const value = variant.results.quality.contextLevels?.[level]?.[measure];
  if (value) return value;
  if (measure !== "f1") return "—";
  if (level === "file") return variant.results.quality.fileF1 ?? "—";
  if (level === "symbol") return variant.results.quality.symbolF1 ?? "—";
  if (level === "block") return variant.results.quality.spanF1 ?? "—";
  return variant.results.quality.avgLineF1 ?? "—";
}

function contextAggregateValue(variant: ComparisonCard["variants"][number], measure: ContextMeasureKey): string {
  if (measure === "f1") return variant.results.quality.contextF1 ?? variant.contextF1 ?? variant.score ?? "—";
  const directValue = measure === "recall" ? variant.results.quality.contextRecall : variant.results.quality.contextPrecision;
  if (directValue) return directValue;

  const values = (["file", "symbol", "block"] as const)
    .map((level) => parseFloatMetric(contextMeasureValue(variant, level, measure)))
    .filter((value): value is number => value !== null);
  return values.length === 3 ? (values.reduce((sum, value) => sum + value, 0) / values.length).toFixed(3) : "—";
}

function contextValueDelta(
  baselineValue: string,
  treatmentValue: string,
  displayMode: DeltaDisplayMode,
) {
  const parsedBaseline = parseFloatMetric(baselineValue);
  const parsedTreatment = parseFloatMetric(treatmentValue);
  if (parsedBaseline === null || parsedTreatment === null) return null;

  const delta = parsedTreatment - parsedBaseline;
  const percentDelta = parsedBaseline === 0 ? (delta === 0 ? 0 : null) : (delta / Math.abs(parsedBaseline)) * 100;
  return {
    delta,
    label: displayMode === "percent" ? (percentDelta === null ? "n/a" : formatPercentDelta(percentDelta)) : formatSignedFixed(delta, 3),
    tone: deltaTone("higher", delta),
  };
}

function contextMeasureDelta(
  baseline: ComparisonCard["variants"][number],
  treatment: ComparisonCard["variants"][number],
  level: ContextLevelKey,
  measure: ContextMeasureKey,
  displayMode: DeltaDisplayMode,
) {
  return contextValueDelta(contextMeasureValue(baseline, level, measure), contextMeasureValue(treatment, level, measure), displayMode);
}

function contextAggregateDelta(
  baseline: ComparisonCard["variants"][number],
  treatment: ComparisonCard["variants"][number],
  measure: ContextMeasureKey,
  displayMode: DeltaDisplayMode,
) {
  return contextValueDelta(contextAggregateValue(baseline, measure), contextAggregateValue(treatment, measure), displayMode);
}

function ContextSummaryCard({
  label,
  explanation,
  value,
  delta,
}: {
  label: string;
  explanation: string;
  value: string;
  delta?: ReturnType<typeof contextValueDelta>;
}) {
  return (
    <div className="rounded-md border p-4">
      <div className="flex flex-wrap items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground">
        <span>{label}</span>
        <MetricDirectionBadge direction="higher" />
        <HelpIcon label={label} explanation={explanation} />
      </div>
      <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
        <div className="text-xl font-semibold tabular-nums">{value}</div>
        {delta ? <DeltaIndicator label={delta.label} delta={delta.delta} tone={delta.tone} /> : null}
      </div>
    </div>
  );
}

function ContextSummaryVersusCard({
  label,
  explanation,
  baselineValue,
  treatmentValue,
}: {
  label: string;
  explanation: string;
  baselineValue: string;
  treatmentValue: string;
}) {
  return (
    <div className="rounded-md border p-4">
      <div className="flex flex-wrap items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground">
        <span>{label}</span>
        <MetricDirectionBadge direction="higher" />
        <HelpIcon label={label} explanation={explanation} />
      </div>
      <MetricVersusValues
        baselineValue={baselineValue}
        treatmentValue={treatmentValue}
        direction="higher"
        baselineNumericValue={parseFloatMetric(baselineValue)}
        treatmentNumericValue={parseFloatMetric(treatmentValue)}
      />
    </div>
  );
}

function ContextMeasureCell({
  value,
  delta,
  emphasize = false,
}: {
  value: string;
  delta?: ReturnType<typeof contextMeasureDelta>;
  emphasize?: boolean;
}) {
  return (
    <td className="px-3 py-3 text-right">
      <div className="inline-flex items-center justify-end gap-2">
        <span className={`${emphasize ? "font-medium " : ""}tabular-nums`}>{value}</span>
        {delta ? <DeltaIndicator label={delta.label} delta={delta.delta} tone={delta.tone} /> : null}
      </div>
    </td>
  );
}

function ContextBenchContextMetricSection({
  comparison,
  viewMode,
  deltaDisplayMode,
  treatmentDeltaDisplay = "delta",
  collapsible,
  defaultOpen = true,
}: Omit<Parameters<typeof MetricSection>[0], "title" | "metrics">) {
  const comparisonPair = getComparisonPair(comparison);
  const showDeltas = viewMode === "treatment-delta" && comparisonPair;
  const showVersus = showDeltas && treatmentDeltaDisplay === "versus";
  const variants = showDeltas && !showVersus ? [comparisonPair.treatment] : comparison.variants;
  const summaryVariants = showVersus ? [comparisonPair.treatment] : variants;

  return (
    <ComparisonSectionShell
      title="Context Metrics"
      collapsible={collapsible}
      defaultOpen={defaultOpen}
    >
      <div className="rounded-lg bg-background p-5">
        <div className="grid gap-5 md:grid-cols-2">
          {summaryVariants.map((variant) => (
            <div key={variant.label}>
              {!showDeltas && comparison.variants.length > 1 ? (
                <div className="mb-4 text-sm font-medium text-muted-foreground">{variant.name}</div>
              ) : null}
              <div className="grid gap-3 sm:grid-cols-3">
                {contextSummaryMeasures.map((measure) => {
                  if (showVersus) {
                    return (
                      <ContextSummaryVersusCard
                        key={measure.key}
                        label={measure.label}
                        explanation={measure.explanation}
                        baselineValue={contextAggregateValue(comparisonPair.baseline, measure.key)}
                        treatmentValue={contextAggregateValue(comparisonPair.treatment, measure.key)}
                      />
                    );
                  }
                  return (
                    <ContextSummaryCard
                      key={measure.key}
                      label={measure.label}
                      explanation={measure.explanation}
                      value={contextAggregateValue(variant, measure.key)}
                      delta={showDeltas ? contextAggregateDelta(comparisonPair.baseline, variant, measure.key, deltaDisplayMode) : undefined}
                    />
                  );
                })}
              </div>
            </div>
          ))}
        </div>
        <div className="mt-6 overflow-x-auto">
          <table className="w-full min-w-[44rem] border-collapse text-base">
            <thead>
              <tr className="border-b text-sm uppercase tracking-wide text-muted-foreground">
                <th className="px-3 py-2 text-left font-medium">Level</th>
                {variants.map((variant) => (
                  <Fragment key={variant.label}>
                    <th className="px-3 py-2 text-right font-medium">F1</th>
                    <th className="px-3 py-2 text-right font-medium">Recall</th>
                    <th className="px-3 py-2 text-right font-medium">Precision</th>
                  </Fragment>
                ))}
              </tr>
            </thead>
            <tbody>
              {contextLevelRows.map((row) => (
                <tr key={row.key} className="border-b last:border-b-0">
                  <th className="px-3 py-3 text-left font-medium">
                    <span className="inline-flex items-center gap-2">
                      {row.label}
                      <HelpIcon label={row.label} explanation={row.explanation} />
                    </span>
                  </th>
                  {variants.map((variant) => (
                    <Fragment key={variant.label}>
                      <ContextMeasureCell
                        value={contextMeasureValue(variant, row.key, "f1")}
                        delta={showDeltas && !showVersus ? contextMeasureDelta(comparisonPair.baseline, variant, row.key, "f1", deltaDisplayMode) : undefined}
                        emphasize
                      />
                      <ContextMeasureCell
                        value={contextMeasureValue(variant, row.key, "recall")}
                        delta={showDeltas && !showVersus ? contextMeasureDelta(comparisonPair.baseline, variant, row.key, "recall", deltaDisplayMode) : undefined}
                      />
                      <ContextMeasureCell
                        value={contextMeasureValue(variant, row.key, "precision")}
                        delta={showDeltas && !showVersus ? contextMeasureDelta(comparisonPair.baseline, variant, row.key, "precision", deltaDisplayMode) : undefined}
                      />
                    </Fragment>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </ComparisonSectionShell>
  );
}
