// SPDX-License-Identifier: Apache-2.0
// Fork note: Modified by Norbert Laszlo on 2026-05-28 from upstream ContextBench.
// Summary of changes: add fork-specific comparison metrics and resource distribution charts.

import { Fragment, useState } from "react";
import { Minus, TrendingDown, TrendingUp } from "lucide-react";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, XAxis, YAxis } from "recharts";

import { completedRunsForOutcome, partialRunsForOutcome, type ComparisonCard, type ComparisonInstance } from "@/data/comparisons";
import {
  deltaIndicatorClassName,
  deltaTone,
  formatOptionalOverlapPercent,
  formatPercent,
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
import { getMetricSignificance, type PairedSignificance } from "@/components/comparison/significance";
import { terminalTrajectoryCoverage } from "@/data/instance-metrics";
import { ComparisonSectionShell, DeltaIndicator, HelpIcon, MetricDirectionBadge, SignificanceBadge } from "@/components/comparison/shared";
import type { ComparisonResultsViewMode, DeltaDisplayMode, MetricDefinition } from "@/components/comparison/types";
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
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

export function ResolutionMetricSection({
  showFixOverlap = true,
  ...props
}: Omit<Parameters<typeof MetricSection>[0], "title" | "metrics"> & { showFixOverlap?: boolean }) {
  return (
    <>
      <PassAt1ResolutionSection {...props} />
      {showFixOverlap ? <FixOverlapVsGoldSection {...props} /> : null}
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
  const significance = showDeltas && comparisonPair ? getMetricSignificance(comparison, passAt1Metric.key) : null;
  const resolutionSetOverlap = buildResolutionSetOverlapCounts(comparison);
  const showResolutionStackedBars = Boolean(
    resolutionSetOverlap && variants.every((variant) => resolutionSetMatchesDisplayedDenominator(resolutionSetOverlap, variant)),
  );

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
            <ResolutionStatusVersusCard baseline={comparisonPair.baseline} treatment={comparisonPair.treatment} metric={passAt1Metric} significance={significance} />
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
              <div className="mt-3 flex justify-end">
                <SignificanceBadge stat={significance} />
              </div>
            </div>
          ) : delta ? (
            <div className="flex flex-col items-end gap-1">
              <div className="text-xs uppercase tracking-wide text-muted-foreground">Treatment delta</div>
              <div className="flex flex-wrap items-center justify-end gap-2">
                <DeltaIndicator label={delta.label} delta={delta.delta} tone={delta.tone} />
                <SignificanceBadge stat={significance} />
              </div>
            </div>
          ) : null}
        </div>
        <div className="space-y-4">
          {variants.map((variant, index) => {
            const rawValue = passAt1Metric.value(variant);
            const numericValue = passAt1Metric.parse(rawValue);
            const percentValue = numericValue === null ? 0 : Math.min(Math.max(numericValue, 0), 100);
            const isTreatment = comparisonPair ? variant.label === comparisonPair.treatment.label : index > 0;
            const solveSegments = showResolutionStackedBars && resolutionSetOverlap && comparisonPair
              ? resolutionMetricBarSegments(variant, comparisonPair.baseline, comparisonPair.treatment, resolutionSetOverlap)
              : null;
            const solveSegmentTotal = resolutionSetOverlap?.comparedTasks ?? 0;
            return (
              <div key={variant.label} className="grid gap-2 md:grid-cols-[12rem_1fr_8rem] md:items-center">
                <div className="text-sm font-medium text-muted-foreground">{variant.name}</div>
                <div className="relative h-8 overflow-hidden rounded-md bg-muted">
                  {solveSegments ? (
                    <div className="flex h-full">
                      {solveSegments.map((segment) => (
                        <div
                          key={segment.label}
                          className={segment.className}
                          style={{ width: `${resolutionSetPercent(segment.value, solveSegmentTotal)}%` }}
                          title={`${segment.label}: ${segment.value}`}
                        />
                      ))}
                    </div>
                  ) : (
                    <div
                      className={`h-full rounded-md ${isTreatment ? "bg-primary" : "bg-muted-foreground/45"}`}
                      style={{ width: `${percentValue}%` }}
                    />
                  )}
                </div>
                <div className="text-right tabular-nums">
                  <div className="text-sm font-medium">{rawValue}</div>
                  <div className="text-xs text-muted-foreground">{resolutionSuccessCountLabel(variant)}</div>
                </div>
              </div>
            );
          })}
        </div>
        {showResolutionStackedBars && resolutionSetOverlap ? (
          <ResolutionMetricBarLegend counts={resolutionSetOverlap} />
        ) : null}
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
      <div className="mt-1 text-xs text-muted-foreground">{resolutionSuccessCountLabel(variant)}</div>
    </div>
  );
}

function ResolutionStatusVersusCard({
  baseline,
  treatment,
  metric,
  significance,
}: {
  baseline: ComparisonCard["variants"][number];
  treatment: ComparisonCard["variants"][number];
  metric: MetricDefinition;
  significance?: PairedSignificance | null;
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
      <div className="mt-3 flex justify-end">
        <SignificanceBadge stat={significance} />
      </div>
      <div className="mt-3 grid gap-2 text-xs text-muted-foreground sm:grid-cols-2">
        <div>
          <span className="font-medium text-foreground">{baseline.name}:</span> {resolutionSuccessCountLabel(baseline)}
        </div>
        <div>
          <span className="font-medium text-foreground">{treatment.name}:</span> {resolutionSuccessCountLabel(treatment)}
        </div>
      </div>
    </div>
  );
}

function resolutionSuccessCountLabel(variant: ComparisonCard["variants"][number]): string {
  const resolvedTasks = variant.results.integrity?.resolvedTasks;
  const totalTasks = variant.results.outcome.expectedTasks ?? variant.instances?.length;
  if (typeof resolvedTasks !== "number") return "Resolved count unavailable";
  if (typeof totalTasks !== "number" || totalTasks <= 0) return `${resolvedTasks.toLocaleString()} resolved`;
  return `${resolvedTasks.toLocaleString()} / ${totalTasks.toLocaleString()} resolved`;
}

function resolutionStatusScore(status: string | undefined): number | null {
  const normalized = (status ?? "").trim().toLowerCase();
  if (normalized === "resolved") return 1;
  if (["unresolved", "error", "missing"].includes(normalized)) return 0;
  return null;
}

type ResolutionSetOverlapCounts = {
  baselineName: string;
  treatmentName: string;
  sharedSolves: number;
  treatmentOnlySolves: number;
  baselineOnlySolves: number;
  neitherSolved: number;
  comparedTasks: number;
  baselineSolved: number;
  treatmentSolved: number;
  unionSolved: number;
  solveRetention: number | null;
  novelSolveRate: number | null;
  resolutionJaccard: number | null;
  netGain: number;
};

const resolutionSetOverlapExplanation = "Compares which exact tasks each variant resolves. This separates shared solves from treatment-only gains and baseline-only regressions.";

export function ResolutionSetOverlapSection({
  comparison,
  collapsible = false,
  defaultOpen = true,
}: {
  comparison: ComparisonCard;
  collapsible?: boolean;
  defaultOpen?: boolean;
}) {
  const counts = buildResolutionSetOverlapCounts(comparison);
  if (!counts || counts.comparedTasks === 0) return null;

  return (
    <ComparisonSectionShell
      title="Resolution Set Overlap"
      collapsible={collapsible}
      defaultOpen={defaultOpen}
      headerInline={<HelpIcon label="Resolution Set Overlap" explanation={resolutionSetOverlapExplanation} />}
    >
      <div className="rounded-lg bg-background p-5">
        <ResolutionSetOverlapVenn counts={counts} />
      </div>
    </ComparisonSectionShell>
  );
}

function buildResolutionSetOverlapCounts(comparison: ComparisonCard): ResolutionSetOverlapCounts | null {
  const comparisonPair = getComparisonPair(comparison);
  if (!comparisonPair) return null;

  const baselineById = new Map((comparisonPair.baseline.instances ?? []).map((instance) => [instance.instanceId, instance]));
  const treatmentById = new Map((comparisonPair.treatment.instances ?? []).map((instance) => [instance.instanceId, instance]));
  const ids = Array.from(new Set([...baselineById.keys(), ...treatmentById.keys()]));

  let sharedSolves = 0;
  let treatmentOnlySolves = 0;
  let baselineOnlySolves = 0;
  let neitherSolved = 0;
  let comparedTasks = 0;

  for (const id of ids) {
    const baselineStatus = baselineById.get(id)?.artifacts?.resolutionStatus;
    const treatmentStatus = treatmentById.get(id)?.artifacts?.resolutionStatus;
    if (!isComparableResolutionStatus(baselineStatus) || !isComparableResolutionStatus(treatmentStatus)) continue;

    comparedTasks += 1;
    const baselineSolved = baselineStatus === "resolved";
    const treatmentSolved = treatmentStatus === "resolved";
    if (baselineSolved && treatmentSolved) sharedSolves += 1;
    else if (!baselineSolved && treatmentSolved) treatmentOnlySolves += 1;
    else if (baselineSolved && !treatmentSolved) baselineOnlySolves += 1;
    else neitherSolved += 1;
  }

  const baselineSolved = sharedSolves + baselineOnlySolves;
  const treatmentSolved = sharedSolves + treatmentOnlySolves;
  const unionSolved = sharedSolves + baselineOnlySolves + treatmentOnlySolves;
  return {
    baselineName: comparisonPair.baseline.name,
    treatmentName: comparisonPair.treatment.name,
    sharedSolves,
    treatmentOnlySolves,
    baselineOnlySolves,
    neitherSolved,
    comparedTasks,
    baselineSolved,
    treatmentSolved,
    unionSolved,
    solveRetention: baselineSolved > 0 ? sharedSolves / baselineSolved : null,
    novelSolveRate: treatmentSolved > 0 ? treatmentOnlySolves / treatmentSolved : null,
    resolutionJaccard: unionSolved > 0 ? sharedSolves / unionSolved : null,
    netGain: treatmentOnlySolves - baselineOnlySolves,
  };
}

function isComparableResolutionStatus(status: string | undefined): boolean {
  return status === "resolved" || status === "unresolved";
}

function resolutionSetMatchesDisplayedDenominator(
  counts: ResolutionSetOverlapCounts,
  variant: ComparisonCard["variants"][number],
): boolean {
  const expectedTasks = variant.results.outcome.expectedTasks;
  const displayedTasks = typeof expectedTasks === "number" ? expectedTasks : variant.instances?.length;
  return typeof displayedTasks === "number" && displayedTasks > 0 && counts.comparedTasks === displayedTasks;
}

type ResolutionMetricBarSegment = {
  label: string;
  value: number;
  className: string;
};

function resolutionMetricBarSegments(
  variant: ComparisonCard["variants"][number],
  baseline: ComparisonCard["variants"][number],
  treatment: ComparisonCard["variants"][number],
  counts: ResolutionSetOverlapCounts,
): ResolutionMetricBarSegment[] | null {
  if (variant.label === baseline.label) {
    return [
      { label: "Shared solves", value: counts.sharedSolves, className: "bg-emerald-500" },
      { label: `${counts.baselineName} only`, value: counts.baselineOnlySolves, className: "bg-amber-500" },
    ];
  }
  if (variant.label === treatment.label) {
    return [
      { label: "Shared solves", value: counts.sharedSolves, className: "bg-emerald-500" },
      { label: `${counts.treatmentName} only`, value: counts.treatmentOnlySolves, className: "bg-primary" },
    ];
  }
  return null;
}

function ResolutionMetricBarLegend({ counts }: { counts: ResolutionSetOverlapCounts }) {
  const items = [
    { label: "Shared solves", value: counts.sharedSolves, className: "bg-emerald-500" },
    { label: `${counts.treatmentName} only`, value: counts.treatmentOnlySolves, className: "bg-primary" },
    { label: `${counts.baselineName} only`, value: counts.baselineOnlySolves, className: "bg-amber-500" },
  ];
  return (
    <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t pt-4">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-muted-foreground">
        {items.map((item) => (
          <span key={item.label} className="inline-flex items-center gap-2">
            <span className={cn("h-2.5 w-2.5 rounded-full", item.className)} />
            {item.label}
            <span className="font-medium tabular-nums text-foreground">{item.value}</span>
          </span>
        ))}
      </div>
      <div className="text-xs text-muted-foreground">{counts.comparedTasks} compared tasks</div>
    </div>
  );
}

function ResolutionSetOverlapVenn({ counts }: { counts: ResolutionSetOverlapCounts }) {
  return (
    <div>
      <ResolutionSetOverlapPanelTitle title="Venn-Style Solve Sets" />
      <div className="relative mx-auto mt-5 h-56 max-w-md">
        <div className="absolute left-[8%] top-4 flex h-44 w-44 items-center justify-start rounded-full border-2 border-amber-500/70 bg-amber-500/10 pl-8">
          <div>
            <div className="text-2xl font-semibold tabular-nums text-amber-700">{counts.baselineOnlySolves}</div>
            <div className="text-xs text-muted-foreground">{counts.baselineName} only</div>
          </div>
        </div>
        <div className="absolute right-[8%] top-4 flex h-44 w-44 items-center justify-end rounded-full border-2 border-primary/70 bg-primary/10 pr-8 text-right">
          <div>
            <div className="text-2xl font-semibold tabular-nums text-primary">{counts.treatmentOnlySolves}</div>
            <div className="text-xs text-muted-foreground">{counts.treatmentName} only</div>
          </div>
        </div>
        <div className="absolute left-1/2 top-12 -translate-x-1/2 rounded-md bg-background/95 px-4 py-2 text-center shadow-sm ring-1 ring-border">
          <div className="text-2xl font-semibold tabular-nums text-emerald-700">{counts.sharedSolves}</div>
          <div className="text-xs text-muted-foreground">shared</div>
        </div>
        <div className="absolute bottom-0 left-1/2 -translate-x-1/2 text-center text-xs text-muted-foreground">
          {counts.neitherSolved} neither solved
        </div>
      </div>
      <div className="mt-5 grid gap-3 text-sm sm:grid-cols-3">
        <ResolutionSetOverlapStat
          label="Solve Retention"
          value={formatNullableResolutionSetPercent(counts.solveRetention)}
          explanation={`Of tasks solved by ${counts.baselineName}, the share also solved by ${counts.treatmentName}.`}
        />
        <ResolutionSetOverlapStat
          label="Novel Solve Rate"
          value={formatNullableResolutionSetPercent(counts.novelSolveRate)}
          explanation={`Of tasks solved by ${counts.treatmentName}, the share that ${counts.baselineName} missed.`}
        />
        <ResolutionSetOverlapStat
          label="Resolution Jaccard"
          value={formatNullableResolutionSetPercent(counts.resolutionJaccard)}
          explanation="Shared solved tasks divided by the union of tasks solved by either variant."
        />
      </div>
      <div className="mt-4 text-xs text-muted-foreground">
        Computed over {counts.comparedTasks} tasks with resolved or unresolved status in both variants.
      </div>
    </div>
  );
}

function ResolutionSetOverlapPanelTitle({ title }: { title: string }) {
  return <h3 className="text-sm font-semibold tracking-tight">{title}</h3>;
}

function ResolutionSetOverlapStat({
  label,
  value,
  explanation,
}: {
  label: string;
  value: string;
  explanation: string;
}) {
  return (
    <div>
      <div className="flex items-center gap-1.5 text-xs uppercase tracking-wide text-muted-foreground">
        <span>{label}</span>
        <HelpIcon label={label} explanation={explanation} />
      </div>
      <div className="mt-1 font-medium tabular-nums">{value}</div>
    </div>
  );
}

function resolutionSetPercent(value: number, total: number): number {
  return total > 0 ? (value / total) * 100 : 0;
}

function formatResolutionSetPercent(value: number, total: number): string {
  return `${resolutionSetPercent(value, total).toFixed(1)}%`;
}

function formatNullableResolutionSetPercent(value: number | null): string {
  return value === null ? "—" : formatPercent(value);
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

export function FixOverlapVsGoldSection({
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
            comparison={comparison}
            baseline={comparisonPair.baseline}
            treatment={primaryVariant}
            deltaDisplayMode={deltaDisplayMode}
            treatmentDeltaDisplay={treatmentDeltaDisplay}
          />
        ) : (
          <FixOverlapVariantGrid variants={comparison.variants} />
        )}
        {showDeltas && !showVersus ? (
          <FixOverlapDumbbellChart
            baseline={comparisonPair.baseline}
            treatment={comparisonPair.treatment}
            deltaDisplayMode={deltaDisplayMode}
          />
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
  comparison,
  baseline,
  treatment,
  deltaDisplayMode,
  treatmentDeltaDisplay = "delta",
}: {
  comparison: ComparisonCard;
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
          const significance = baseline ? getMetricSignificance(comparison, fixOverlapMetricKey(measure.key)) : null;
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
                  <SignificanceBadge stat={significance} />
                </div>
              )}
              {showVersus ? (
                <div className="flex justify-end">
                  <SignificanceBadge stat={significance} />
                </div>
              ) : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function fixOverlapMetricKey(measure: FixOverlapMeasureKey): string {
  if (measure === "recall") return "fixOverlapVsGoldRecall";
  if (measure === "precision") return "fixOverlapVsGoldPrecision";
  return "fixOverlapVsGoldF1";
}

function FixOverlapDumbbellChart({
  baseline,
  treatment,
  deltaDisplayMode,
}: {
  baseline: ComparisonCard["variants"][number];
  treatment: ComparisonCard["variants"][number];
  deltaDisplayMode: DeltaDisplayMode;
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
              <FixOverlapDumbbellValues
                baselineLabel={formatFixOverlapChartValue(baselineValue, deltaDisplayMode)}
                treatmentLabel={formatFixOverlapChartValue(treatmentValue, deltaDisplayMode)}
                baselineValue={baselineValue}
                treatmentValue={treatmentValue}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}

function formatFixOverlapChartValue(value: number, displayMode: DeltaDisplayMode): string {
  return displayMode === "percent" ? `${value.toFixed(1)}%` : value.toFixed(1);
}

function FixOverlapDumbbellValues({
  baselineLabel,
  treatmentLabel,
  baselineValue,
  treatmentValue,
}: {
  baselineLabel: string;
  treatmentLabel: string;
  baselineValue: number;
  treatmentValue: number;
}) {
  const delta = treatmentValue - baselineValue;
  const Icon = delta > 0 ? TrendingUp : delta < 0 ? TrendingDown : Minus;
  const tone = delta === 0 ? "text-muted-foreground" : deltaIndicatorClassName(deltaTone("higher", delta));

  return (
    <div className="flex items-center justify-end gap-1.5 text-sm tabular-nums text-muted-foreground">
      <span>{baselineLabel}</span>
      <Icon className={cn("h-4 w-4 shrink-0", tone)} aria-label="Baseline to treatment" />
      <span className="font-medium text-foreground">{treatmentLabel}</span>
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
      <LanguageMetricsSection comparison={comparison} />
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
type ContextSummaryKey = ContextMeasureKey | "trajectoryGoldFound";
type ContextAggregateMeasure = { key: ContextSummaryKey; label: string; shortLabel: string; explanation: string };

const contextAggregateMeasures: ContextAggregateMeasure[] = [
  {
    key: "f1",
    label: "Macro Context F1",
    shortLabel: "Macro F1",
    explanation: "Macro-average final-context retrieval F1 across file, block, line, and symbol levels.",
  },
  {
    key: "trajectoryGoldFound",
    label: "Gold Found",
    shortLabel: "Gold Found",
    explanation: "Macro-average terminal cumulative trajectory coverage across file, block, line, and symbol levels.",
  },
  {
    key: "recall",
    label: "Recall",
    shortLabel: "Recall",
    explanation: "Macro-average final-context recall: of the gold context, how much remained in the final retrieved context.",
  },
  {
    key: "precision",
    label: "Precision",
    shortLabel: "Precision",
    explanation: "Macro-average final-context precision: of the final retrieved context, how much was gold.",
  },
];

const contextMetricAggregateMeasures = contextAggregateMeasures.filter((measure) => measure.key !== "trajectoryGoldFound");
const contextGoldFoundAggregateMeasures = contextAggregateMeasures.filter((measure) => measure.key === "trajectoryGoldFound");

const contextSummaryMeasures: Array<{ key: ContextMeasureKey; label: string; shortLabel: string; explanation: string }> = [
  {
    key: "f1",
    label: "F1",
    shortLabel: "F1",
    explanation: "Macro-average final-context retrieval F1 across file, block, line, and symbol levels.",
  },
  {
    key: "recall",
    label: "Recall",
    shortLabel: "Recall",
    explanation: "Macro-average final-context recall: of the gold context, how much remained in the final retrieved context.",
  },
  {
    key: "precision",
    label: "Precision",
    shortLabel: "Precision",
    explanation: "Macro-average retrieved-context precision across file, block, line, and symbol levels: of the context the model viewed, how much was gold.",
  },
];

const contextLevelRows: Array<{ key: ContextLevelKey; label: string; explanation: string }> = [
  {
    key: "file",
    label: "File Level",
    explanation: "F1, recall, and precision over files in the final retrieved context.",
  },
  {
    key: "block",
    label: "Block Level",
    explanation: "F1, recall, and precision over final retrieved code spans. This is the local span metric reported with ContextBench block-level terminology.",
  },
  {
    key: "line",
    label: "Line Level",
    explanation: "F1, recall, and precision over final retrieved line intervals.",
  },
  {
    key: "symbol",
    label: "Symbol Level",
    explanation: "Supplemental fork metric over retrieved functions, classes, methods, or other named code entities.",
  },
];

function ContextLevelSampleSize({
  variants,
  level,
  taskCount,
}: {
  variants: ComparisonCard["variants"];
  level: ContextLevelKey;
  taskCount: number;
}) {
  const sampleSizes = Array.from(
    new Set(
      variants
        .map((variant) => variant.results.quality.contextLevels?.[level]?.n)
        .filter((value): value is number => typeof value === "number"),
    ),
  ).sort((left, right) => left - right);
  if (sampleSizes.length === 0) return null;
  // Only call out levels where instances were excluded for having no gold.
  if (sampleSizes.every((value) => value >= taskCount)) return null;
  const label = sampleSizes.length === 1 ? `n=${sampleSizes[0]}` : `n=${sampleSizes[0]}–${sampleSizes[sampleSizes.length - 1]}`;
  return (
    <span
      className="text-xs font-normal text-muted-foreground"
      title="Number of tasks with gold context at this granularity; tasks without gold are excluded from the macro average."
    >
      {label}
    </span>
  );
}

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

  const values = (["file", "block", "line", "symbol"] as const)
    .map((level) => parseFloatMetric(contextMeasureValue(variant, level, measure)))
    .filter((value): value is number => value !== null);
  return values.length === 4 ? (values.reduce((sum, value) => sum + value, 0) / values.length).toFixed(3) : "—";
}

function contextTopSummaryValue(variant: ComparisonCard["variants"][number], measure: ContextSummaryKey): string {
  if (measure === "trajectoryGoldFound") return variant.results.quality.trajectoryGoldFound ?? trajectoryGoldFoundAggregateValue(variant) ?? "—";
  return contextAggregateValue(variant, measure);
}

function trajectoryGoldFoundAggregateValue(variant: ComparisonCard["variants"][number]): string | null {
  const values = (["file", "block", "line", "symbol"] as const)
    .map((level) => contextLevelRecallPercent(variant, level))
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  return values.length > 0 ? (values.reduce((sum, value) => sum + value, 0) / values.length).toFixed(3) : null;
}

function contextSummaryMetricKey(measure: ContextSummaryKey): string {
  if (measure === "trajectoryGoldFound") return "trajectoryGoldFound";
  if (measure === "recall") return "contextRecall";
  if (measure === "precision") return "contextPrecision";
  return "contextF1";
}

function contextLevelMetricKey(level: ContextLevelKey, measure: ContextMeasureKey): string {
  return `context.${level}.${measure}`;
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
  measure: ContextSummaryKey,
  displayMode: DeltaDisplayMode,
) {
  if (measure === "trajectoryGoldFound") {
    return contextValueDelta(contextTopSummaryValue(baseline, measure), contextTopSummaryValue(treatment, measure), displayMode);
  }
  return contextValueDelta(contextAggregateValue(baseline, measure), contextAggregateValue(treatment, measure), displayMode);
}

function contextLevelRecallPercent(variant: ComparisonCard["variants"][number], level: ContextLevelKey): number | null {
  const trajectoryValue = terminalContextTrajectoryCoveragePercent(variant, level);
  if (trajectoryValue !== null) return trajectoryValue;

  const exportedGoldFound = variant.results.quality.trajectoryContextLevels?.[level]?.goldFound;
  const exportedValue = exportedGoldFound ? parseFloatMetric(exportedGoldFound) : null;
  if (exportedValue !== null) return Math.min(Math.max(exportedValue, 0), 1);
  return null;
}

function terminalContextTrajectoryCoveragePercent(
  variant: ComparisonCard["variants"][number],
  level: ContextLevelKey,
): number | null {
  const trajectoryLevel = level === "block" ? "span" : level;
  const values = (variant.instances ?? [])
    .map((instance) => terminalTrajectoryCoverage(instance, trajectoryLevel))
    .filter((value): value is number => value !== null);
  if (values.length === 0) return null;
  const average = values.reduce((sum, value) => sum + value, 0) / values.length;
  return Math.min(Math.max(average, 0), 1);
}

function contextGoldFoundDelta(
  baseline: ComparisonCard["variants"][number],
  treatment: ComparisonCard["variants"][number],
  level: ContextLevelKey,
  displayMode: DeltaDisplayMode,
) {
  const baselineValue = contextLevelRecallPercent(baseline, level);
  const treatmentValue = contextLevelRecallPercent(treatment, level);
  return contextValueDelta(
    baselineValue === null ? "—" : baselineValue.toFixed(4),
    treatmentValue === null ? "—" : treatmentValue.toFixed(4),
    displayMode,
  );
}

function ContextGoldFoundByLevel({
  comparison,
  variants,
  showDeltas,
  showVersus,
  deltaDisplayMode,
}: {
  comparison: ComparisonCard;
  variants: ComparisonCard["variants"];
  showDeltas: boolean | null;
  showVersus: boolean | null;
  deltaDisplayMode: DeltaDisplayMode;
}) {
  const comparisonPair = getComparisonPair(comparison);
  const displayedVariants = showVersus && comparisonPair ? [comparisonPair.baseline, comparisonPair.treatment] : variants;

  return (
    <div className="rounded-md border p-4">
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <h3 className="text-base font-semibold tracking-tight">Gold Context Found by Level</h3>
        <HelpIcon
          label="Gold Context Found by Level"
          explanation="Gold Found is terminal cumulative trajectory coverage: the percentage of gold files, blocks, lines, or symbols found by the end of the retrieval trajectory."
        />
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        {contextLevelRows.map((level) => (
          <div key={level.key} className="space-y-2">
            <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground">
              <span>{level.label}</span>
              <HelpIcon label={level.label} explanation={level.explanation} />
            </div>
            <div className="space-y-2">
              {displayedVariants.map((variant) => {
                const recall = contextLevelRecallPercent(variant, level.key);
                const delta =
                  showDeltas && comparisonPair && variant.label !== comparisonPair.baseline.label
                    ? contextGoldFoundDelta(comparisonPair.baseline, variant, level.key, deltaDisplayMode)
                    : null;

                return (
                  <div key={variant.label} className="grid gap-x-3 gap-y-2 sm:grid-cols-[10rem_minmax(0,1fr)_max-content] sm:items-center">
                    <div className="truncate text-sm text-muted-foreground">{variant.name}</div>
                    <div className="min-w-0 overflow-hidden rounded-full bg-muted">
                      <div
                        className={cn(
                          "h-3 rounded-full",
                          comparisonPair && variant.label === comparisonPair.baseline.label ? "bg-muted-foreground/55" : "bg-primary",
                        )}
                        style={{ width: `${(recall ?? 0) * 100}%` }}
                      />
                    </div>
                    <div className="flex min-w-[5.5rem] items-center justify-end gap-2 whitespace-nowrap text-sm tabular-nums">
                      <span>{recall === null ? "—" : formatPercent(recall)}</span>
                      {delta ? <DeltaIndicator label={delta.label} delta={delta.delta} tone={delta.tone} /> : null}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ContextSummaryCard({
  label,
  explanation,
  value,
  delta,
  significance,
}: {
  label: string;
  explanation: string;
  value: string;
  delta?: ReturnType<typeof contextValueDelta>;
  significance?: PairedSignificance | null;
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
        {delta ? (
          <div className="flex flex-wrap items-center gap-2">
            <DeltaIndicator label={delta.label} delta={delta.delta} tone={delta.tone} />
            <SignificanceBadge stat={significance} />
          </div>
        ) : null}
      </div>
    </div>
  );
}

function ContextSummaryVersusCard({
  label,
  explanation,
  baselineValue,
  treatmentValue,
  significance,
}: {
  label: string;
  explanation: string;
  baselineValue: string;
  treatmentValue: string;
  significance?: PairedSignificance | null;
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
      <div className="mt-3 flex justify-end">
        <SignificanceBadge stat={significance} />
      </div>
    </div>
  );
}

function ContextAggregateSummaryGrid({
  comparison,
  measures,
  variants,
  summaryVariants,
  showDeltas,
  showVersus,
  deltaDisplayMode,
  cardGridClassName = "sm:grid-cols-2 xl:grid-cols-4",
}: {
  comparison: ComparisonCard;
  measures: ContextAggregateMeasure[];
  variants: ComparisonCard["variants"];
  summaryVariants: ComparisonCard["variants"];
  showDeltas: boolean;
  showVersus: boolean;
  deltaDisplayMode: DeltaDisplayMode;
  cardGridClassName?: string;
}) {
  const comparisonPair = getComparisonPair(comparison);

  return (
    <div className={cn("grid gap-5", summaryVariants.length > 1 ? "md:grid-cols-2" : "grid-cols-1")}>
      {summaryVariants.map((variant) => (
        <div key={variant.label}>
          {!showDeltas && variants.length > 1 ? (
            <div className="mb-4 text-sm font-medium text-muted-foreground">{variant.name}</div>
          ) : null}
          <div className={cn("grid gap-3", cardGridClassName)}>
            {measures.map((measure) => {
              if (showVersus && comparisonPair) {
                return (
                  <ContextSummaryVersusCard
                    key={measure.key}
                    label={measure.label}
                    explanation={measure.explanation}
                    baselineValue={contextTopSummaryValue(comparisonPair.baseline, measure.key)}
                    treatmentValue={contextTopSummaryValue(comparisonPair.treatment, measure.key)}
                    significance={getMetricSignificance(comparison, contextSummaryMetricKey(measure.key))}
                  />
                );
              }
              return (
                <ContextSummaryCard
                  key={measure.key}
                  label={measure.label}
                  explanation={measure.explanation}
                  value={contextTopSummaryValue(variant, measure.key)}
                  delta={
                    showDeltas && comparisonPair
                      ? contextAggregateDelta(comparisonPair.baseline, variant, measure.key, deltaDisplayMode)
                      : undefined
                  }
                  significance={showDeltas ? getMetricSignificance(comparison, contextSummaryMetricKey(measure.key)) : undefined}
                />
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}

function ContextMeasureCell({
  value,
  delta,
  significance,
  emphasize = false,
  className,
}: {
  value: string;
  delta?: ReturnType<typeof contextMeasureDelta>;
  significance?: PairedSignificance | null;
  emphasize?: boolean;
  className?: string;
}) {
  return (
    <td className={cn("px-3 py-3 text-right", className)}>
      <div className="inline-flex items-center justify-end gap-2">
        <span className={`${emphasize ? "font-medium " : ""}tabular-nums`}>{value}</span>
        {delta ? <DeltaIndicator label={delta.label} delta={delta.delta} tone={delta.tone} /> : null}
        <SignificanceBadge stat={significance} />
      </div>
    </td>
  );
}

function ContextEfficiencyByLevelTable({
  comparison,
  variants,
  showDeltas,
  showVersus,
  deltaDisplayMode,
}: {
  comparison: ComparisonCard;
  variants: ComparisonCard["variants"];
  showDeltas: boolean;
  showVersus: boolean;
  deltaDisplayMode: DeltaDisplayMode;
}) {
  const comparisonPair = getComparisonPair(comparison);
  const showPairwiseTable = showVersus || variants.length > 1;

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[48rem] border-collapse text-base">
        <thead>
          <tr className="border-b text-sm uppercase tracking-wide text-muted-foreground">
            <th className="px-3 py-2 text-left font-medium">Level</th>
            {showPairwiseTable ? (
              contextSummaryMeasures.map((measure, measureIndex) => (
                <Fragment key={measure.key}>
                  {variants.map((variant, variantIndex) => (
                    <th
                      key={variant.label}
                      className={cn(
                        "py-2 text-right font-medium",
                        variantIndex === 0 ? "pl-3 pr-1.5" : "pl-1.5 pr-3",
                      )}
                    >
                      {variantIndex === 0 ? measure.label : ""}
                    </th>
                  ))}
                  {measureIndex < contextSummaryMeasures.length - 1 ? (
                    <th aria-hidden="true" className="w-8 px-0 py-2" />
                  ) : null}
                </Fragment>
              ))
            ) : (
              variants.map((variant) => (
                <Fragment key={variant.label}>
                  <th className="px-3 py-2 text-right font-medium">F1</th>
                  <th className="px-3 py-2 text-right font-medium">Recall</th>
                  <th className="px-3 py-2 text-right font-medium">Precision</th>
                </Fragment>
              ))
            )}
          </tr>
        </thead>
        <tbody>
          {contextLevelRows.map((row) => (
            <tr key={row.key} className="border-b last:border-b-0">
              <th className="px-3 py-3 text-left font-medium">
                <span className="inline-flex items-center gap-2">
                  {row.label}
                  <HelpIcon label={row.label} explanation={row.explanation} />
                  <ContextLevelSampleSize variants={variants} level={row.key} taskCount={comparison.tasks} />
                </span>
              </th>
              {showPairwiseTable ? (
                contextSummaryMeasures.map((measure, measureIndex) => (
                  <Fragment key={measure.key}>
                    {variants.map((variant, variantIndex) => (
                      <ContextMeasureCell
                        key={variant.label}
                        value={contextMeasureValue(variant, row.key, measure.key)}
                        emphasize={measure.key === "f1"}
                        className={variantIndex === 0 ? "pl-3 pr-1.5" : "pl-1.5 pr-3"}
                      />
                    ))}
                    {measureIndex < contextSummaryMeasures.length - 1 ? (
                      <td aria-hidden="true" className="w-8 px-0 py-3" />
                    ) : null}
                  </Fragment>
                ))
              ) : (
                variants.map((variant) => (
                  <Fragment key={variant.label}>
                    <ContextMeasureCell
                      value={contextMeasureValue(variant, row.key, "f1")}
                      delta={
                        showDeltas && comparisonPair
                          ? contextMeasureDelta(comparisonPair.baseline, variant, row.key, "f1", deltaDisplayMode)
                          : undefined
                      }
                      significance={showDeltas ? getMetricSignificance(comparison, contextLevelMetricKey(row.key, "f1")) : undefined}
                      emphasize
                    />
                    <ContextMeasureCell
                      value={contextMeasureValue(variant, row.key, "recall")}
                      delta={
                        showDeltas && comparisonPair
                          ? contextMeasureDelta(comparisonPair.baseline, variant, row.key, "recall", deltaDisplayMode)
                          : undefined
                      }
                      significance={showDeltas ? getMetricSignificance(comparison, contextLevelMetricKey(row.key, "recall")) : undefined}
                    />
                    <ContextMeasureCell
                      value={contextMeasureValue(variant, row.key, "precision")}
                      delta={
                        showDeltas && comparisonPair
                          ? contextMeasureDelta(comparisonPair.baseline, variant, row.key, "precision", deltaDisplayMode)
                          : undefined
                      }
                      significance={showDeltas ? getMetricSignificance(comparison, contextLevelMetricKey(row.key, "precision")) : undefined}
                    />
                  </Fragment>
                ))
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
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
    <>
      <ComparisonSectionShell
        title="Context Metrics"
        collapsible={collapsible}
        defaultOpen={defaultOpen}
      >
        <div className="space-y-6 rounded-lg bg-background p-5">
          <ContextAggregateSummaryGrid
            comparison={comparison}
            measures={contextMetricAggregateMeasures}
            variants={comparison.variants}
            summaryVariants={summaryVariants}
            showDeltas={Boolean(showDeltas)}
            showVersus={Boolean(showVersus)}
            deltaDisplayMode={deltaDisplayMode}
            cardGridClassName="md:grid-cols-3"
          />
          <ContextEfficiencyByLevelTable
            comparison={comparison}
            variants={variants}
            showDeltas={Boolean(showDeltas)}
            showVersus={Boolean(showVersus)}
            deltaDisplayMode={deltaDisplayMode}
          />
        </div>
      </ComparisonSectionShell>
      <ComparisonSectionShell
        title="Gold Context Retrieved"
        collapsible={collapsible}
        defaultOpen={defaultOpen}
      >
        <div className="space-y-6 rounded-lg bg-background p-5">
          <ContextAggregateSummaryGrid
            comparison={comparison}
            measures={contextGoldFoundAggregateMeasures}
            variants={comparison.variants}
            summaryVariants={summaryVariants}
            showDeltas={Boolean(showDeltas)}
            showVersus={Boolean(showVersus)}
            deltaDisplayMode={deltaDisplayMode}
            cardGridClassName="grid-cols-1"
          />
          <ContextGoldFoundByLevel
            comparison={comparison}
            variants={variants}
            showDeltas={Boolean(showDeltas)}
            showVersus={showVersus}
            deltaDisplayMode={deltaDisplayMode}
          />
          <ContextTrajectoryCoverageChart variants={comparison.variants} />
        </div>
      </ComparisonSectionShell>
    </>
  );
}

type ContextTrajectoryLevelKey = "file" | "span" | "line" | "symbol";

type ContextTrajectoryChartRow = {
  step: number;
  [key: string]: number | null;
};

type ContextTrajectoryStep = NonNullable<NonNullable<ComparisonInstance["evaluatedTrajectory"]>["steps"]>[number];

const contextTrajectoryLevels: Array<{ key: ContextTrajectoryLevelKey; label: string }> = [
  { key: "file", label: "File" },
  { key: "span", label: "Block" },
  { key: "line", label: "Line" },
  { key: "symbol", label: "Symbol" },
];

const contextTrajectoryColors = [
  "hsl(var(--chart-1))",
  "hsl(var(--chart-2))",
  "hsl(var(--chart-3))",
  "hsl(var(--chart-4))",
  "hsl(var(--chart-5))",
];

function ContextTrajectoryCoverageChart({ variants }: { variants: ComparisonCard["variants"] }) {
  const [activeLevel, setActiveLevel] = useState<ContextTrajectoryLevelKey>("file");
  const chartData = buildContextTrajectoryRows(variants, activeLevel);
  if (chartData.length === 0) return null;

  const chartConfig = buildContextTrajectoryChartConfig(variants);
  const activeLabel = contextTrajectoryLevels.find((level) => level.key === activeLevel)?.label ?? "File";

  return (
    <div className="mt-6 rounded-md border p-4">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-base font-semibold tracking-tight">Trajectory Gold Context Retrieval</h3>
            <HelpIcon
              label="Trajectory Gold Context Retrieval"
              explanation="Average cumulative gold-context coverage at each scored retrieval step across evaluated instances. When an instance has finished, its latest cumulative coverage is carried forward."
            />
          </div>
        </div>
        <ToggleGroup
          type="single"
          variant="outline"
          value={activeLevel}
          onValueChange={(value) => value && setActiveLevel(value as ContextTrajectoryLevelKey)}
          className="gap-0 rounded-md"
        >
          {contextTrajectoryLevels.map((level, index) => (
            <ToggleGroupItem
              key={level.key}
              value={level.key}
              className={cn(
                "rounded-none bg-background px-3 data-[state=on]:bg-primary data-[state=on]:text-primary-foreground",
                index === 0 ? "rounded-l-md border-r-0" : "",
                index === contextTrajectoryLevels.length - 1 ? "rounded-r-md" : "border-r-0",
              )}
            >
              {level.label}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </div>
      <ChartContainer config={chartConfig} className="h-[320px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 16, right: 16, bottom: 16, left: 0 }}>
            <CartesianGrid vertical={false} />
            <XAxis
              dataKey="step"
              type="number"
              tickLine={false}
              tickMargin={10}
              axisLine={false}
              allowDecimals={false}
            />
            <YAxis
              tickLine={false}
              axisLine={false}
              tickMargin={10}
              width={44}
              domain={[0, 1]}
              ticks={[0, 0.25, 0.5, 0.75, 1]}
              tickFormatter={(value) => `${Math.round(Number(value) * 100)}%`}
            />
            <ChartTooltip
              cursor={false}
              content={
                <ChartTooltipContent
                  labelFormatter={(value) => `Context step ${value} · ${activeLabel}`}
                  formatter={(value) => `${(Number(value) * 100).toFixed(1)}%`}
                />
              }
            />
            <ChartLegend content={<ChartLegendContent />} />
            {variants.map((_variant, index) => (
              <Area
                key={contextTrajectoryVariantKey(index)}
                type="monotone"
                dataKey={contextTrajectoryVariantKey(index)}
                stroke={`var(--color-${contextTrajectoryVariantKey(index)})`}
                fill={`var(--color-${contextTrajectoryVariantKey(index)})`}
                fillOpacity={0.14}
                strokeWidth={2}
                dot={false}
                connectNulls
                isAnimationActive={false}
              />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </ChartContainer>
    </div>
  );
}

function contextTrajectoryVariantKey(index: number): string {
  return `trajectoryVariant${index}`;
}

function buildContextTrajectoryChartConfig(variants: ComparisonCard["variants"]): ChartConfig {
  return Object.fromEntries(
    variants.map((variant, index) => [
      contextTrajectoryVariantKey(index),
      {
        label: variant.name,
        color: contextTrajectoryColors[index % contextTrajectoryColors.length],
      },
    ]),
  ) satisfies ChartConfig;
}

function buildContextTrajectoryRows(
  variants: ComparisonCard["variants"],
  level: ContextTrajectoryLevelKey,
): ContextTrajectoryChartRow[] {
  const steps = new Set<number>();
  for (const variant of variants) {
    for (const instance of variant.instances ?? []) {
      const series = contextTrajectoryCoverageSeries(instance.evaluatedTrajectory?.steps ?? [], level);
      for (let index = 0; index < series.length; index += 1) {
        steps.add(index + 1);
      }
    }
  }

  return Array.from(steps)
    .sort((left, right) => left - right)
    .map((step) => {
      const row: ContextTrajectoryChartRow = { step };
      variants.forEach((variant, variantIndex) => {
        const values = (variant.instances ?? [])
          .map((instance) => contextTrajectoryCoverageAtContextStep(instance, step, level))
          .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
        row[contextTrajectoryVariantKey(variantIndex)] = values.length > 0
          ? Number((values.reduce((sum, value) => sum + value, 0) / values.length).toFixed(4))
          : null;
      });
      return row;
    })
    .filter((row) => variants.some((_variant, index) => typeof row[contextTrajectoryVariantKey(index)] === "number"));
}

function contextTrajectoryCoverageAtContextStep(
  instance: ComparisonInstance,
  targetContextStep: number,
  level: ContextTrajectoryLevelKey,
): number | null {
  if (instance.artifacts?.evaluationStatus && instance.artifacts.evaluationStatus !== "valid") return null;
  if (!instance.evaluatedTrajectory) return null;
  const value = lastContextTrajectoryCoverageAtContextStep(instance.evaluatedTrajectory.steps, targetContextStep, level);
  return value === null ? 0 : value;
}

function lastContextTrajectoryCoverageAtContextStep(
  steps: ContextTrajectoryStep[] | undefined,
  targetContextStep: number,
  level: ContextTrajectoryLevelKey,
): number | null {
  const series = contextTrajectoryCoverageSeries(steps, level);
  if (series.length === 0 || targetContextStep < 1) return null;
  return series[Math.min(targetContextStep, series.length) - 1] ?? null;
}

function contextTrajectoryCoverageSeries(
  steps: ContextTrajectoryStep[] | undefined,
  level: ContextTrajectoryLevelKey,
): number[] {
  const series: number[] = [];
  for (const step of steps ?? []) {
    if (step.isSkillRead) continue;
    const value = step.coverage[level];
    if (typeof value === "number" && Number.isFinite(value)) {
      series.push(value);
    }
  }
  return series;
}
