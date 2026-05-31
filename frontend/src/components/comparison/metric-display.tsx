// SPDX-License-Identifier: Apache-2.0

import { Minus, TrendingDown, TrendingUp } from "lucide-react";

import type { ComparisonCard } from "@/data/comparisons";
import { deltaIndicatorClassName, getComparisonPair } from "@/components/comparison/format";
import { metricDelta } from "@/components/comparison/metrics";
import { ComparisonSectionShell, DeltaIndicator, HelpIcon, MetricDirectionBadge } from "@/components/comparison/shared";
import type { ComparisonResultsViewMode, DeltaDisplayMode, MetricDefinition } from "@/components/comparison/types";
import { cn } from "@/lib/utils";

export type TreatmentDeltaDisplay = "delta" | "versus";

export type MetricSectionProps = {
  comparison: ComparisonCard;
  viewMode: ComparisonResultsViewMode;
  deltaDisplayMode: DeltaDisplayMode;
  treatmentDeltaDisplay?: TreatmentDeltaDisplay;
  nonGraphDisplay?: boolean;
  collapsible?: boolean;
  defaultOpen?: boolean;
};

export function MetricSection({
  title,
  comparison,
  metrics,
  viewMode,
  deltaDisplayMode,
  treatmentDeltaDisplay = "delta",
  collapsible,
  defaultOpen = true,
}: MetricSectionProps & {
  title: string;
  metrics: MetricDefinition[];
}) {
  const visibleMetrics = metrics.filter((metric) => comparison.variants.some((variant) => metric.value(variant) !== "—"));
  const comparisonPair = getComparisonPair(comparison);

  if (viewMode === "treatment-delta" && comparisonPair) {
    const { baseline, treatment } = comparisonPair;
    const showVersus = treatmentDeltaDisplay === "versus";
    return (
      <ComparisonSectionShell
        title={title}
        collapsible={collapsible}
        defaultOpen={defaultOpen}
      >
        <div className="rounded-lg bg-background p-5">
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {visibleMetrics.map((metric) => (
              showVersus ? (
                <MetricVersusCard key={metric.key} metric={metric} baseline={baseline} treatment={treatment} />
              ) : (
                <MetricCard
                  key={metric.key}
                  metric={metric}
                  value={metric.value(treatment)}
                  delta={metricDelta(metric, baseline, treatment, deltaDisplayMode)}
                />
              )
            ))}
          </div>
        </div>
      </ComparisonSectionShell>
    );
  }

  return (
    <ComparisonSectionShell title={title} collapsible={collapsible} defaultOpen={defaultOpen}>
      <div className="rounded-lg bg-background p-5">
        <div className="grid gap-5 md:grid-cols-2">
          {comparison.variants.map((variant) => (
            <div key={variant.label}>
              <div className="mb-4 text-sm font-medium text-muted-foreground">{variant.name}</div>
              <div className="grid gap-3 sm:grid-cols-2">
                {visibleMetrics.map((metric) => (
                  <MetricCard key={metric.key} metric={metric} value={metric.value(variant)} />
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </ComparisonSectionShell>
  );
}

function MetricCard({
  metric,
  value,
  delta,
}: {
  metric: MetricDefinition;
  value: string;
  delta?: ReturnType<typeof metricDelta>;
}) {
  return (
    <div className="rounded-md border p-4">
      <div className="flex flex-wrap items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground">
        <span>{metric.label}</span>
        <MetricDirectionBadge direction={metric.direction} />
        <HelpIcon label={metric.label} explanation={metric.explanation} />
      </div>
      <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
        <div className="font-medium">{value}</div>
        {delta ? <DeltaIndicator label={delta.label} delta={delta.delta} tone={delta.tone} /> : null}
      </div>
    </div>
  );
}

function MetricVersusCard({
  metric,
  baseline,
  treatment,
}: {
  metric: MetricDefinition;
  baseline: ComparisonCard["variants"][number];
  treatment: ComparisonCard["variants"][number];
}) {
  const baselineValue = metric.value(baseline);
  const treatmentValue = metric.value(treatment);
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
        baselineNumericValue={metric.parse(baselineValue)}
        treatmentNumericValue={metric.parse(treatmentValue)}
      />
    </div>
  );
}

export function MetricVersusValues({
  baselineValue,
  treatmentValue,
  direction,
  baselineNumericValue,
  treatmentNumericValue,
  baselineClassName,
  treatmentClassName,
}: {
  baselineValue: string;
  treatmentValue: string;
  direction: MetricDefinition["direction"];
  baselineNumericValue?: number | null;
  treatmentNumericValue?: number | null;
  baselineClassName?: string;
  treatmentClassName?: string;
}) {
  return (
    <div className="mt-3 grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-center gap-2">
      <MetricVersusValue value={baselineValue} className={baselineClassName} />
      <MetricVersusSeparator
        direction={direction}
        baselineValue={baselineValue}
        treatmentValue={treatmentValue}
        baselineNumericValue={baselineNumericValue}
        treatmentNumericValue={treatmentNumericValue}
      />
      <MetricVersusValue value={treatmentValue} className={treatmentClassName} />
    </div>
  );
}

function MetricVersusValue({ value, className }: { value: string; className?: string }) {
  return (
    <div className="min-w-0 rounded-md bg-muted/30 px-3 py-2 text-center">
      <div className={cn("truncate text-sm font-medium tabular-nums", className)} title={value}>{value}</div>
    </div>
  );
}

function MetricVersusSeparator({
  direction,
  baselineValue,
  treatmentValue,
  baselineNumericValue,
  treatmentNumericValue,
}: {
  direction: MetricDefinition["direction"];
  baselineValue: string;
  treatmentValue: string;
  baselineNumericValue?: number | null;
  treatmentNumericValue?: number | null;
}) {
  const matches = baselineValue === treatmentValue;
  const delta = baselineNumericValue !== null && baselineNumericValue !== undefined && treatmentNumericValue !== null && treatmentNumericValue !== undefined
    ? treatmentNumericValue - baselineNumericValue
    : null;
  const Icon = delta === null
    ? matches ? Minus : direction === "higher" ? TrendingUp : direction === "lower" ? TrendingDown : Minus
    : delta > 0
      ? TrendingUp
      : delta < 0
        ? TrendingDown
        : Minus;
  const improved = delta === null || delta === 0 || direction === "neutral"
    ? null
    : direction === "higher"
      ? delta > 0
      : delta < 0;
  const tone = matches || improved === null ? "text-muted-foreground" : deltaIndicatorClassName(improved ? "success" : "danger");
  return (
    <div className={cn("flex h-8 w-8 items-center justify-center", tone)} aria-label="Baseline to treatment">
      <Icon className="h-5 w-5" />
    </div>
  );
}
