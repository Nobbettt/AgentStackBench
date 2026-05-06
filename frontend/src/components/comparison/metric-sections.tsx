
import { completedRunsForOutcome, partialRunsForOutcome, type ComparisonCard } from "@/data/comparisons";
import { formatMetric, formatOptionalOverlapPercent, getComparisonPair } from "@/components/comparison/format";
import {
  contextRetrievalMetricDefinitions,
  executionMetricDefinitions,
  metricDelta,
  outcomeDelta,
  resourceMetricDefinitions,
  resolutionMetricDefinitions,
} from "@/components/comparison/metrics";
import { ComparisonSectionShell, DeltaIndicator, DeltaSectionLabel, HelpIcon, MetricDirectionBadge } from "@/components/comparison/shared";
import type { ComparisonResultsViewMode, DeltaDisplayMode, MetricDefinition } from "@/components/comparison/types";
import { cn } from "@/lib/utils";

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

function MetricSection({
  title,
  comparison,
  metrics,
  viewMode,
  deltaDisplayMode,
  collapsible,
  defaultOpen = true,
}: {
  title: string;
  comparison: ComparisonCard;
  metrics: MetricDefinition[];
  viewMode: ComparisonResultsViewMode;
  deltaDisplayMode: DeltaDisplayMode;
  collapsible?: boolean;
  defaultOpen?: boolean;
}) {
  const visibleMetrics = metrics.filter((metric) => comparison.variants.some((variant) => metric.value(variant) !== "—"));
  const comparisonPair = getComparisonPair(comparison);

  if (viewMode === "treatment-delta" && comparisonPair) {
    const { baseline, treatment } = comparisonPair;
    return (
      <ComparisonSectionShell
        title={title}
        collapsible={collapsible}
        defaultOpen={defaultOpen}
        headerAside={<DeltaSectionLabel baseline={baseline} treatment={treatment} />}
      >
        <div className={cn(collapsible ? "rounded-lg bg-background p-5" : "rounded-lg border bg-background p-5")}>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {visibleMetrics.map((metric) => (
              <MetricCard
                key={metric.key}
                metric={metric}
                value={metric.value(treatment)}
                delta={metricDelta(metric, baseline, treatment, deltaDisplayMode)}
              />
            ))}
          </div>
        </div>
      </ComparisonSectionShell>
    );
  }

  return (
    <ComparisonSectionShell title={title} collapsible={collapsible} defaultOpen={defaultOpen}>
      <div className={cn(collapsible ? "rounded-lg bg-background p-5" : "rounded-lg border bg-background p-5")}>
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

function isFullyCompletedVariant(variant: ComparisonCard["variants"][number], comparison: ComparisonCard): boolean {
  const totalTasks = variant.instances?.length ?? comparison.tasks;
  return (
    totalTasks > 0 &&
    completedRunsForOutcome(variant.results.outcome) === totalTasks &&
    partialRunsForOutcome(variant.results.outcome) === 0 &&
    variant.results.outcome.failures === 0
  );
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
  const variants = viewMode === "treatment-delta" && comparisonPair ? [comparisonPair.treatment] : comparison.variants;
  const headerAside = viewMode === "treatment-delta" && comparisonPair
    ? <DeltaSectionLabel baseline={comparisonPair.baseline} treatment={comparisonPair.treatment} />
    : undefined;

  return (
    <ComparisonSectionShell title="Execution Outcomes" collapsible={collapsible} headerAside={headerAside}>
      <div className={cn(collapsible ? "rounded-lg bg-background p-5" : "rounded-lg border bg-background p-5")}>
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
                    const delta = comparisonPair ? outcomeDelta(item.name, comparisonPair.baseline, comparisonPair.treatment, deltaDisplayMode) : null;
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
        <MetricSection
          title="Execution Metrics"
          comparison={comparison}
          metrics={executionMetricDefinitions}
          viewMode={viewMode}
          deltaDisplayMode={deltaDisplayMode}
          collapsible={collapsible}
          defaultOpen={false}
        />
      ) : null}
      <MetricSection title="Resolution Metrics" comparison={comparison} metrics={resolutionMetricDefinitions} viewMode={viewMode} deltaDisplayMode={deltaDisplayMode} collapsible={collapsible} />
      <MetricSection title="Context Retrieval Metrics" comparison={comparison} metrics={contextRetrievalMetricDefinitions} viewMode={viewMode} deltaDisplayMode={deltaDisplayMode} collapsible={collapsible} />
      <PatchOverlapBetweenVariantsSection overlap={comparison.fixOverlapBetweenVariants} collapsible={collapsible} />
      <MetricSection title="Resource Usage Metrics" comparison={comparison} metrics={resourceMetricDefinitions} viewMode={viewMode} deltaDisplayMode={deltaDisplayMode} collapsible={collapsible} />
      <ContextSummary comparison={comparison} />
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

  return (
    <ComparisonSectionShell title="Patch Overlap" collapsible={collapsible} defaultOpen={false}>
      <div className={cn(collapsible ? "rounded-lg bg-background p-5" : "rounded-lg border bg-background p-5")}>
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

function ContextSummary({ comparison }: { comparison: ComparisonCard }) {
  const byLanguage = new Map<string, number[]>();
  for (const variant of comparison.variants) {
    for (const instance of variant.instances ?? []) {
      if (typeof instance.trajectory.efficiency !== "number") continue;
      const values = byLanguage.get(instance.language) ?? [];
      values.push(instance.trajectory.efficiency);
      byLanguage.set(instance.language, values);
    }
  }

  if (byLanguage.size === 0) return null;
  return (
    <ComparisonSectionShell title="Language Metrics" collapsible defaultOpen={false}>
      <div className="rounded-lg bg-background p-5">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from(byLanguage.entries()).map(([language, values]) => (
            <div key={language} className="rounded-md border p-4">
              <div className="text-xs uppercase tracking-wide text-muted-foreground">{language}</div>
              <div className="mt-2 text-sm font-medium">{values.length} runs</div>
              <div className="mt-1 text-xs text-muted-foreground">
                Avg. efficiency {formatMetric(values.reduce((sum, value) => sum + value, 0) / values.length)}
              </div>
            </div>
          ))}
        </div>
      </div>
    </ComparisonSectionShell>
  );
}
