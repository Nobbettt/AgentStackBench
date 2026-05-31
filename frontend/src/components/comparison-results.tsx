
import type { ComparisonCard } from "@/data/comparisons";
import { TooltipProvider } from "@/components/ui/tooltip";
import { getComparisonPair } from "@/components/comparison/format";
import { InstanceResultsSection } from "@/components/comparison/instance-results-section";
import { ComparisonInstanceDetailPage } from "@/components/comparison/instance-detail-page";
import {
  contextRetrievalMetricDefinitions,
  executionMetricDefinitions,
  metricDelta,
  resolutionMetricDefinitions,
  resourceMetricDefinitions,
} from "@/components/comparison/metrics";
import {
  ContextRetrievalMetricSection,
  LanguageMetricsSection,
  OutcomeBreakdownSection,
  PatchOverlapBetweenVariantsSection,
  ResolutionMetricSection,
  ResourceUsageMetricSection,
} from "@/components/comparison/metric-sections";
import { DeltaIndicator, HelpIcon, MetricDirectionBadge } from "@/components/comparison/shared";
import { comparisonHasToolUsage, SkillUsageSection, ToolUsageSection } from "@/components/comparison/usage-sections";
import type { ComparisonResultsViewMode, DeltaDisplayMode, MetricDefinition } from "@/components/comparison/types";

export { ComparisonInstanceDetailPage };

export type ComparisonResultsTab = "overview" | "execution" | "resolution" | "context" | "languages" | "resources" | "usage" | "tools" | "issues";

export function ComparisonResults({
  comparison,
  viewMode,
  deltaDisplayMode,
  activeTab,
}: {
  comparison: ComparisonCard;
  viewMode: ComparisonResultsViewMode;
  deltaDisplayMode: DeltaDisplayMode;
  activeTab: ComparisonResultsTab;
}) {
  return (
    <TooltipProvider>
      <div className="space-y-6">
        {activeTab === "overview" ? (
          <OverviewSection comparison={comparison} viewMode={viewMode} deltaDisplayMode={deltaDisplayMode} />
        ) : null}
        {activeTab === "execution" ? (
          <>
            <OutcomeBreakdownSection comparison={comparison} viewMode={viewMode} deltaDisplayMode={deltaDisplayMode} />
            <ResourceUsageMetricSection comparison={comparison} viewMode={viewMode} deltaDisplayMode={deltaDisplayMode} />
          </>
        ) : null}
        {activeTab === "resolution" ? (
          <>
            <ResolutionMetricSection comparison={comparison} viewMode={viewMode} deltaDisplayMode={deltaDisplayMode} />
            <PatchOverlapBetweenVariantsSection overlap={comparison.fixOverlapBetweenVariants} />
          </>
        ) : null}
        {activeTab === "context" ? (
          <ContextRetrievalMetricSection comparison={comparison} viewMode={viewMode} deltaDisplayMode={deltaDisplayMode} />
        ) : null}
        {activeTab === "languages" ? (
          <LanguageMetricsSection comparison={comparison} viewMode={viewMode} deltaDisplayMode={deltaDisplayMode} />
        ) : null}
        {activeTab === "usage" ? (
          <SkillUsageSection comparison={comparison} viewMode={viewMode} deltaDisplayMode={deltaDisplayMode} />
        ) : null}
        {activeTab === "tools" ? (
          comparisonHasToolUsage(comparison) ? (
            <ToolUsageSection comparison={comparison} viewMode={viewMode} deltaDisplayMode={deltaDisplayMode} />
          ) : (
            <section className="rounded-lg bg-background p-5">
              <h2 className="text-xl font-semibold tracking-tight">Tool Usage</h2>
              <p className="mt-3 text-sm text-muted-foreground">No tools were used.</p>
            </section>
          )
        ) : null}
        {activeTab === "issues" ? (
          <InstanceResultsSection comparison={comparison} viewMode={viewMode} deltaDisplayMode={deltaDisplayMode} />
        ) : null}
      </div>
    </TooltipProvider>
  );
}

const overviewMetricKeys = [
  { key: "officialPassAt1", category: "Resolution", metrics: resolutionMetricDefinitions },
  { key: "fixOverlapVsGoldF1", category: "Patch Quality", metrics: resolutionMetricDefinitions },
  { key: "contextF1", category: "Context", metrics: contextRetrievalMetricDefinitions },
  { key: "averageDuration", category: "Resources", metrics: resourceMetricDefinitions },
  { key: "totalTokens", category: "Resources", metrics: resourceMetricDefinitions },
] as const;

type OverviewMetricCategory = typeof overviewMetricKeys[number]["category"];

function OverviewSection({
  comparison,
  viewMode,
  deltaDisplayMode,
}: {
  comparison: ComparisonCard;
  viewMode: ComparisonResultsViewMode;
  deltaDisplayMode: DeltaDisplayMode;
}) {
  const comparisonPair = getComparisonPair(comparison);
  const showDeltas = viewMode === "treatment-delta" && comparisonPair;
  const primaryVariant = showDeltas ? comparisonPair.treatment : comparison.variants[0];
  const visibleVariants = showDeltas ? [comparisonPair.treatment] : comparison.variants;
  if (!primaryVariant) return null;

  const overviewMetrics: { category: OverviewMetricCategory; metric: MetricDefinition }[] = overviewMetricKeys.flatMap((entry) => {
    const metric = entry.metrics.find((candidate) => candidate.key === entry.key);
    if (!metric || !comparison.variants.some((variant) => metric.value(variant) !== "—")) {
      return [];
    }

    return [{ category: entry.category, metric }];
  });

  return (
    <section className="space-y-4">
      <div>
        <h2 className="text-xl font-semibold tracking-tight">Overview</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Quick glance at the main resolution, context, execution, and resource metrics.
        </p>
      </div>
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {overviewMetrics.map(({ category, metric }) => {
          const value = metric.value(primaryVariant);
          const baselineValue = showDeltas ? metric.value(comparisonPair.baseline) : null;
          const delta = showDeltas ? metricDelta(metric, comparisonPair.baseline, comparisonPair.treatment, deltaDisplayMode) : null;

          return (
            <div key={metric.key} className="rounded-md border bg-background p-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="text-xs uppercase tracking-wide text-muted-foreground">{category}</div>
                <MetricDirectionBadge direction={metric.direction} />
              </div>
              <div className="mt-3 flex items-center gap-2 text-sm font-medium text-muted-foreground">
                <span>{metric.label}</span>
                <HelpIcon label={metric.label} explanation={metric.explanation} />
              </div>
              {showDeltas ? (
                <div className="mt-3 flex flex-wrap items-end justify-between gap-3">
                  <div>
                    <div className="text-2xl font-semibold tabular-nums">{value}</div>
                    {baselineValue ? (
                      <div className="mt-1 text-xs text-muted-foreground">
                        Baseline {baselineValue}
                      </div>
                    ) : null}
                  </div>
                  {delta ? <DeltaIndicator label={delta.label} delta={delta.delta} tone={delta.tone} /> : null}
                </div>
              ) : (
                <div className="mt-3 space-y-2">
                  {visibleVariants.map((variant) => (
                    <div key={variant.label} className="flex items-center justify-between gap-3 text-sm">
                      <span className="truncate text-muted-foreground">{variant.name}</span>
                      <span className="font-semibold tabular-nums text-foreground">{metric.value(variant)}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
