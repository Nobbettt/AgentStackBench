
import {
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
} from "recharts";

import type { ComparisonCard } from "@/data/comparisons";
import { TooltipProvider } from "@/components/ui/tooltip";
import { getComparisonPair } from "@/components/comparison/format";
import { InstanceResultsSection } from "@/components/comparison/instance-results-section";
import { ComparisonInstanceDetailPage } from "@/components/comparison/instance-detail-page";
import { CorrelationSection } from "@/components/comparison/correlation-section";
import {
  contextRetrievalMetricDefinitions,
  executionMetricDefinitions,
  metricDelta,
  resolutionMetricDefinitions,
  resourceMetricDefinitions,
} from "@/components/comparison/metrics";
import {
  ContextRetrievalMetricSection,
  FixOverlapVsGoldSection,
  LanguageMetricsSection,
  OutcomeBreakdownSection,
  PatchOverlapBetweenVariantsSection,
  ResolutionMetricSection,
  ResolutionSetOverlapSection,
  ResourceUsageMetricSection,
} from "@/components/comparison/metric-sections";
import { getMetricSignificance } from "@/components/comparison/significance";
import { DeltaIndicator, HelpIcon, MetricDirectionBadge, SignificanceBadge } from "@/components/comparison/shared";
import { comparisonHasToolUsage, SkillUsageSection, ToolUsageSection } from "@/components/comparison/usage-sections";
import type { ComparisonResultsViewMode, DeltaDisplayMode, MetricDefinition } from "@/components/comparison/types";
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart";

export { ComparisonInstanceDetailPage };

export type ComparisonResultsTab = "overview" | "execution" | "resolution" | "context" | "correlations" | "languages" | "resources" | "usage" | "tools" | "issues";

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
            <ResolutionMetricSection comparison={comparison} viewMode={viewMode} deltaDisplayMode={deltaDisplayMode} showFixOverlap={false} />
            <ResolutionSetOverlapSection comparison={comparison} />
            <FixOverlapVsGoldSection comparison={comparison} viewMode={viewMode} deltaDisplayMode={deltaDisplayMode} />
            <PatchOverlapBetweenVariantsSection overlap={comparison.fixOverlapBetweenVariants} />
          </>
        ) : null}
        {activeTab === "context" ? (
          <ContextRetrievalMetricSection comparison={comparison} viewMode={viewMode} deltaDisplayMode={deltaDisplayMode} />
        ) : null}
        {activeTab === "correlations" ? (
          <CorrelationSection comparison={comparison} />
        ) : null}
        {activeTab === "languages" ? (
          <LanguageMetricsSection comparison={comparison} />
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

type OverviewRadarMetric = {
  key: string;
  label: string;
  metric: MetricDefinition;
};

type OverviewRadarRow = {
  metric: string;
  fullMetric: string;
  [key: string]: string | number;
};

const overviewRadarMetrics: OverviewRadarMetric[] = [
  {
    key: "officialPassAt1",
    label: "Pass@1",
    metric: resolutionMetricDefinitions.find((metric) => metric.key === "officialPassAt1")!,
  },
  {
    key: "fixOverlapVsGoldF1",
    label: "Patch F1",
    metric: resolutionMetricDefinitions.find((metric) => metric.key === "fixOverlapVsGoldF1")!,
  },
  {
    key: "trajectoryGoldFound",
    label: "Gold Found",
    metric: contextRetrievalMetricDefinitions.find((metric) => metric.key === "trajectoryGoldFound")!,
  },
  {
    key: "contextRecall",
    label: "Context Recall",
    metric: contextRetrievalMetricDefinitions.find((metric) => metric.key === "contextRecall")!,
  },
  {
    key: "contextPrecision",
    label: "Context Precision",
    metric: contextRetrievalMetricDefinitions.find((metric) => metric.key === "contextPrecision")!,
  },
];

const overviewRadarColors = [
  "hsl(var(--chart-1))",
  "hsl(var(--chart-2))",
  "hsl(var(--chart-3))",
  "hsl(var(--chart-4))",
  "hsl(var(--chart-5))",
];

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
      <OverviewCapabilityRadar comparison={comparison} />
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-6 xl:grid-cols-5">
        {overviewMetrics.map(({ category, metric }, index) => {
          const value = metric.value(primaryVariant);
          const baselineValue = showDeltas ? metric.value(comparisonPair.baseline) : null;
          const delta = showDeltas ? metricDelta(metric, comparisonPair.baseline, comparisonPair.treatment, deltaDisplayMode) : null;
          const significance = showDeltas ? getMetricSignificance(comparison, metric.key) : null;
          const centeredGridClassName = overviewMetrics.length === 5
            ? index === 0
              ? "xl:col-start-auto"
              : index === 3
                ? "lg:col-start-2 xl:col-start-auto"
                : ""
            : "";

          return (
            <div key={metric.key} className={`min-w-0 rounded-md border bg-background p-4 lg:col-span-2 xl:col-span-1 ${centeredGridClassName}`}>
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
                  {delta ? (
                    <div className="flex flex-wrap items-center justify-end gap-2">
                      <DeltaIndicator label={delta.label} delta={delta.delta} tone={delta.tone} />
                      <SignificanceBadge stat={significance} />
                    </div>
                  ) : null}
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

function OverviewCapabilityRadar({ comparison }: { comparison: ComparisonCard }) {
  const chartData = buildOverviewRadarRows(comparison);
  if (chartData.length < 3) return null;

  const chartConfig = buildOverviewRadarChartConfig(comparison);

  return (
    <div className="rounded-md border bg-background p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-base font-semibold tracking-tight">Capability Profile</h3>
          <p className="mt-1 text-sm text-muted-foreground">
            Higher is better on every axis; resource usage is shown separately with explicit units.
          </p>
        </div>
        <HelpIcon
          label="Capability Profile"
          explanation="Radar chart comparing bounded quality metrics only: task resolution, patch overlap with gold edits, cumulative gold-context discovery, final retained gold context, and retrieval precision."
        />
      </div>
      <ChartContainer config={chartConfig} className="mt-4 h-[340px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={chartData} margin={{ top: 24, right: 48, bottom: 24, left: 48 }}>
            <ChartTooltip
              cursor={false}
              content={
                <ChartTooltipContent
                  labelKey="fullMetric"
                  formatter={(value) => `${(Number(value) * 100).toFixed(1)}%`}
                />
              }
            />
            <PolarGrid />
            <PolarAngleAxis dataKey="metric" tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }} />
            <PolarRadiusAxis angle={90} domain={[0, 1]} tick={false} axisLine={false} />
            {comparison.variants.map((_variant, index) => (
              <Radar
                key={overviewRadarVariantKey(index)}
                dataKey={overviewRadarVariantKey(index)}
                stroke={`var(--color-${overviewRadarVariantKey(index)})`}
                fill={`var(--color-${overviewRadarVariantKey(index)})`}
                fillOpacity={0.12}
                strokeWidth={2}
                isAnimationActive={false}
              />
            ))}
            <ChartLegend content={<ChartLegendContent />} />
          </RadarChart>
        </ResponsiveContainer>
      </ChartContainer>
    </div>
  );
}

function overviewRadarVariantKey(index: number): string {
  return `radarVariant${index}`;
}

function buildOverviewRadarChartConfig(comparison: ComparisonCard): ChartConfig {
  return Object.fromEntries(
    comparison.variants.map((variant, index) => [
      overviewRadarVariantKey(index),
      {
        label: variant.name,
        color: overviewRadarColors[index % overviewRadarColors.length],
      },
    ]),
  ) satisfies ChartConfig;
}

function buildOverviewRadarRows(comparison: ComparisonCard): OverviewRadarRow[] {
  return overviewRadarMetrics.flatMap((radarMetric) => {
    const rawValues = comparison.variants.map((variant) => radarMetric.metric.parse(radarMetric.metric.value(variant)));
    if (rawValues.some((value) => value === null)) return [];

    const values = rawValues as number[];
    const scores = values.map(normalizeQualityRadarValue);

    return [{
      metric: radarMetric.label,
      fullMetric: radarMetric.label,
      ...Object.fromEntries(scores.map((score, index) => [overviewRadarVariantKey(index), score])),
    }];
  });
}

function normalizeQualityRadarValue(value: number): number {
  const normalized = value > 1 ? value / 100 : value;
  return clampRadarValue(normalized);
}

function clampRadarValue(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}
