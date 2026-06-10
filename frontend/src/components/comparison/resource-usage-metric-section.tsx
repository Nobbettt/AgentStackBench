// SPDX-License-Identifier: Apache-2.0

import { Area, AreaChart, CartesianGrid, ResponsiveContainer, XAxis, YAxis } from "recharts";

import type { ComparisonCard } from "@/data/comparisons";
import {
  deltaTone,
  formatCompactMagnitude,
  formatDurationMs,
  formatPercent,
  formatPercentDelta,
  formatSignedFixed,
  formatTokens,
  getComparisonPair,
} from "@/components/comparison/format";
import { metricDelta, resourceMetricDefinitions } from "@/components/comparison/metrics";
import { MetricVersusValues, type MetricSectionProps, type TreatmentDeltaDisplay } from "@/components/comparison/metric-display";
import { getMetricSignificance } from "@/components/comparison/significance";
import { ComparisonSectionShell, DeltaIndicator, HelpIcon, MetricDirectionBadge, SignificanceBadge } from "@/components/comparison/shared";
import type { DeltaDisplayMode, MetricDefinition } from "@/components/comparison/types";
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart";

type ResourceMetricKey = "durationMs" | "totalTokens" | "steps" | "rawTraceEvents" | "rawAgentActions";

type ResourceDistributionMetric = {
  key: ResourceMetricKey;
  metricKey: string;
  valueLabel: string;
  getValue: (instance: ComparisonInstanceForChart) => number | null | undefined;
  formatValue: (value: number) => string;
};

type ComparisonInstanceForChart = NonNullable<ComparisonCard["variants"][number]["instances"]>[number];

const resourceDistributionMetrics: ResourceDistributionMetric[] = [
  {
    key: "steps",
    metricKey: "averageSteps",
    valueLabel: "Scored Retrieval Steps",
    getValue: (instance) => instance.trajectory.steps,
    formatValue: formatSteps,
  },
  {
    key: "durationMs",
    metricKey: "averageDuration",
    valueLabel: "Runtime",
    getValue: (instance) => instance.resources.durationMs,
    formatValue: formatDurationMs,
  },
  {
    key: "rawTraceEvents",
    metricKey: "rawTraceEvents",
    valueLabel: "Raw Trace Events",
    getValue: (instance) => instance.resources.rawTraceEvents,
    formatValue: formatCount,
  },
  {
    key: "rawAgentActions",
    metricKey: "rawAgentActions",
    valueLabel: "Raw Agent Actions",
    getValue: (instance) => instance.resources.rawAgentActions,
    formatValue: formatCount,
  },
  {
    key: "totalTokens",
    metricKey: "totalTokens",
    valueLabel: "Tokens",
    getValue: (instance) => instance.resources.totalTokens,
    formatValue: formatTokens,
  },
];

const resourceDistributionColors = [
  "hsl(var(--chart-1))",
  "hsl(var(--chart-2))",
  "hsl(var(--chart-3))",
  "hsl(var(--chart-4))",
  "hsl(var(--chart-5))",
];

type ResourceUsageMetricSectionProps = MetricSectionProps & {
  metricDefinitions?: MetricDefinition[];
};

export function ResourceUsageMetricSection(props: ResourceUsageMetricSectionProps) {
  return <CombinedResourceUsageSection {...props} />;
}

function resourceVariantKey(index: number): string {
  return `variant${index}`;
}

function formatSteps(value: number): string {
  return `${Number.isInteger(value) ? value.toFixed(0) : value.toFixed(1)} retrieval steps`;
}

function formatCount(value: number): string {
  return Number.isInteger(value) ? value.toFixed(0) : value.toFixed(1);
}

function getResourceValues(
  variant: ComparisonCard["variants"][number],
  metric: ResourceDistributionMetric,
): number[] {
  return (variant.instances ?? [])
    .map((instance) => metric.getValue(instance))
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value) && value > 0)
    .sort((left, right) => left - right);
}

type ResourceDensityRow = {
  value: number;
  [key: string]: string | number;
};

function standardDeviation(values: number[]): number {
  if (values.length < 2) return 0;
  const average = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - average) ** 2, 0) / (values.length - 1);
  return Math.sqrt(variance);
}

function estimateBandwidth(values: number[], domainSpan: number): number {
  const fallback = Math.max(domainSpan / 10, 1);
  if (values.length < 2) return fallback;
  const deviation = standardDeviation(values);
  if (deviation === 0) return fallback;
  return Math.max(1.06 * deviation * values.length ** -0.2, fallback / 5);
}

function gaussianKernel(value: number): number {
  return Math.exp(-0.5 * value * value);
}

function densityAt(xValue: number, values: number[], bandwidth: number): number {
  if (values.length === 0) return 0;
  const density = values.reduce((sum, value) => sum + gaussianKernel((xValue - value) / bandwidth), 0);
  return density / (values.length * bandwidth);
}

function getDensityTicks(domain: [number, number]): number[] {
  const [start, end] = domain;
  return [start, start + (end - start) * 0.25, start + (end - start) * 0.5, start + (end - start) * 0.75, end];
}

function buildResourceDensityRows({
  metric,
  valuesByVariant,
}: {
  metric: ResourceDistributionMetric;
  valuesByVariant: number[][];
}): { rows: ResourceDensityRow[]; domain: [number, number] } {
  const allValues = valuesByVariant.flat();
  if (allValues.length === 0) return { rows: [], domain: [0, 1] };

  const minValue = Math.min(...allValues);
  const maxValue = Math.max(...allValues);
  const rawSpan = maxValue - minValue;
  const metricFallback = metric.key === "durationMs" ? 60_000 : metric.key === "totalTokens" ? 100_000 : 4;
  const initialSpan = rawSpan > 0 ? rawSpan : Math.max(Math.abs(maxValue) * 0.2, metricFallback);
  const bandwidths = valuesByVariant.map((values) => estimateBandwidth(values, initialSpan));
  const largestBandwidth = Math.max(...bandwidths, initialSpan / 10);
  const domainStart = Math.max(0, minValue - largestBandwidth * 3);
  const domainEnd = maxValue + largestBandwidth * 3;
  const sampleCount = 90;
  const step = (domainEnd - domainStart) / (sampleCount - 1);
  const rawDensitiesByVariant = valuesByVariant.map((values, variantIndex) => {
    const bandwidth = bandwidths[variantIndex];
    return Array.from({ length: sampleCount }, (_entry, index) => {
      const xValue = domainStart + step * index;
      return densityAt(xValue, values, bandwidth);
    });
  });
  const peakByVariant = rawDensitiesByVariant.map((densities) => Math.max(...densities, 0));
  const rows = Array.from({ length: sampleCount }, (_entry, index) => {
    const xValue = domainStart + step * index;
    const row: ResourceDensityRow = { value: xValue };
    rawDensitiesByVariant.forEach((densities, variantIndex) => {
      const peak = peakByVariant[variantIndex];
      row[resourceVariantKey(variantIndex)] = peak > 0 ? Number((densities[index] / peak).toFixed(4)) : 0;
    });
    return row;
  });

  return { rows, domain: [domainStart, domainEnd] };
}

function buildResourceChartConfig(variants: ComparisonCard["variants"]): ChartConfig {
  return Object.fromEntries(
    variants.map((variant, index) => [
      resourceVariantKey(index),
      {
        label: variant.name,
        color: resourceDistributionColors[index % resourceDistributionColors.length],
      },
    ]),
  ) satisfies ChartConfig;
}

function CombinedResourceUsageSection({
  comparison,
  viewMode,
  deltaDisplayMode,
  treatmentDeltaDisplay = "delta",
  nonGraphDisplay = false,
  collapsible = false,
  defaultOpen = true,
  metricDefinitions = resourceMetricDefinitions,
}: ResourceUsageMetricSectionProps) {
  const comparisonPair = getComparisonPair(comparison);
  const showDeltas = viewMode === "treatment-delta" && comparisonPair;
  const showVersus = showDeltas && treatmentDeltaDisplay === "versus";
  const summaryVariants = showDeltas && !showVersus ? [comparisonPair.treatment] : comparison.variants;
  const graphVariants = comparison.variants;
  const panels = resourceDistributionMetrics
    .map((distributionMetric) => ({
      distributionMetric,
      metricDefinition: metricDefinitions.find((metric) => metric.key === distributionMetric.metricKey),
    }))
    .filter((panel): panel is { distributionMetric: ResourceDistributionMetric; metricDefinition: MetricDefinition } =>
      Boolean(panel.metricDefinition),
    )
    .filter(({ distributionMetric, metricDefinition }) =>
      graphVariants.some((variant) => metricDefinition.value(variant) !== "—" || getResourceValues(variant, distributionMetric).length > 0),
    );

  if (panels.length === 0) return null;

  return (
    <>
      <ResourceUsageSummary
        comparison={comparison}
        comparisonPair={comparisonPair}
        showDeltas={Boolean(showDeltas)}
        treatmentDeltaDisplay={treatmentDeltaDisplay}
        deltaDisplayMode={deltaDisplayMode}
        panels={panels}
        variants={summaryVariants}
      />
      <TokenUsageSection
        comparisonPair={comparisonPair}
        showDeltas={Boolean(showDeltas)}
        treatmentDeltaDisplay={treatmentDeltaDisplay}
        deltaDisplayMode={deltaDisplayMode}
        variants={graphVariants}
      />
      {nonGraphDisplay ? null : panels.map(({ distributionMetric, metricDefinition }) => (
        <ComparisonSectionShell
          key={distributionMetric.key}
          title={metricDefinition.label}
          collapsible={collapsible}
          defaultOpen={defaultOpen}
          headerInline={<HelpIcon label={metricDefinition.label} explanation={metricDefinition.explanation} />}
        >
          <div className="rounded-lg bg-background p-5">
            <ResourceUsagePanel
              comparison={comparison}
              comparisonPair={comparisonPair}
              showDeltas={Boolean(showDeltas)}
              treatmentDeltaDisplay={treatmentDeltaDisplay}
              deltaDisplayMode={deltaDisplayMode}
              distributionMetric={distributionMetric}
              metricDefinition={metricDefinition}
              variants={graphVariants}
            />
          </div>
        </ComparisonSectionShell>
      ))}
    </>
  );
}

function ResourceUsageSummary({
  comparison,
  comparisonPair,
  showDeltas,
  treatmentDeltaDisplay,
  deltaDisplayMode,
  panels,
  variants,
}: {
  comparison: ComparisonCard;
  comparisonPair: ReturnType<typeof getComparisonPair>;
  showDeltas: boolean;
  treatmentDeltaDisplay: TreatmentDeltaDisplay;
  deltaDisplayMode: DeltaDisplayMode;
  panels: Array<{ distributionMetric: ResourceDistributionMetric; metricDefinition: MetricDefinition }>;
  variants: ComparisonCard["variants"];
}) {
  const primaryVariant = comparisonPair?.treatment ?? variants[0];
  if (!primaryVariant) return null;

  return (
    <ComparisonSectionShell title="Resource Usage Summary">
      <div className="rounded-lg bg-background p-5">
        <div className="grid gap-3 sm:grid-cols-3">
          {panels.map(({ distributionMetric, metricDefinition }) => {
            const delta = showDeltas && comparisonPair ? metricDelta(metricDefinition, comparisonPair.baseline, comparisonPair.treatment, deltaDisplayMode) : null;
            const showVersus = showDeltas && treatmentDeltaDisplay === "versus" && comparisonPair;
            const significance = comparisonPair ? getMetricSignificance(comparison, metricDefinition.key) : null;
            return (
              <div key={distributionMetric.key} className="rounded-md border p-4">
                <div className="flex flex-wrap items-center gap-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                  <span>{metricDefinition.label}</span>
                  <MetricDirectionBadge direction={metricDefinition.direction} />
                  <HelpIcon label={metricDefinition.label} explanation={metricDefinition.explanation} />
                </div>
                {showVersus ? (
                  <MetricVersusValues
                    baselineValue={metricDefinition.value(comparisonPair.baseline)}
                    treatmentValue={metricDefinition.value(comparisonPair.treatment)}
                    direction={metricDefinition.direction}
                    baselineNumericValue={metricDefinition.parse(metricDefinition.value(comparisonPair.baseline))}
                    treatmentNumericValue={metricDefinition.parse(metricDefinition.value(comparisonPair.treatment))}
                  />
                ) : showDeltas ? (
                  <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
                    <div className="text-2xl font-semibold tabular-nums">{metricDefinition.value(primaryVariant)}</div>
                    {delta ? (
                      <div className="flex flex-wrap items-center gap-2">
                        <DeltaIndicator label={delta.label} delta={delta.delta} tone={delta.tone} />
                        <SignificanceBadge stat={significance} />
                      </div>
                    ) : null}
                  </div>
                ) : (
                  <div className="mt-3 space-y-2">
                    {variants.map((variant) => (
                      <div key={variant.label} className="flex items-center justify-between gap-3 text-sm">
                        <span className="truncate text-muted-foreground">{variant.name}</span>
                        <span className="font-semibold tabular-nums text-foreground">{metricDefinition.value(variant)}</span>
                      </div>
                    ))}
                  </div>
                )}
                {showDeltas && !showVersus ? (
                  <div className="mt-1 text-xs text-muted-foreground">{primaryVariant.name}</div>
                ) : null}
                {showVersus ? (
                  <div className="mt-3 flex justify-end">
                    <SignificanceBadge stat={significance} />
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>
      </div>
    </ComparisonSectionShell>
  );
}

type TokenUsageBreakdown = {
  totalTokens: number;
  inputTokens: number;
  outputTokens: number;
  cachedInputTokens: number;
  nonCachedInputTokens: number;
};

function tokenUsageBreakdown(variant: ComparisonCard["variants"][number]): TokenUsageBreakdown | null {
  const instances = variant.instances ?? [];
  if (instances.length === 0) return null;
  const breakdown = instances.reduce<TokenUsageBreakdown>(
    (sum, instance) => {
      const resources = instance.resources ?? {};
      sum.totalTokens += resources.totalTokens ?? 0;
      sum.inputTokens += resources.inputTokens ?? 0;
      sum.outputTokens += resources.outputTokens ?? 0;
      sum.cachedInputTokens += resources.cachedInputTokens ?? 0;
      sum.nonCachedInputTokens += resources.nonCachedInputTokens ?? 0;
      return sum;
    },
    { totalTokens: 0, inputTokens: 0, outputTokens: 0, cachedInputTokens: 0, nonCachedInputTokens: 0 },
  );
  return breakdown.totalTokens > 0 || breakdown.inputTokens > 0 || breakdown.outputTokens > 0 ? breakdown : null;
}

type TokenMetricDelta = {
  delta: number;
  label: string;
  tone: "success" | "danger" | "neutral";
};

function tokenCountDelta(
  baselineValue: number,
  treatmentValue: number,
  displayMode: DeltaDisplayMode,
): TokenMetricDelta {
  const delta = treatmentValue - baselineValue;
  const percentDelta = baselineValue === 0 ? (delta === 0 ? 0 : null) : (delta / Math.abs(baselineValue)) * 100;
  return {
    delta,
    label: displayMode === "percent"
      ? (percentDelta === null ? "n/a" : formatPercentDelta(percentDelta))
      : `${delta > 0 ? "+" : delta < 0 ? "-" : ""}${formatCompactMagnitude(delta)}`,
    tone: deltaTone("lower", delta),
  };
}

function tokenShareDelta(
  baselineValue: number,
  treatmentValue: number,
  displayMode: DeltaDisplayMode,
): TokenMetricDelta {
  const delta = treatmentValue - baselineValue;
  const percentDelta = baselineValue === 0 ? (delta === 0 ? 0 : null) : (delta / Math.abs(baselineValue)) * 100;
  return {
    delta,
    label: displayMode === "percent"
      ? (percentDelta === null ? "n/a" : formatPercentDelta(percentDelta))
      : `${formatSignedFixed(delta * 100, 1)} pts`,
    tone: "neutral",
  };
}

function tokenCachedShare(breakdown: TokenUsageBreakdown): number {
  const inputTotal = Math.max(breakdown.inputTokens, 0);
  return inputTotal > 0 ? breakdown.cachedInputTokens / inputTotal : 0;
}

function tokenWidths(breakdown: TokenUsageBreakdown) {
  const denominator = Math.max(breakdown.totalTokens, breakdown.inputTokens + breakdown.outputTokens, 1);
  return {
    nonCached: (breakdown.nonCachedInputTokens / denominator) * 100,
    cached: (breakdown.cachedInputTokens / denominator) * 100,
    output: (breakdown.outputTokens / denominator) * 100,
  };
}

function TokenUsageSection({
  comparisonPair,
  showDeltas,
  treatmentDeltaDisplay,
  deltaDisplayMode,
  variants,
}: {
  comparisonPair: ReturnType<typeof getComparisonPair>;
  showDeltas: boolean;
  treatmentDeltaDisplay: TreatmentDeltaDisplay;
  deltaDisplayMode: DeltaDisplayMode;
  variants: ComparisonCard["variants"];
}) {
  const rows = variants
    .map((variant) => ({ variant, breakdown: tokenUsageBreakdown(variant) }))
    .filter((row): row is { variant: ComparisonCard["variants"][number]; breakdown: TokenUsageBreakdown } => row.breakdown !== null);

  if (rows.length === 0) return null;

  const baselineRow = comparisonPair ? rows.find((row) => row.variant.label === comparisonPair.baseline.label) : null;
  const treatmentRow = comparisonPair ? rows.find((row) => row.variant.label === comparisonPair.treatment.label) : null;
  const showDiff = Boolean(showDeltas && treatmentDeltaDisplay !== "versus" && baselineRow && treatmentRow);

  return (
    <ComparisonSectionShell
      title="Token Usage"
      headerInline={
        <HelpIcon
          label="Token Usage"
          explanation="Provider-reported cumulative token usage across included runs. Cached input tokens are counted by the provider and can make totals exceed any single model context window."
        />
      }
    >
      <div className="rounded-lg bg-background p-5">
        {showDiff && baselineRow && treatmentRow ? (
          <TokenUsageDiffCard
            baseline={baselineRow}
            treatment={treatmentRow}
            deltaDisplayMode={deltaDisplayMode}
          />
        ) : (
          <div className="grid gap-4 lg:grid-cols-2">
            {rows.map(({ variant, breakdown }) => (
              <TokenUsageVariantCard key={variant.label} variant={variant} breakdown={breakdown} />
            ))}
          </div>
        )}
      </div>
    </ComparisonSectionShell>
  );
}

function TokenUsageDiffCard({
  baseline,
  treatment,
  deltaDisplayMode,
}: {
  baseline: { variant: ComparisonCard["variants"][number]; breakdown: TokenUsageBreakdown };
  treatment: { variant: ComparisonCard["variants"][number]; breakdown: TokenUsageBreakdown };
  deltaDisplayMode: DeltaDisplayMode;
}) {
  const baselineShare = tokenCachedShare(baseline.breakdown);
  const treatmentShare = tokenCachedShare(treatment.breakdown);
  const totalDelta = tokenCountDelta(baseline.breakdown.totalTokens, treatment.breakdown.totalTokens, deltaDisplayMode);
  const metrics = [
    {
      label: "Input",
      baselineValue: formatTokens(baseline.breakdown.inputTokens),
      treatmentValue: formatTokens(treatment.breakdown.inputTokens),
      delta: tokenCountDelta(baseline.breakdown.inputTokens, treatment.breakdown.inputTokens, deltaDisplayMode),
    },
    {
      label: "Output",
      baselineValue: formatTokens(baseline.breakdown.outputTokens),
      treatmentValue: formatTokens(treatment.breakdown.outputTokens),
      delta: tokenCountDelta(baseline.breakdown.outputTokens, treatment.breakdown.outputTokens, deltaDisplayMode),
    },
    {
      label: "Cached Input",
      baselineValue: formatTokens(baseline.breakdown.cachedInputTokens),
      treatmentValue: formatTokens(treatment.breakdown.cachedInputTokens),
      delta: tokenCountDelta(baseline.breakdown.cachedInputTokens, treatment.breakdown.cachedInputTokens, deltaDisplayMode),
    },
    {
      label: "Cached Share",
      baselineValue: formatPercent(baselineShare),
      treatmentValue: formatPercent(treatmentShare),
      delta: tokenShareDelta(baselineShare, treatmentShare, deltaDisplayMode),
    },
  ];

  return (
    <div className="rounded-md border p-4">
      <div className="mb-5 flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="text-sm font-medium">{treatment.variant.name}</div>
          <div className="mt-1 text-xs text-muted-foreground">Compared with {baseline.variant.name}</div>
        </div>
        <div className="text-right">
          <div className="text-xs uppercase tracking-wide text-muted-foreground">Total token delta</div>
          <div className="mt-1 flex justify-end">
            <DeltaIndicator label={totalDelta.label} delta={totalDelta.delta} tone={totalDelta.tone} />
          </div>
        </div>
      </div>

      <div className="space-y-4">
        <TokenUsageStackedBar variant={baseline.variant} breakdown={baseline.breakdown} />
        <TokenUsageStackedBar variant={treatment.variant} breakdown={treatment.breakdown} />
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
          <TokenLegendItem color="hsl(var(--chart-1))" label="Non-cached input" />
          <TokenLegendItem color="hsl(var(--chart-2))" label="Cached input" />
          <TokenLegendItem color="hsl(var(--chart-3))" label="Output" />
        </div>
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {metrics.map((metric) => (
            <TokenDiffMetric key={metric.label} {...metric} />
          ))}
        </div>
      </div>
    </div>
  );
}

function TokenUsageVariantCard({
  variant,
  breakdown,
}: {
  variant: ComparisonCard["variants"][number];
  breakdown: TokenUsageBreakdown;
}) {
  const cachedShare = tokenCachedShare(breakdown);

  return (
    <div className="rounded-md border p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-sm font-medium">{variant.name}</div>
          <div className="mt-1 text-xs text-muted-foreground">Cumulative provider-reported tokens</div>
        </div>
        <div className="text-right">
          <div className="text-xl font-semibold tabular-nums">{formatTokens(breakdown.totalTokens)}</div>
          <div className="text-xs text-muted-foreground">total</div>
        </div>
      </div>

      <TokenUsageStackedBar variant={variant} breakdown={breakdown} className="mt-4" showLabel={false} />

      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <TokenBreakdownMetric label="Input" value={formatTokens(breakdown.inputTokens)} />
        <TokenBreakdownMetric label="Output" value={formatTokens(breakdown.outputTokens)} />
        <TokenBreakdownMetric label="Cached Input" value={formatTokens(breakdown.cachedInputTokens)} />
        <TokenBreakdownMetric label="Cached Share" value={formatPercent(cachedShare)} />
      </div>
      <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
        <TokenLegendItem color="hsl(var(--chart-1))" label="Non-cached input" />
        <TokenLegendItem color="hsl(var(--chart-2))" label="Cached input" />
        <TokenLegendItem color="hsl(var(--chart-3))" label="Output" />
      </div>
    </div>
  );
}

function TokenUsageStackedBar({
  variant,
  breakdown,
  className,
  showLabel = true,
}: {
  variant: ComparisonCard["variants"][number];
  breakdown: TokenUsageBreakdown;
  className?: string;
  showLabel?: boolean;
}) {
  const widths = tokenWidths(breakdown);

  return (
    <div className={className}>
      {showLabel ? (
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{variant.name}</div>
          <div className="text-xs tabular-nums text-muted-foreground">{formatTokens(breakdown.totalTokens)}</div>
        </div>
      ) : null}
      <div className="h-2 overflow-hidden rounded-full bg-muted">
        <div className="flex h-full w-full">
          <div className="bg-[hsl(var(--chart-1))]" style={{ width: `${widths.nonCached}%` }} />
          <div className="bg-[hsl(var(--chart-2))]" style={{ width: `${widths.cached}%` }} />
          <div className="bg-[hsl(var(--chart-3))]" style={{ width: `${widths.output}%` }} />
        </div>
      </div>
    </div>
  );
}

function TokenDiffMetric({
  label,
  baselineValue,
  treatmentValue,
  delta,
}: {
  label: string;
  baselineValue: string;
  treatmentValue: string;
  delta: TokenMetricDelta;
}) {
  return (
    <div className="rounded-md bg-muted/35 p-3">
      <div className="text-xs uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-2 flex flex-wrap items-end justify-between gap-2">
        <div>
          <div className="font-semibold tabular-nums">{treatmentValue}</div>
          <div className="mt-0.5 text-xs text-muted-foreground">baseline {baselineValue}</div>
        </div>
        <DeltaIndicator label={delta.label} delta={delta.delta} tone={delta.tone} />
      </div>
    </div>
  );
}

function TokenBreakdownMetric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-1 font-semibold tabular-nums">{value}</div>
    </div>
  );
}

function TokenLegendItem({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
      {label}
    </span>
  );
}

function ResourceUsagePanel({
  comparison,
  comparisonPair,
  showDeltas,
  treatmentDeltaDisplay,
  deltaDisplayMode,
  distributionMetric,
  metricDefinition,
  variants,
}: {
  comparison: ComparisonCard;
  comparisonPair: ReturnType<typeof getComparisonPair>;
  showDeltas: boolean;
  treatmentDeltaDisplay: TreatmentDeltaDisplay;
  deltaDisplayMode: DeltaDisplayMode;
  distributionMetric: ResourceDistributionMetric;
  metricDefinition: MetricDefinition;
  variants: ComparisonCard["variants"];
}) {
  const valuesByVariant = variants.map((variant) => getResourceValues(variant, distributionMetric));
  const hasChartData = valuesByVariant.some((values) => values.length > 0);
  const showVersus = showDeltas && treatmentDeltaDisplay === "versus" && comparisonPair;
  const delta = showDeltas && !showVersus && comparisonPair ? metricDelta(metricDefinition, comparisonPair.baseline, comparisonPair.treatment, deltaDisplayMode) : null;
  const significance = comparisonPair ? getMetricSignificance(comparison, metricDefinition.key) : null;
  const excludedValueCount = getExcludedResourceValueCount(variants, distributionMetric);

  return (
    <div className="rounded-md border p-4">
      <div className="mb-5 flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="inline-flex items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground">
            <MetricDirectionBadge direction={metricDefinition.direction} />
          </div>
          <div className="mt-1 text-sm text-muted-foreground">Smoothed {distributionMetric.valueLabel.toLowerCase()} density across task runs.</div>
          {excludedValueCount > 0 ? (
            <div className="mt-1 text-xs text-muted-foreground">
              {excludedValueCount.toLocaleString()} timeout-inconsistent duration {excludedValueCount === 1 ? "value is" : "values are"} excluded.
            </div>
          ) : null}
        </div>
        {showVersus ? (
          <div className="w-full sm:w-80">
            <MetricVersusValues
              baselineValue={metricDefinition.value(comparisonPair.baseline)}
              treatmentValue={metricDefinition.value(comparisonPair.treatment)}
              direction={metricDefinition.direction}
              baselineNumericValue={metricDefinition.parse(metricDefinition.value(comparisonPair.baseline))}
              treatmentNumericValue={metricDefinition.parse(metricDefinition.value(comparisonPair.treatment))}
            />
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
      <div className="mb-4 grid gap-3 sm:grid-cols-2">
        {variants.map((variant) => (
          <div key={variant.label} className="rounded-md bg-muted/40 p-3">
            <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{variant.name}</div>
            <div className="mt-2 text-lg font-medium tabular-nums">{metricDefinition.value(variant)}</div>
          </div>
        ))}
      </div>
      {hasChartData ? (
        <ResourceDistributionChart
          metric={distributionMetric}
          variants={variants}
          valuesByVariant={valuesByVariant}
        />
      ) : null}
    </div>
  );
}

function getExcludedResourceValueCount(
  variants: ComparisonCard["variants"],
  metric: ResourceDistributionMetric,
): number {
  if (metric.key !== "durationMs") return 0;
  return variants.reduce((sum, variant) => sum + (variant.results.efficiency.excludedDurationValues ?? 0), 0);
}

function ResourceDistributionChart({
  metric,
  variants,
  valuesByVariant,
}: {
  metric: ResourceDistributionMetric;
  variants: ComparisonCard["variants"];
  valuesByVariant: number[][];
}) {
  const chartConfig = buildResourceChartConfig(variants);
  const { rows: chartData, domain } = buildResourceDensityRows({ metric, valuesByVariant });

  return (
    <ChartContainer config={chartConfig} className="h-[320px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={chartData} margin={{ top: 16, right: 16, bottom: 16, left: 0 }}>
          <CartesianGrid vertical={false} />
          <XAxis
            dataKey="value"
            type="number"
            domain={domain}
            ticks={getDensityTicks(domain)}
            tickFormatter={(value) => metric.formatValue(Number(value))}
            tickLine={false}
            tickMargin={10}
            axisLine={false}
          />
          <YAxis
            tickLine={false}
            axisLine={false}
            tickMargin={10}
            width={44}
            domain={[0, 1]}
            ticks={[0, 0.5, 1]}
            tickFormatter={(value) => Number(value).toFixed(1)}
          />
          <ChartTooltip
            cursor={false}
            content={
              <ChartTooltipContent
                labelFormatter={(value) => `${metric.valueLabel}: ${metric.formatValue(Number(value))}`}
                formatter={(value) => Number(value).toFixed(3)}
              />
            }
          />
          <ChartLegend content={<ChartLegendContent />} />
          {variants.map((_variant, index) => (
            <Area
              key={resourceVariantKey(index)}
              type="monotone"
              dataKey={resourceVariantKey(index)}
              stroke={`var(--color-${resourceVariantKey(index)})`}
              fill={`var(--color-${resourceVariantKey(index)})`}
              fillOpacity={0.16}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
    </ChartContainer>
  );
}
