// SPDX-License-Identifier: Apache-2.0

import { Area, AreaChart, CartesianGrid, ResponsiveContainer, XAxis, YAxis } from "recharts";

import type { ComparisonCard } from "@/data/comparisons";
import { formatDurationMs, formatTokens, getComparisonPair } from "@/components/comparison/format";
import { metricDelta, resourceMetricDefinitions } from "@/components/comparison/metrics";
import { MetricVersusValues, type MetricSectionProps, type TreatmentDeltaDisplay } from "@/components/comparison/metric-display";
import { ComparisonSectionShell, DeltaIndicator, HelpIcon, MetricDirectionBadge } from "@/components/comparison/shared";
import type { DeltaDisplayMode, MetricDefinition } from "@/components/comparison/types";
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart";

type ResourceMetricKey = "durationMs" | "totalTokens" | "steps";

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
    valueLabel: "Steps",
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

export function ResourceUsageMetricSection(props: MetricSectionProps) {
  return <CombinedResourceUsageSection {...props} />;
}

function resourceVariantKey(index: number): string {
  return `variant${index}`;
}

function formatSteps(value: number): string {
  return `${Number.isInteger(value) ? value.toFixed(0) : value.toFixed(1)} steps`;
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
}: MetricSectionProps) {
  const comparisonPair = getComparisonPair(comparison);
  const showDeltas = viewMode === "treatment-delta" && comparisonPair;
  const showVersus = showDeltas && treatmentDeltaDisplay === "versus";
  const variants = showDeltas && !showVersus ? [comparisonPair.treatment] : comparison.variants;
  const panels = resourceDistributionMetrics
    .map((distributionMetric) => ({
      distributionMetric,
      metricDefinition: resourceMetricDefinitions.find((metric) => metric.key === distributionMetric.metricKey),
    }))
    .filter((panel): panel is { distributionMetric: ResourceDistributionMetric; metricDefinition: MetricDefinition } =>
      Boolean(panel.metricDefinition),
    )
    .filter(({ distributionMetric, metricDefinition }) =>
      variants.some((variant) => metricDefinition.value(variant) !== "—" || getResourceValues(variant, distributionMetric).length > 0),
    );

  if (panels.length === 0) return null;

  return (
    <>
      <ResourceUsageSummary
        comparisonPair={comparisonPair}
        showDeltas={Boolean(showDeltas)}
        treatmentDeltaDisplay={treatmentDeltaDisplay}
        deltaDisplayMode={deltaDisplayMode}
        panels={panels}
        variants={variants}
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
              comparisonPair={comparisonPair}
              showDeltas={Boolean(showDeltas)}
              treatmentDeltaDisplay={treatmentDeltaDisplay}
              deltaDisplayMode={deltaDisplayMode}
              distributionMetric={distributionMetric}
              metricDefinition={metricDefinition}
              variants={variants}
            />
          </div>
        </ComparisonSectionShell>
      ))}
    </>
  );
}

function ResourceUsageSummary({
  comparisonPair,
  showDeltas,
  treatmentDeltaDisplay,
  deltaDisplayMode,
  panels,
  variants,
}: {
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
                    {delta ? <DeltaIndicator label={delta.label} delta={delta.delta} tone={delta.tone} /> : null}
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
              </div>
            );
          })}
        </div>
      </div>
    </ComparisonSectionShell>
  );
}

function ResourceUsagePanel({
  comparisonPair,
  showDeltas,
  treatmentDeltaDisplay,
  deltaDisplayMode,
  distributionMetric,
  metricDefinition,
  variants,
}: {
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

  return (
    <div className="rounded-md border p-4">
      <div className="mb-5 flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="inline-flex items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground">
            <MetricDirectionBadge direction={metricDefinition.direction} />
          </div>
          <div className="mt-1 text-sm text-muted-foreground">Smoothed {distributionMetric.valueLabel.toLowerCase()} density across task runs.</div>
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
            <DeltaIndicator label={delta.label} delta={delta.delta} tone={delta.tone} />
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
