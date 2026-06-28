// SPDX-License-Identifier: Apache-2.0

import { useMemo, useState } from "react";
import {
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  XAxis,
  YAxis,
} from "recharts";

import type { ComparisonCard, ComparisonInstance } from "@/data/comparisons";
import {
  CORRELATION_METRIC_GROUPS,
  CORRELATION_METRICS,
  DEFAULT_X_METRIC_ID,
  DEFAULT_Y_METRIC_ID,
  getCorrelationMetric,
  type CorrelationMetric,
} from "@/data/correlation-metrics";
import { formatResolutionStatus } from "@/components/comparison/format";
import { HelpIcon } from "@/components/comparison/shared";
import { ChartContainer, ChartTooltip } from "@/components/ui/chart";

const VARIANT_COLORS = [
  "hsl(var(--chart-1))",
  "hsl(var(--chart-2))",
  "hsl(var(--chart-3))",
  "hsl(var(--chart-4))",
  "hsl(var(--chart-5))",
];

type ResolutionFilter = "all" | "resolved" | "unresolved";

const RESOLUTION_FILTER_OPTIONS: Array<{ value: ResolutionFilter; label: string }> = [
  { value: "all", label: "All tasks" },
  { value: "resolved", label: "Resolved only" },
  { value: "unresolved", label: "Unresolved only" },
];

type CorrelationPoint = {
  x: number;
  y: number;
  instanceId: string;
  bench: string;
  language: string;
  variantName: string;
  resolutionStatus?: string;
};

type VariantSeries = {
  key: string;
  name: string;
  color: string;
  points: CorrelationPoint[];
  excluded: number;
  pearsonR: number | null;
  trend: { x1: number; y1: number; x2: number; y2: number } | null;
};

type AxisState = {
  metricId: string;
  log: boolean;
};

function defaultAxisState(metricId: string): AxisState {
  return { metricId, log: getCorrelationMetric(metricId)?.preferLog ?? false };
}

export function CorrelationSection({ comparison }: { comparison: ComparisonCard }) {
  const [xAxis, setXAxis] = useState<AxisState>(() => defaultAxisState(DEFAULT_X_METRIC_ID));
  const [yAxis, setYAxis] = useState<AxisState>(() => defaultAxisState(DEFAULT_Y_METRIC_ID));
  const [resolutionFilter, setResolutionFilter] = useState<ResolutionFilter>("all");

  const xMetric = getCorrelationMetric(xAxis.metricId) ?? CORRELATION_METRICS[0];
  const yMetric = getCorrelationMetric(yAxis.metricId) ?? CORRELATION_METRICS[0];
  const showTrend = !xAxis.log && !yAxis.log;

  const availableMetricIds = useMemo(() => collectAvailableMetricIds(comparison), [comparison]);
  const series = useMemo(
    () => buildVariantSeries(comparison, xMetric, yMetric, xAxis.log, yAxis.log, showTrend, resolutionFilter),
    [comparison, xMetric, yMetric, xAxis.log, yAxis.log, showTrend, resolutionFilter],
  );

  const plottedCount = series.reduce((sum, variant) => sum + variant.points.length, 0);
  const excludedCount = series.reduce((sum, variant) => sum + variant.excluded, 0);

  return (
    <section className="space-y-4 rounded-lg bg-background p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-xl font-semibold tracking-tight">Correlations</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Each dot is a task instance. Pick a metric per axis to explore how per-task metrics relate.
          </p>
        </div>
        <HelpIcon
          label="Correlations"
          explanation="Scatter plot of per-task-instance metrics. Instances missing a value for either selected metric are excluded; the Pearson r per variant is computed on the values as displayed (log-transformed when a log axis is enabled). The dashed trend line is an ordinary least-squares fit and is shown only when both axes are linear. The resolution filter restricts dots to tasks the official evaluation marked resolved or unresolved."
        />
      </div>
      <div className="flex flex-wrap gap-x-6 gap-y-3">
        <AxisControls axisLabel="X axis" state={xAxis} onChange={setXAxis} availableMetricIds={availableMetricIds} />
        <AxisControls axisLabel="Y axis" state={yAxis} onChange={setYAxis} availableMetricIds={availableMetricIds} />
        <label className="flex flex-col gap-1 text-sm">
          <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Resolution</span>
          <select
            className="h-9 min-w-40 rounded-md border bg-background px-2 text-sm"
            value={resolutionFilter}
            onChange={(event) => setResolutionFilter(event.target.value as ResolutionFilter)}
          >
            {RESOLUTION_FILTER_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        {series.map((variant) => (
          <div key={variant.key} className="flex items-center gap-2 rounded-md border px-3 py-1.5 text-sm">
            <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: variant.color }} />
            <span className="font-medium">{variant.name}</span>
            <span className="text-muted-foreground tabular-nums">
              n = {variant.points.length}
              {variant.pearsonR !== null ? ` · r = ${variant.pearsonR.toFixed(2)}` : ""}
            </span>
          </div>
        ))}
      </div>
      {plottedCount === 0 ? (
        <p className="rounded-md border p-4 text-sm text-muted-foreground">
          No instances match the current resolution filter and have values for both selected metrics.
        </p>
      ) : (
        <CorrelationChart series={series} xMetric={xMetric} yMetric={yMetric} xLog={xAxis.log} yLog={yAxis.log} comparisonId={comparison.id} showTrend={showTrend} />
      )}
      {excludedCount > 0 ? (
        <p className="text-xs text-muted-foreground">
          {excludedCount} of {plottedCount + excludedCount} runs matching the resolution filter are excluded because at
          least one selected metric is unavailable for them
          {xAxis.log || yAxis.log ? " or is non-positive on a log-scaled axis" : ""}.
        </p>
      ) : null}
    </section>
  );
}

function AxisControls({
  axisLabel,
  state,
  onChange,
  availableMetricIds,
}: {
  axisLabel: string;
  state: AxisState;
  onChange: (state: AxisState) => void;
  availableMetricIds: Set<string>;
}) {
  const metric = getCorrelationMetric(state.metricId);
  return (
    <div className="flex flex-wrap items-end gap-3">
      <label className="flex flex-col gap-1 text-sm">
        <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{axisLabel}</span>
        <select
          className="h-9 min-w-56 rounded-md border bg-background px-2 text-sm"
          value={state.metricId}
          onChange={(event) => onChange(defaultAxisState(event.target.value))}
        >
          {CORRELATION_METRIC_GROUPS.map((group) => (
            <optgroup key={group} label={group}>
              {CORRELATION_METRICS.filter((candidate) => candidate.group === group).map((candidate) => {
                const available = availableMetricIds.has(candidate.id);
                return (
                  <option key={candidate.id} value={candidate.id} disabled={!available}>
                    {available ? candidate.label : `${candidate.label} (no data)`}
                  </option>
                );
              })}
            </optgroup>
          ))}
        </select>
      </label>
      <label className="flex h-9 items-center gap-2 text-sm text-muted-foreground">
        <input
          type="checkbox"
          checked={state.log}
          onChange={(event) => onChange({ ...state, log: event.target.checked })}
        />
        Log scale
      </label>
      {metric ? <HelpIcon label={metric.label} explanation={metric.explanation} /> : null}
    </div>
  );
}

function CorrelationChart({
  series,
  xMetric,
  yMetric,
  xLog,
  yLog,
  comparisonId,
  showTrend,
}: {
  series: VariantSeries[];
  xMetric: CorrelationMetric;
  yMetric: CorrelationMetric;
  xLog: boolean;
  yLog: boolean;
  comparisonId: string;
  showTrend: boolean;
}) {
  const chartConfig = Object.fromEntries(
    series.map((variant) => [variant.key, { label: variant.name, color: variant.color }]),
  );

  return (
    <ChartContainer config={chartConfig} className="h-[420px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 12, right: 24, bottom: 8, left: 12 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            type="number"
            dataKey="x"
            name={xMetric.label}
            scale={xLog ? "log" : "linear"}
            domain={xLog ? ["auto", "auto"] : [0, "auto"]}
            tickFormatter={xMetric.format}
            tick={{ fontSize: 12 }}
            label={{ value: xMetric.label, position: "insideBottom", offset: -2, fontSize: 12 }}
            height={42}
          />
          <YAxis
            type="number"
            dataKey="y"
            name={yMetric.label}
            scale={yLog ? "log" : "linear"}
            domain={yLog ? ["auto", "auto"] : [0, "auto"]}
            tickFormatter={yMetric.format}
            tick={{ fontSize: 12 }}
            label={{ value: yMetric.label, angle: -90, position: "insideLeft", offset: 0, fontSize: 12 }}
            width={72}
          />
          <ChartTooltip
            cursor={{ strokeDasharray: "3 3" }}
            content={<CorrelationTooltip xMetric={xMetric} yMetric={yMetric} />}
          />
          {showTrend
            ? series.flatMap((variant) =>
                variant.trend
                  ? [
                      <ReferenceLine
                        key={`${variant.key}-trend`}
                        segment={[
                          { x: variant.trend.x1, y: variant.trend.y1 },
                          { x: variant.trend.x2, y: variant.trend.y2 },
                        ]}
                        stroke={variant.color}
                        strokeDasharray="6 4"
                        ifOverflow="hidden"
                      />,
                    ]
                  : [],
              )
            : null}
          {series.map((variant) => (
            <Scatter
              key={variant.key}
              name={variant.name}
              data={variant.points}
              fill={variant.color}
              fillOpacity={0.7}
              isAnimationActive={false}
              cursor="pointer"
              onClick={(point: CorrelationPoint) => {
                if (point?.instanceId) {
                  window.location.hash = `#/comparisons/${comparisonId}/instances/${encodeURIComponent(point.instanceId)}`;
                }
              }}
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </ChartContainer>
  );
}

function CorrelationTooltip({
  active,
  payload,
  xMetric,
  yMetric,
}: {
  active?: boolean;
  payload?: Array<{ payload?: CorrelationPoint }>;
  xMetric: CorrelationMetric;
  yMetric: CorrelationMetric;
}) {
  const point = payload?.[0]?.payload;
  if (!active || !point) return null;
  return (
    <div className="rounded-md border bg-background p-3 text-xs shadow-md">
      <div className="font-medium">{point.instanceId}</div>
      <div className="mt-1 text-muted-foreground">
        {point.variantName} · {point.bench} · {point.language}
        {point.resolutionStatus ? ` · ${formatResolutionStatus(point.resolutionStatus)}` : ""}
      </div>
      <div className="mt-2 space-y-1 tabular-nums">
        <div>
          {xMetric.label}: <span className="font-medium">{xMetric.format(point.x)}</span>
        </div>
        <div>
          {yMetric.label}: <span className="font-medium">{yMetric.format(point.y)}</span>
        </div>
      </div>
    </div>
  );
}

function collectAvailableMetricIds(comparison: ComparisonCard): Set<string> {
  const instances = comparison.variants.flatMap((variant) => variant.instances ?? []);
  return new Set(
    CORRELATION_METRICS
      .filter((metric) => instances.some((instance) => metric.extract(instance) !== null))
      .map((metric) => metric.id),
  );
}

function matchesResolutionFilter(instance: ComparisonInstance, filter: ResolutionFilter): boolean {
  if (filter === "all") return true;
  return instance.artifacts?.resolutionStatus === filter;
}

function buildVariantSeries(
  comparison: ComparisonCard,
  xMetric: CorrelationMetric,
  yMetric: CorrelationMetric,
  xLog: boolean,
  yLog: boolean,
  showTrend: boolean,
  resolutionFilter: ResolutionFilter,
): VariantSeries[] {
  return comparison.variants.map((variant, index) => {
    const instances = (variant.instances ?? []).filter((instance) => matchesResolutionFilter(instance, resolutionFilter));
    const points = instances.flatMap((instance) => {
      const point = buildPoint(instance, variant.name, xMetric, yMetric, xLog, yLog);
      return point ? [point] : [];
    });
    return {
      key: `variant${index}`,
      name: variant.name,
      color: VARIANT_COLORS[index % VARIANT_COLORS.length],
      points,
      excluded: instances.length - points.length,
      // Correlation is computed on the coordinates as displayed so the
      // reported r matches the visual trend when a log axis is enabled.
      pearsonR: pearsonCorrelation(displayedCoordinates(points, xLog, yLog)),
      trend: showTrend ? leastSquaresSegment(points) : null,
    };
  });
}

function buildPoint(
  instance: ComparisonInstance,
  variantName: string,
  xMetric: CorrelationMetric,
  yMetric: CorrelationMetric,
  xLog: boolean,
  yLog: boolean,
): CorrelationPoint | null {
  const x = xMetric.extract(instance);
  const y = yMetric.extract(instance);
  if (x === null || y === null) return null;
  // Log axes cannot place non-positive values.
  if ((xLog && x <= 0) || (yLog && y <= 0)) return null;
  return {
    x,
    y,
    instanceId: instance.instanceId,
    bench: instance.bench,
    language: instance.language,
    variantName,
    resolutionStatus: instance.artifacts?.resolutionStatus,
  };
}

type XYPoint = { x: number; y: number };

function displayedCoordinates(points: CorrelationPoint[], xLog: boolean, yLog: boolean): XYPoint[] {
  if (!xLog && !yLog) return points;
  return points.map((point) => ({
    x: xLog ? Math.log10(point.x) : point.x,
    y: yLog ? Math.log10(point.y) : point.y,
  }));
}

type Moments = {
  meanX: number;
  meanY: number;
  covariance: number;
  varianceX: number;
  varianceY: number;
};

function computeMoments(points: XYPoint[]): Moments | null {
  if (points.length < 3) return null;
  const n = points.length;
  const meanX = points.reduce((sum, point) => sum + point.x, 0) / n;
  const meanY = points.reduce((sum, point) => sum + point.y, 0) / n;
  let covariance = 0;
  let varianceX = 0;
  let varianceY = 0;
  for (const point of points) {
    const dx = point.x - meanX;
    const dy = point.y - meanY;
    covariance += dx * dy;
    varianceX += dx * dx;
    varianceY += dy * dy;
  }
  return { meanX, meanY, covariance, varianceX, varianceY };
}

function pearsonCorrelation(points: XYPoint[]): number | null {
  const moments = computeMoments(points);
  if (!moments || moments.varianceX === 0 || moments.varianceY === 0) return null;
  return moments.covariance / Math.sqrt(moments.varianceX * moments.varianceY);
}

function leastSquaresSegment(points: CorrelationPoint[]): VariantSeries["trend"] {
  const moments = computeMoments(points);
  if (!moments || moments.varianceX === 0) return null;
  const slope = moments.covariance / moments.varianceX;
  const intercept = moments.meanY - slope * moments.meanX;
  const minX = Math.min(...points.map((point) => point.x));
  const maxX = Math.max(...points.map((point) => point.x));
  return { x1: minX, y1: intercept + slope * minX, x2: maxX, y2: intercept + slope * maxX };
}
