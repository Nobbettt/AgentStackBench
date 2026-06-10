// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
import { Bar, BarChart, CartesianGrid, ReferenceDot, ResponsiveContainer, XAxis, YAxis } from "recharts";

import type { ComparisonCard, ComparisonInstance } from "@/data/comparisons";
import {
  formatLanguageLabel,
  getComparisonPair,
  sortBench,
} from "@/components/comparison/format";
import { coveragePrecisionOrNull, f1, hasValidEvaluation, instanceContextAggregate } from "@/data/instance-metrics";
import {
  formatPValue,
  formatSignificanceEffect,
  formatSignificanceInterval,
  getGroupedMetricSignificance,
  type PairedSignificance,
} from "@/components/comparison/significance";
import { ComparisonSectionShell, HelpIcon } from "@/components/comparison/shared";
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart";

export function LanguageMetricsSection({
  comparison,
}: {
  comparison: ComparisonCard;
}) {
  const [activeSlideKey, setActiveSlideKey] = useState<LanguageMetricSlideKey>("resolution");
  const comparisonPair = getComparisonPair(comparison);
  if (!comparisonPair) {
    return (
      <ComparisonSectionShell title="Metric Breakdowns">
        <div className="rounded-lg bg-background p-5 text-sm text-muted-foreground">
          Metric breakdowns require a baseline and treatment variant.
        </div>
      </ComparisonSectionShell>
    );
  }

  const chartConfig = {
    baseline: {
      label: comparisonPair.baseline.name,
      color: "hsl(var(--chart-1))",
    },
    treatment: {
      label: comparisonPair.treatment.name,
      color: "hsl(var(--chart-2))",
    },
  } satisfies ChartConfig;
  const resolutionLanguageData = getResolutionRows(comparisonPair.baseline, comparisonPair.treatment, "language");
  const resolutionBenchData = getResolutionRows(comparisonPair.baseline, comparisonPair.treatment, "bench");
  const resolutionRepoSizeData = getResolutionRows(comparisonPair.baseline, comparisonPair.treatment, "repositorySize");
  const contextLanguageData = getContextRows(comparisonPair.baseline, comparisonPair.treatment, "language");
  const contextBenchData = getContextRows(comparisonPair.baseline, comparisonPair.treatment, "bench");
  const contextRepoSizeData = getContextRows(comparisonPair.baseline, comparisonPair.treatment, "repositorySize");
  const skillUsageData = getUsageRows(comparisonPair.baseline, comparisonPair.treatment, "skills", "language");
  const skillUsageBenchData = getUsageRows(comparisonPair.baseline, comparisonPair.treatment, "skills", "bench");
  const skillUsageRepoSizeData = getUsageRows(comparisonPair.baseline, comparisonPair.treatment, "skills", "repositorySize");
  const skillUsageAxis = getUsageAxis([...skillUsageData, ...skillUsageBenchData, ...skillUsageRepoSizeData]);
  const toolUsageData = getUsageRows(comparisonPair.baseline, comparisonPair.treatment, "tools", "language");
  const toolUsageBenchData = getUsageRows(comparisonPair.baseline, comparisonPair.treatment, "tools", "bench");
  const toolUsageRepoSizeData = getUsageRows(comparisonPair.baseline, comparisonPair.treatment, "tools", "repositorySize");
  const toolUsageAxis = getUsageAxis([...toolUsageData, ...toolUsageBenchData, ...toolUsageRepoSizeData]);
  const availableSlides: LanguageMetricSlide[] = [
    {
      key: "resolution",
      label: "Resolution",
      languageTitle: "Resolution Rate by Language",
      languageDescription: "Resolved tasks divided by resolved and unresolved tasks for each programming language.",
      benchmarkTitle: "Resolution Rate by Benchmark",
      benchmarkDescription: "Resolved tasks divided by resolved and unresolved tasks for each benchmark slice.",
      repositorySizeTitle: "Resolution Rate by Repository Size",
      repositorySizeDescription: "Resolved tasks divided by resolved and unresolved tasks for each git-tracked file count bucket. Tasks without local repository-size metadata are omitted.",
      data: resolutionLanguageData,
      benchmarkData: resolutionBenchData,
      repositorySizeData: resolutionRepoSizeData,
      domain: [0, 100],
      yTicks: [0, 25, 50, 75, 100],
      yTickFormatter: (value) => `${value}%`,
      tooltipFormatter: (value, key, item) => {
        const resolved = item.payload?.[`${key}Resolved`];
        const total = item.payload?.[`${key}Total`];
        const countLabel = typeof resolved === "number" && typeof total === "number"
          ? ` (${resolved}/${total})`
          : "";
        return `${Number(value).toFixed(1)}%${countLabel}`;
      },
    },
    {
      key: "context",
      label: "Macro Context F1",
      languageTitle: "Macro Context F1 by Language",
      languageDescription: "ContextBench F1 by language, computed from file, block, line, and symbol retrieval F1.",
      benchmarkTitle: "Macro Context F1 by Benchmark",
      benchmarkDescription: "ContextBench F1 by benchmark slice, computed from file, block, line, and symbol retrieval F1.",
      repositorySizeTitle: "Macro Context F1 by Repository Size",
      repositorySizeDescription: "ContextBench F1 by git-tracked file count bucket, computed from file, block, line, and symbol retrieval F1. Tasks without local repository-size metadata are omitted.",
      data: contextLanguageData,
      benchmarkData: contextBenchData,
      repositorySizeData: contextRepoSizeData,
      domain: [0, 1],
      yTicks: [0, 0.25, 0.5, 0.75, 1],
      yTickFormatter: (value) => Number(value).toFixed(2),
      tooltipFormatter: (value, key, item) => {
        const taskCount = item.payload?.[`${key}Total`];
        const countLabel = typeof taskCount === "number" ? ` (${taskCount} valid evals)` : "";
        return `${Number(value).toFixed(3)}${countLabel}`;
      },
    },
    ...(skillUsageData.length > 0 || skillUsageBenchData.length > 0 || skillUsageRepoSizeData.length > 0 ? [{
      key: "skills",
      label: "Skills",
      languageTitle: "Skill Usage by Language",
      languageDescription: "Average skill invocations per run for each programming language.",
      benchmarkTitle: "Skill Usage by Benchmark",
      benchmarkDescription: "Average skill invocations per run for each benchmark slice.",
      repositorySizeTitle: "Skill Usage by Repository Size",
      repositorySizeDescription: "Average skill invocations per run for each git-tracked file count bucket. Tasks without local repository-size metadata are omitted.",
      data: skillUsageData,
      benchmarkData: skillUsageBenchData,
      repositorySizeData: skillUsageRepoSizeData,
      domain: skillUsageAxis.domain,
      yTicks: skillUsageAxis.ticks,
      yTickFormatter: (value) => Number(value).toFixed(2),
      tooltipFormatter: (value, key, item) => {
        const totalInvocations = item.payload?.[`${key}Invocations`];
        const totalRuns = item.payload?.[`${key}Total`];
        const countLabel = typeof totalInvocations === "number" && typeof totalRuns === "number"
          ? ` (${totalInvocations} across ${totalRuns} runs)`
          : "";
        return `${Number(value).toFixed(2)} / run${countLabel}`;
      },
    } satisfies LanguageMetricSlide] : []),
    ...(toolUsageData.length > 0 || toolUsageBenchData.length > 0 || toolUsageRepoSizeData.length > 0 ? [{
      key: "tools",
      label: "Tools",
      languageTitle: "Tool Usage by Language",
      languageDescription: "Average tool and MCP invocations per run for each programming language.",
      benchmarkTitle: "Tool Usage by Benchmark",
      benchmarkDescription: "Average tool and MCP invocations per run for each benchmark slice.",
      repositorySizeTitle: "Tool Usage by Repository Size",
      repositorySizeDescription: "Average tool and MCP invocations per run for each git-tracked file count bucket. Tasks without local repository-size metadata are omitted.",
      data: toolUsageData,
      benchmarkData: toolUsageBenchData,
      repositorySizeData: toolUsageRepoSizeData,
      domain: toolUsageAxis.domain,
      yTicks: toolUsageAxis.ticks,
      yTickFormatter: (value) => Number(value).toFixed(2),
      tooltipFormatter: (value, key, item) => {
        const totalInvocations = item.payload?.[`${key}Invocations`];
        const totalRuns = item.payload?.[`${key}Total`];
        const countLabel = typeof totalInvocations === "number" && typeof totalRuns === "number"
          ? ` (${totalInvocations} across ${totalRuns} runs)`
          : "";
        return `${Number(value).toFixed(2)} / run${countLabel}`;
      },
    } satisfies LanguageMetricSlide] : []),
  ];
  const activeSlide = availableSlides.find((slide) => slide.key === activeSlideKey) ?? availableSlides[0];

  if (availableSlides.every((slide) => slide.data.length === 0 && slide.benchmarkData.length === 0 && slide.repositorySizeData.length === 0)) {
    return (
      <ComparisonSectionShell title="Metric Breakdowns">
        <div className="rounded-lg bg-background p-5 text-sm text-muted-foreground">
          No language- or benchmark-level metric data is available.
        </div>
      </ComparisonSectionShell>
    );
  }

  return (
    <ComparisonSectionShell title="Metric Breakdowns">
      <div className="rounded-lg bg-background p-5">
        <div className="mb-4 flex flex-wrap items-start justify-end gap-4">
          <div className="inline-flex rounded-md border bg-background p-1" role="tablist" aria-label="Metric breakdown slides">
            {availableSlides.map((slide) => (
              <button
                key={slide.key}
                type="button"
                role="tab"
                aria-selected={slide.key === activeSlide.key}
                className={`rounded px-3 py-1.5 text-sm font-medium transition-colors ${
                  slide.key === activeSlide.key
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:text-foreground"
                }`}
                onClick={() => setActiveSlideKey(slide.key)}
              >
                {slide.label}
              </button>
            ))}
          </div>
        </div>
        <div className="space-y-8">
          {activeSlide.data.length > 0 ? (
            <div>
              <div className="mb-3 flex items-center gap-2">
                <h3 className="text-lg font-semibold tracking-tight">{activeSlide.languageTitle}</h3>
                <HelpIcon label={activeSlide.languageTitle} explanation={activeSlide.languageDescription} />
              </div>
              <LanguageMetricChart
                comparison={comparison}
                groupKey="language"
                chartConfig={chartConfig}
                slide={activeSlide}
                data={activeSlide.data}
              />
            </div>
          ) : null}
          <div>
            <div className="mb-3 flex items-center gap-2">
              <h3 className="text-lg font-semibold tracking-tight">{activeSlide.benchmarkTitle}</h3>
              <HelpIcon label={activeSlide.benchmarkTitle} explanation={activeSlide.benchmarkDescription} />
            </div>
            {activeSlide.benchmarkData.length > 0 ? (
              <LanguageMetricChart
                comparison={comparison}
                groupKey="bench"
                chartConfig={chartConfig}
                slide={activeSlide}
                data={activeSlide.benchmarkData}
              />
            ) : (
              <div className="rounded-lg bg-muted/40 p-6 text-sm text-muted-foreground">
                No benchmark-level {activeSlide.label.toLowerCase()} data is available for the current filters.
              </div>
            )}
          </div>
          {activeSlide.repositorySizeData.length > 0 ? (
            <div>
              <div className="mb-3 flex items-center gap-2">
                <h3 className="text-lg font-semibold tracking-tight">{activeSlide.repositorySizeTitle}</h3>
                <HelpIcon label={activeSlide.repositorySizeTitle} explanation={activeSlide.repositorySizeDescription} />
              </div>
              <LanguageMetricChart
                comparison={comparison}
                groupKey="repositorySize"
                chartConfig={chartConfig}
                slide={activeSlide}
                data={activeSlide.repositorySizeData}
              />
            </div>
          ) : null}
        </div>
      </div>
    </ComparisonSectionShell>
  );
}

type LanguageMetricSlideKey = "resolution" | "context" | "skills" | "tools";

type LanguageMetricSlide = {
  key: LanguageMetricSlideKey;
  label: string;
  languageTitle: string;
  languageDescription: string;
  benchmarkTitle: string;
  benchmarkDescription: string;
  repositorySizeTitle: string;
  repositorySizeDescription: string;
  data: LanguageMetricRow[];
  benchmarkData: LanguageMetricRow[];
  repositorySizeData: LanguageMetricRow[];
  domain: [number, number];
  yTicks: number[];
  yTickFormatter: (value: number) => string;
  tooltipFormatter: (value: unknown, key: "baseline" | "treatment", item: any) => string;
};

type LanguageMetricRow = {
  language: string;
  sortKey?: string;
  baseline: number;
  treatment: number;
  baselineTotal: number;
  treatmentTotal: number;
  baselineResolved?: number;
  treatmentResolved?: number;
  baselineInvocations?: number;
  treatmentInvocations?: number;
};

function LanguageMetricChart({
  comparison,
  groupKey,
  chartConfig,
  slide,
  data,
}: {
  comparison: ComparisonCard;
  groupKey: MetricGroupKey;
  chartConfig: ChartConfig;
  slide: LanguageMetricSlide;
  data: LanguageMetricRow[];
}) {
  const chartData = buildLanguageMetricChartRows(comparison, slide, data, groupKey);

  return (
    <ChartContainer config={chartConfig} className="h-[360px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} margin={{ top: 38, right: 16, bottom: 24, left: 0 }} barGap={6}>
          <CartesianGrid vertical={false} />
          <XAxis
            dataKey="language"
            tickLine={false}
            tickMargin={10}
            axisLine={false}
            interval={0}
          />
          <YAxis
            tickLine={false}
            axisLine={false}
            tickMargin={10}
            width={44}
            domain={slide.domain}
            ticks={slide.yTicks}
            tickFormatter={slide.yTickFormatter}
          />
          <ChartTooltip
            cursor={false}
            content={
              <ChartTooltipContent
                formatter={(value, _name, item) => {
                  const key = item.dataKey === "baseline" ? "baseline" : "treatment";
                  const formattedValue = slide.tooltipFormatter(value, key, item);
                  const significance = key === "treatment" ? item.payload?.significance : null;
                  return significance ? `${formattedValue}; ${languageMetricSignificanceTooltip(significance)}` : formattedValue;
                }}
              />
            }
          />
          <ChartLegend content={<ChartLegendContent />} />
          <Bar dataKey="baseline" fill="var(--color-baseline)" radius={[4, 4, 0, 0]} />
          <Bar dataKey="treatment" fill="var(--color-treatment)" radius={[4, 4, 0, 0]} />
          {chartData.map((row) => row.significance ? (
            <ReferenceDot
              key={row.sortKey ?? row.language}
              x={row.language}
              y={Math.max(row.baseline, row.treatment)}
              r={0}
              isFront
              ifOverflow="visible"
              shape={(props) => (
                <LanguageMetricSignificanceReferenceBadge
                  {...props}
                  significance={row.significance}
                />
              )}
            />
          ) : null)}
        </BarChart>
      </ResponsiveContainer>
    </ChartContainer>
  );
}

type LanguageMetricChartRow = LanguageMetricRow & {
  significance?: PairedSignificance | null;
};

function buildLanguageMetricChartRows(
  comparison: ComparisonCard,
  slide: LanguageMetricSlide,
  rows: LanguageMetricRow[],
  groupKey: MetricGroupKey,
): LanguageMetricChartRow[] {
  const metricKey = languageMetricSignificanceKey(slide.key);
  return rows.map((row) => ({
    ...row,
    significance: getGroupedMetricSignificance(comparison, metricKey, groupKey, row.sortKey),
  }));
}

function LanguageMetricSignificanceReferenceBadge({
  cx,
  cy,
  significance,
}: {
  cx?: number | string;
  cy?: number | string;
  significance: PairedSignificance;
}) {
  const numericX = Number(cx);
  const numericY = Number(cy);
  if (!Number.isFinite(numericX) || !Number.isFinite(numericY)) return null;

  const badgeWidth = significance.significant ? 46 : 50;
  const badgeHeight = 18;
  const centerX = numericX;
  const centerY = numericY - 15;
  const startX = centerX - badgeWidth / 2;
  const startY = centerY - badgeHeight / 2;
  const iconX = startX + 12;
  const textX = startX + 31;
  const iconStroke = significance.significant ? "rgb(4 120 87)" : "hsl(var(--muted-foreground))";
  const badgeFill = significance.significant ? "rgb(236 253 245)" : "hsl(var(--muted))";
  const badgeStroke = significance.significant ? "rgb(167 243 208)" : "hsl(var(--border))";
  const textFill = significance.significant ? "rgb(4 120 87)" : "hsl(var(--muted-foreground))";
  const opacity = significance.significant ? 1 : 0.66;

  return (
    <g opacity={opacity}>
      <rect
        x={startX}
        y={startY}
        width={badgeWidth}
        height={badgeHeight}
        rx={9}
        fill={badgeFill}
        stroke={badgeStroke}
      />
      <circle cx={iconX} cy={centerY} r={4.6} fill="none" stroke={iconStroke} strokeWidth={1.8} />
      <path
        d={`M ${iconX - 2.3} ${centerY - 0.1} L ${iconX - 0.7} ${centerY + 1.7} L ${iconX + 2.7} ${centerY - 2.3}`}
        fill="none"
        stroke={iconStroke}
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.8}
      />
      <text
        x={textX}
        y={centerY}
        dominantBaseline="central"
        textAnchor="middle"
        fill={textFill}
        className="text-[9px] font-semibold"
      >
        {significance.significant ? "sig" : "n.s."}
      </text>
    </g>
  );
}

function languageMetricSignificanceKey(slideKey: LanguageMetricSlideKey): string {
  if (slideKey === "resolution") return "officialPassAt1";
  if (slideKey === "context") return "contextF1";
  if (slideKey === "skills") return "skillInvocations";
  return "toolInvocations";
}

function languageMetricSignificanceTooltip(stat: PairedSignificance): string {
  const label = stat.significant ? "significant" : "not significant";
  return `${label}, n=${stat.n}, q=${formatPValue(stat.qValue)}, ${stat.effectLabel ?? "Effect"}: ${formatSignificanceEffect(stat)}, 95% CI ${formatSignificanceInterval(stat)}`;
}

type LanguageResolutionStat = {
  resolved: number;
  total: number;
};

type MetricGroupKey = "language" | "bench" | "repositorySize";

const repositorySizeBuckets = [
  { key: "lt-1k", label: "<1k files", min: 0, max: 1_000 },
  { key: "1k-5k", label: "1k-5k files", min: 1_000, max: 5_000 },
  { key: "5k-20k", label: "5k-20k files", min: 5_000, max: 20_000 },
  { key: "20k-50k", label: "20k-50k files", min: 20_000, max: 50_000 },
  { key: "50k-plus", label: "50k+ files", min: 50_000, max: Number.POSITIVE_INFINITY },
] as const;

function metricGroupValue(
  instance: NonNullable<ComparisonCard["variants"][number]["instances"]>[number],
  groupKey: MetricGroupKey,
): string | null {
  if (groupKey === "language") return instance.language;
  if (groupKey === "bench") return instance.bench;

  const trackedFiles = instance.repositorySize?.trackedFiles;
  if (typeof trackedFiles !== "number") return null;
  return repositorySizeBuckets.find((bucket) => trackedFiles >= bucket.min && trackedFiles < bucket.max)?.key ?? null;
}

function formatMetricGroupLabel(value: string, groupKey: MetricGroupKey): string {
  if (groupKey === "language") return formatLanguageLabel(value);
  if (groupKey === "repositorySize") {
    return repositorySizeBuckets.find((bucket) => bucket.key === value)?.label ?? value;
  }
  return value;
}

function sortMetricRows(left: LanguageMetricRow, right: LanguageMetricRow, groupKey: MetricGroupKey): number {
  if (groupKey === "repositorySize") {
    const leftIndex = repositorySizeBuckets.findIndex((bucket) => bucket.key === left.sortKey);
    const rightIndex = repositorySizeBuckets.findIndex((bucket) => bucket.key === right.sortKey);
    return (leftIndex < 0 ? repositorySizeBuckets.length : leftIndex) - (rightIndex < 0 ? repositorySizeBuckets.length : rightIndex);
  }
  if (groupKey === "bench") {
    const benchOrder = sortBench(left.sortKey ?? left.language, right.sortKey ?? right.language);
    if (benchOrder !== 0) return benchOrder;
  }
  const rightTotal = right.baselineTotal + right.treatmentTotal;
  const leftTotal = left.baselineTotal + left.treatmentTotal;
  return rightTotal - leftTotal || left.language.localeCompare(right.language);
}

function getResolutionRows(
  baseline: ComparisonCard["variants"][number],
  treatment: ComparisonCard["variants"][number],
  groupKey: MetricGroupKey,
): LanguageMetricRow[] {
  const baselineStats = getResolutionStatsByGroup(baseline, groupKey);
  const treatmentStats = getResolutionStatsByGroup(treatment, groupKey);
  const groups = new Set([...baselineStats.keys(), ...treatmentStats.keys()]);

  return Array.from(groups)
    .map((group) => {
      const baselineStat = baselineStats.get(group) ?? { resolved: 0, total: 0 };
      const treatmentStat = treatmentStats.get(group) ?? { resolved: 0, total: 0 };
      return {
        language: formatMetricGroupLabel(group, groupKey),
        sortKey: group,
        baseline: formatChartPercent(baselineStat),
        treatment: formatChartPercent(treatmentStat),
        baselineResolved: baselineStat.resolved,
        baselineTotal: baselineStat.total,
        treatmentResolved: treatmentStat.resolved,
        treatmentTotal: treatmentStat.total,
      };
    })
    .filter((row) => row.baselineTotal > 0 || row.treatmentTotal > 0)
    .sort((left, right) => sortMetricRows(left, right, groupKey));
}

function getResolutionStatsByGroup(
  variant: ComparisonCard["variants"][number],
  groupKey: MetricGroupKey,
): Map<string, LanguageResolutionStat> {
  const byGroup = new Map<string, LanguageResolutionStat>();
  for (const instance of variant.instances ?? []) {
    const status = instance.artifacts?.resolutionStatus;
    if (status !== "resolved" && status !== "unresolved") continue;
    const group = metricGroupValue(instance, groupKey);
    if (!group) continue;
    const stats = byGroup.get(group) ?? { resolved: 0, total: 0 };
    byGroup.set(group, {
      resolved: stats.resolved + (status === "resolved" ? 1 : 0),
      total: stats.total + 1,
    });
  }
  return byGroup;
}

function formatChartPercent(stat: LanguageResolutionStat): number {
  if (stat.total === 0) return 0;
  return Number(((stat.resolved / stat.total) * 100).toFixed(1));
}

type QualityTotals = {
  file: QualityBucket;
  symbol: QualityBucket;
  span: QualityBucket;
  line: QualityBucket;
  contextF1Sum: number;
  contextF1Count: number;
  total: number;
};

type QualityBucket = {
  intersection: number;
  goldSize: number;
  predSize: number;
};

function getContextRows(
  baseline: ComparisonCard["variants"][number],
  treatment: ComparisonCard["variants"][number],
  groupKey: MetricGroupKey,
): LanguageMetricRow[] {
  const baselineStats = getContextStatsByGroup(baseline, groupKey);
  const treatmentStats = getContextStatsByGroup(treatment, groupKey);
  const groups = new Set([...baselineStats.keys(), ...treatmentStats.keys()]);

  return Array.from(groups)
    .map((group) => {
      const baselineStat = baselineStats.get(group);
      const treatmentStat = treatmentStats.get(group);
      return {
        language: formatMetricGroupLabel(group, groupKey),
        sortKey: group,
        baseline: baselineStat ? Number(contextF1FromTotals(baselineStat).toFixed(3)) : 0,
        treatment: treatmentStat ? Number(contextF1FromTotals(treatmentStat).toFixed(3)) : 0,
        baselineTotal: baselineStat?.total ?? 0,
        treatmentTotal: treatmentStat?.total ?? 0,
      };
    })
    .filter((row) => row.baselineTotal > 0 || row.treatmentTotal > 0)
    .sort((left, right) => sortMetricRows(left, right, groupKey));
}

function getContextStatsByGroup(
  variant: ComparisonCard["variants"][number],
  groupKey: MetricGroupKey,
): Map<string, QualityTotals> {
  const byGroup = new Map<string, QualityTotals>();
  for (const instance of variant.instances ?? []) {
    if (!hasValidEvaluation(instance)) continue;
    const group = metricGroupValue(instance, groupKey);
    if (!group) continue;
    const stats = byGroup.get(group) ?? {
      file: { intersection: 0, goldSize: 0, predSize: 0 },
      symbol: { intersection: 0, goldSize: 0, predSize: 0 },
      span: { intersection: 0, goldSize: 0, predSize: 0 },
      line: { intersection: 0, goldSize: 0, predSize: 0 },
      contextF1Sum: 0,
      contextF1Count: 0,
      total: 0,
    };
    for (const level of ["file", "span", "line", "symbol"] as const) {
      stats[level].intersection += instance.quality[level].intersection;
      stats[level].goldSize += instance.quality[level].goldSize;
      stats[level].predSize += instance.quality[level].predSize;
    }
    const instanceContextF1 = instanceContextAggregate(instance, "f1");
    if (instanceContextF1 !== null) {
      stats.contextF1Sum += instanceContextF1;
      stats.contextF1Count += 1;
    }
    stats.total += 1;
    byGroup.set(group, stats);
  }
  return byGroup;
}

function contextF1FromTotals(totals: QualityTotals): number {
  if (totals.contextF1Count > 0) return totals.contextF1Sum / totals.contextF1Count;
  const levelF1Values = (["file", "span", "line", "symbol"] as const)
    .map((level) => coveragePrecisionOrNull(totals[level].predSize, totals[level].goldSize, totals[level].intersection))
    .filter((values): values is NonNullable<typeof values> => values !== null)
    .map((values) => f1(values.coverage, values.precision));
  if (levelF1Values.length === 0) return 0;
  return levelF1Values.reduce((sum, value) => sum + value, 0) / levelF1Values.length;
}

type UsageKind = "skills" | "tools";

type LanguageUsageStat = {
  invocations: number;
  total: number;
};

function getUsageRows(
  baseline: ComparisonCard["variants"][number],
  treatment: ComparisonCard["variants"][number],
  kind: UsageKind,
  groupKey: MetricGroupKey,
): LanguageMetricRow[] {
  const baselineStats = getUsageStatsByGroup(baseline, kind, groupKey);
  const treatmentStats = getUsageStatsByGroup(treatment, kind, groupKey);
  const groups = new Set([...baselineStats.keys(), ...treatmentStats.keys()]);

  return Array.from(groups)
    .map((group) => {
      const baselineStat = baselineStats.get(group) ?? { invocations: 0, total: 0 };
      const treatmentStat = treatmentStats.get(group) ?? { invocations: 0, total: 0 };
      return {
        language: formatMetricGroupLabel(group, groupKey),
        sortKey: group,
        baseline: formatUsageAverage(baselineStat),
        treatment: formatUsageAverage(treatmentStat),
        baselineTotal: baselineStat.total,
        treatmentTotal: treatmentStat.total,
        baselineInvocations: baselineStat.invocations,
        treatmentInvocations: treatmentStat.invocations,
      };
    })
    .filter((row) => (row.baselineInvocations ?? 0) > 0 || (row.treatmentInvocations ?? 0) > 0)
    .sort((left, right) => {
      const rightInvocations = (right.baselineInvocations ?? 0) + (right.treatmentInvocations ?? 0);
      const leftInvocations = (left.baselineInvocations ?? 0) + (left.treatmentInvocations ?? 0);
      return rightInvocations - leftInvocations || sortMetricRows(left, right, groupKey);
    });
}

function getUsageStatsByGroup(
  variant: ComparisonCard["variants"][number],
  kind: UsageKind,
  groupKey: MetricGroupKey,
): Map<string, LanguageUsageStat> {
  const byGroup = new Map<string, LanguageUsageStat>();
  for (const instance of variant.instances ?? []) {
    const group = metricGroupValue(instance, groupKey);
    if (!group) continue;
    const stats = byGroup.get(group) ?? { invocations: 0, total: 0 };
    byGroup.set(group, {
      invocations: stats.invocations + (instance[kind]?.totalInvocations ?? 0),
      total: stats.total + 1,
    });
  }
  return byGroup;
}

function formatUsageAverage(stat: LanguageUsageStat): number {
  if (stat.total === 0) return 0;
  return Number((stat.invocations / stat.total).toFixed(2));
}

function getUsageAxis(rows: LanguageMetricRow[]): { domain: [number, number]; ticks: number[] } {
  const maxValue = Math.max(0, ...rows.flatMap((row) => [row.baseline, row.treatment]));
  const upper = maxValue <= 0 ? 1 : Math.ceil(maxValue * 1.2 * 4) / 4;
  return {
    domain: [0, upper],
    ticks: [0, upper * 0.25, upper * 0.5, upper * 0.75, upper].map((value) => Number(value.toFixed(2))),
  };
}
