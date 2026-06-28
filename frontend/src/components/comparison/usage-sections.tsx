// SPDX-License-Identifier: Apache-2.0

import { Bar, BarChart, CartesianGrid, LabelList, ResponsiveContainer, XAxis, YAxis } from "recharts";

import type { ComparisonCard } from "@/data/comparisons";
import { formatPercentDelta, formatSignedFixed, getComparisonPair } from "@/components/comparison/format";
import { getMetricSignificance } from "@/components/comparison/significance";
import { ComparisonSectionShell, DeltaIndicator, DeltaSectionLabel, HelpIcon, SignificanceBadge } from "@/components/comparison/shared";
import type { ComparisonResultsViewMode, DeltaDisplayMode, MetricDelta } from "@/components/comparison/types";
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart";

type UsageBreakdownEntry = {
  name: string;
  averagePerRun: number;
};

const usageChartConfig = {
  invocations: {
    label: "Invocations / Run",
    color: "hsl(var(--chart-1))",
  },
} satisfies ChartConfig;

function hasBreakdownData(usage: { totalInvocations?: number; byType?: UsageBreakdownEntry[] } | undefined): boolean {
  return (usage?.totalInvocations ?? 0) > 0 || (usage?.byType?.length ?? 0) > 0;
}

function hasMcpData(usage: ComparisonCard["variants"][number]["results"]["mcp"] | undefined): boolean {
  return (
    (usage?.toolCalls ?? 0) > 0
    || (usage?.successfulToolCalls ?? 0) > 0
    || (usage?.callsWithResults ?? 0) > 0
    || (usage?.meaningfulCalls ?? 0) > 0
    || (usage?.byTool?.length ?? 0) > 0
  );
}

export function comparisonHasToolUsage(comparison: ComparisonCard): boolean {
  return comparison.variants.some((variant) => hasBreakdownData(variant.results.tools));
}

export function comparisonHasMcpUsage(comparison: ComparisonCard): boolean {
  return comparison.variants.some((variant) => hasMcpData(variant.results.mcp));
}

function UsageSection({
  comparison,
  viewMode,
  deltaDisplayMode,
  collapsible,
  kind,
}: {
  comparison: ComparisonCard;
  viewMode: ComparisonResultsViewMode;
  deltaDisplayMode: DeltaDisplayMode;
  collapsible?: boolean;
  kind: "skills" | "tools";
}) {
  const title = kind === "skills" ? "Skill Usage" : "Tool Usage";
  const metricLabel = kind === "skills" ? "Skill Invocations / Run" : "Tool Calls / Run";
  const explanation = kind === "skills"
    ? "Average number of skill file invocations detected per run."
    : "Average recorded native, non-MCP tool telemetry events per run.";
  const hasData = comparison.variants.some((variant) => hasBreakdownData(variant.results[kind]));
  if (!hasData) return null;

  const comparisonPair = getComparisonPair(comparison);
  const showDeltas = viewMode === "treatment-delta" && comparisonPair;
  const variants = showDeltas ? [comparisonPair.treatment] : comparison.variants;
  const headerAside = showDeltas
    ? <DeltaSectionLabel baseline={comparisonPair.baseline} treatment={comparisonPair.treatment} />
    : undefined;

  return (
    <ComparisonSectionShell title={title} collapsible={collapsible} headerAside={headerAside}>
      <div className="rounded-lg bg-background p-5">
        <div className="grid gap-5 md:grid-cols-2">
          {variants.map((variant) => {
            const usage = variant.results[kind];
            const delta = showDeltas
              ? usageDelta(kind, comparisonPair.baseline, comparisonPair.treatment, deltaDisplayMode)
              : null;
            const significance = showDeltas ? getMetricSignificance(comparison, kind === "skills" ? "skillInvocations" : "toolInvocations") : null;
            return (
              <div key={variant.label}>
                <div className="mb-4 text-sm font-medium text-muted-foreground">{variant.name}</div>
                <div className="rounded-md border p-4">
                  <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground">
                    <span>{metricLabel}</span>
                    <HelpIcon label={metricLabel} explanation={explanation} />
                  </div>
                  <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
                    <div className="font-medium">{(usage?.averageInvocationsPerRun ?? 0).toFixed(2)}</div>
                    {delta ? (
                      <div className="flex flex-wrap items-center gap-2">
                        <DeltaIndicator label={delta.label} delta={delta.delta} tone={delta.tone} />
                        <SignificanceBadge stat={significance} />
                      </div>
                    ) : null}
                  </div>
                </div>
                <UsageBreakdownChart entries={usage?.byType ?? []} />
              </div>
            );
          })}
        </div>
      </div>
    </ComparisonSectionShell>
  );
}

function UsageBreakdownChart({ entries }: { entries: UsageBreakdownEntry[] }) {
  if (entries.length === 0) {
    return (
      <p className="mt-3 text-sm text-muted-foreground">
        No per-type invocation breakdown available.
      </p>
    );
  }

  const chartData = entries
    .map((entry) => ({
      name: entry.name,
      invocations: Number(entry.averagePerRun.toFixed(2)),
    }))
    .sort((a, b) => b.invocations - a.invocations);
  const chartHeight = Math.max(140, chartData.length * 36 + 28);

  return (
    <ChartContainer config={usageChartConfig} className="mt-3 w-full" style={{ height: chartHeight }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          accessibilityLayer
          data={chartData}
          layout="vertical"
          margin={{ left: 4, right: 44, top: 8, bottom: 8 }}
        >
          <CartesianGrid horizontal={false} />
          <YAxis
            dataKey="name"
            type="category"
            tickLine={false}
            tickMargin={10}
            axisLine={false}
            width={190}
          />
          <XAxis dataKey="invocations" type="number" hide />
          <ChartTooltip
            cursor={false}
            content={
              <ChartTooltipContent
                hideLabel
                formatter={(value) => Number(value).toFixed(2)}
              />
            }
          />
          <Bar dataKey="invocations" fill="var(--color-invocations)" radius={4}>
            <LabelList
              dataKey="invocations"
              position="right"
              offset={8}
              className="fill-foreground text-xs font-medium"
              formatter={(value: number) => value.toFixed(2)}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartContainer>
  );
}

function usageDelta(
  kind: "skills" | "tools",
  baseline: ComparisonCard["variants"][number],
  treatment: ComparisonCard["variants"][number],
  displayMode: DeltaDisplayMode,
): MetricDelta {
  const baselineValue = baseline.results[kind]?.averageInvocationsPerRun ?? 0;
  const treatmentValue = treatment.results[kind]?.averageInvocationsPerRun ?? 0;
  const delta = treatmentValue - baselineValue;
  const percentDelta = baselineValue === 0 ? (delta === 0 ? 0 : null) : (delta / Math.abs(baselineValue)) * 100;
  return {
    delta,
    label: displayMode === "percent"
      ? (percentDelta === null ? "n/a" : formatPercentDelta(percentDelta))
      : formatSignedFixed(delta, 2),
    tone: "neutral",
  };
}

export function SkillUsageSection(props: Omit<Parameters<typeof UsageSection>[0], "kind">) {
  return <UsageSection {...props} kind="skills" />;
}

export function ToolUsageSection(props: Omit<Parameters<typeof UsageSection>[0], "kind">) {
  return <UsageSection {...props} kind="tools" />;
}

export function McpUsageSection({
  comparison,
  viewMode,
  deltaDisplayMode,
  collapsible,
}: {
  comparison: ComparisonCard;
  viewMode: ComparisonResultsViewMode;
  deltaDisplayMode: DeltaDisplayMode;
  collapsible?: boolean;
}) {
  const hasData = comparisonHasMcpUsage(comparison);
  if (!hasData) return null;

  const comparisonPair = getComparisonPair(comparison);
  const showDeltas = viewMode === "treatment-delta" && comparisonPair;
  const variants = showDeltas ? [comparisonPair.treatment] : comparison.variants;
  const headerAside = showDeltas
    ? <DeltaSectionLabel baseline={comparisonPair.baseline} treatment={comparisonPair.treatment} />
    : undefined;

  return (
    <ComparisonSectionShell title="MCP Usage" collapsible={collapsible} headerAside={headerAside}>
      <div className="rounded-lg bg-background p-5">
        <div className="grid gap-5 xl:grid-cols-2">
          {variants.map((variant) => {
            const usage = variant.results.mcp;
            return (
              <div key={variant.label}>
                <div className="mb-4 text-sm font-medium text-muted-foreground">{variant.name}</div>
                <div className="grid gap-3 sm:grid-cols-2">
                  <McpMetricCard
                    label="MCP Calls"
                    value={usage?.toolCalls ?? 0}
                    explanation="Total MCP calls exported from native MCP events or generic mcp-tool bridge commands."
                    delta={showDeltas ? mcpDelta("toolCalls", comparisonPair.baseline, comparisonPair.treatment, deltaDisplayMode) : null}
                  />
                  <McpMetricCard
                    label="Successful Calls"
                    value={usage?.successfulToolCalls ?? 0}
                    explanation="MCP calls whose trace status or command exit code indicated success."
                    delta={showDeltas ? mcpDelta("successfulToolCalls", comparisonPair.baseline, comparisonPair.treatment, deltaDisplayMode) : null}
                  />
                  <McpMetricCard
                    label="Calls With Results"
                    value={usage?.callsWithResults ?? 0}
                    explanation="MCP calls whose exported payload contained result, match, item, or rule rows."
                    delta={showDeltas ? mcpDelta("callsWithResults", comparisonPair.baseline, comparisonPair.treatment, deltaDisplayMode) : null}
                  />
                  <McpMetricCard
                    label="Meaningful Calls"
                    value={usage?.meaningfulCalls ?? 0}
                    explanation="MCP calls that returned paths later inspected by the agent or overlapping final context or patch files."
                    delta={showDeltas ? mcpDelta("meaningfulCalls", comparisonPair.baseline, comparisonPair.treatment, deltaDisplayMode) : null}
                  />
                  <McpMetricCard
                    label="Instances With MCP"
                    value={usage?.instancesWithMcpCalls ?? 0}
                    explanation="Included instances with at least one exported MCP call."
                  />
                  <McpMetricCard
                    label="Instances Meaningful"
                    value={usage?.instancesWithMeaningfulMcpUse ?? 0}
                    explanation="Included instances with at least one MCP call linked to follow-up inspection, final context, or patch overlap."
                  />
                </div>
                <McpToolBreakdown entries={usage?.byTool ?? []} />
              </div>
            );
          })}
        </div>
      </div>
    </ComparisonSectionShell>
  );
}

function McpMetricCard({
  label,
  value,
  explanation,
  delta,
}: {
  label: string;
  value: number;
  explanation: string;
  delta?: MetricDelta | null;
}) {
  return (
    <div className="rounded-md border p-4">
      <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground">
        <span>{label}</span>
        <HelpIcon label={label} explanation={explanation} />
      </div>
      <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
        <div className="font-medium tabular-nums">{value}</div>
        {delta ? <DeltaIndicator label={delta.label} delta={delta.delta} tone={delta.tone} /> : null}
      </div>
    </div>
  );
}

function McpToolBreakdown({ entries }: { entries: NonNullable<ComparisonCard["variants"][number]["results"]["mcp"]>["byTool"] }) {
  if (!entries || entries.length === 0) {
    return <p className="mt-3 text-sm text-muted-foreground">No per-tool MCP breakdown available.</p>;
  }

  return (
    <div className="mt-4 space-y-2">
      {entries.map((entry) => (
        <div key={entry.name} className="grid grid-cols-[minmax(0,1fr)_auto_auto] items-center gap-3 rounded-md border px-3 py-2 text-sm">
          <span className="min-w-0 break-all">{entry.name}</span>
          <span className="text-muted-foreground">ok {entry.successfulCalls ?? 0}</span>
          <span className="font-medium tabular-nums">{entry.calls}</span>
        </div>
      ))}
    </div>
  );
}

function mcpDelta(
  key: "toolCalls" | "successfulToolCalls" | "callsWithResults" | "meaningfulCalls",
  baseline: ComparisonCard["variants"][number],
  treatment: ComparisonCard["variants"][number],
  displayMode: DeltaDisplayMode,
): MetricDelta {
  const baselineValue = baseline.results.mcp?.[key] ?? 0;
  const treatmentValue = treatment.results.mcp?.[key] ?? 0;
  const delta = treatmentValue - baselineValue;
  const percentDelta = baselineValue === 0 ? (delta === 0 ? 0 : null) : (delta / Math.abs(baselineValue)) * 100;
  return {
    delta,
    label: displayMode === "percent"
      ? (percentDelta === null ? "n/a" : formatPercentDelta(percentDelta))
      : formatSignedFixed(delta, 0),
    tone: "neutral",
  };
}
