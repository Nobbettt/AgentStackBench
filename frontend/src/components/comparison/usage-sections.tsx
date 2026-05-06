
import type { ComparisonCard } from "@/data/comparisons";
import { formatPercentDelta, formatSignedFixed, getComparisonPair } from "@/components/comparison/format";
import { ComparisonSectionShell, DeltaIndicator, DeltaSectionLabel, HelpIcon } from "@/components/comparison/shared";
import type { ComparisonResultsViewMode, DeltaDisplayMode, MetricDelta } from "@/components/comparison/types";
import { cn } from "@/lib/utils";

function hasBreakdownData(usage: { totalInvocations?: number; byType?: Array<{ name: string; averagePerRun: number }> } | undefined): boolean {
  return (usage?.totalInvocations ?? 0) > 0 || (usage?.byType?.length ?? 0) > 0;
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
    : "Average recorded tool or MCP telemetry events per run.";
  const hasData = comparison.variants.some((variant) => hasBreakdownData(variant.results[kind]));
  if (!hasData) return null;

  const comparisonPair = getComparisonPair(comparison);
  const variants = viewMode === "treatment-delta" && comparisonPair ? [comparisonPair.treatment] : comparison.variants;
  const headerAside = viewMode === "treatment-delta" && comparisonPair
    ? <DeltaSectionLabel baseline={comparisonPair.baseline} treatment={comparisonPair.treatment} />
    : undefined;

  return (
    <ComparisonSectionShell title={title} collapsible={collapsible} headerAside={headerAside}>
      <div className={cn(collapsible ? "rounded-lg bg-background p-5" : "rounded-lg border bg-background p-5")}>
        <div className="grid gap-5 md:grid-cols-2">
          {variants.map((variant) => {
            const usage = variant.results[kind];
            const delta = comparisonPair
              ? usageDelta(kind, comparisonPair.baseline, comparisonPair.treatment, deltaDisplayMode)
              : null;
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
                    {delta ? <DeltaIndicator label={delta.label} delta={delta.delta} tone={delta.tone} /> : null}
                  </div>
                </div>
                {(usage?.byType?.length ?? 0) > 0 ? (
                  <div className="mt-3 grid gap-2">
                    {usage?.byType?.map((entry) => (
                      <div key={entry.name} className="flex items-center justify-between rounded-md border px-3 py-2 text-sm">
                        <span>{entry.name}</span>
                        <span className="font-medium tabular-nums">{entry.averagePerRun.toFixed(2)}</span>
                      </div>
                    ))}
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
