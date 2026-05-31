import { Fragment, useMemo, useState, type ReactNode } from "react";
import {
  Columns2,
  ExternalLink,
  Minus,
  Percent,
  TrendingDown,
  TrendingUp,
  TrendingUpDown,
} from "lucide-react";

import type { ComparisonCard, ComparisonInstanceDetail } from "@/data/comparisons";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { TooltipProvider } from "@/components/ui/tooltip";
import { TrajectoryTableHead } from "@/components/comparison/detail-section";
import { FinalAnswerSection, ModelPatchSection } from "@/components/comparison/instance-detail-panels";
import { buildInstanceComparison, buildInstanceRows } from "@/components/comparison/instance-data";
import { compactBenchLabel, compactInstanceName } from "@/components/comparison/instance-naming";
import {
  ContextRetrievalMetricSection,
  PatchOverlapBetweenVariantsSection,
  ResolutionMetricSection,
  ResourceUsageMetricSection,
} from "@/components/comparison/metric-sections";
import {
  formatInstanceMetric,
  formatLanguageLabel,
  formatResolutionStatus,
  resolutionStatusClassName,
  getComparisonPair,
  deltaIndicatorClassName,
} from "@/components/comparison/format";
import {
  contextRetrievalMetricDefinitions,
  resolutionMetricDefinitions,
  resourceMetricDefinitions,
} from "@/components/comparison/metrics";
import { HelpIcon, MetricDirectionBadge } from "@/components/comparison/shared";
import { TraceSection } from "@/components/comparison/trace-section";
import type { ComparisonResultsViewMode, DeltaDisplayMode, MetricDefinition } from "@/components/comparison/types";
import { cn } from "@/lib/utils";

type InstanceDetailTab = "overview" | "resolution" | "context" | "resources" | "skills" | "mcp" | "answer" | "trajectory" | "patch" | "trace";

const instanceDetailTabs: Array<{ id: InstanceDetailTab; label: string }> = [
  { id: "overview", label: "Overview" },
  { id: "resolution", label: "Resolution" },
  { id: "context", label: "Context" },
  { id: "resources", label: "Resources" },
  { id: "skills", label: "Skills" },
  { id: "mcp", label: "MCP" },
  { id: "answer", label: "Final Answer" },
  { id: "trajectory", label: "Trajectory" },
  { id: "patch", label: "Model Patch" },
  { id: "trace", label: "Trace" },
];

const segmentedControlClassName = "gap-0 rounded-md shadow-lg backdrop-blur";
const segmentedControlItemClassName = "w-14 rounded-none bg-background px-0 data-[state=on]:bg-primary data-[state=on]:text-primary-foreground";
const trajectoryExplanation = "Shows how cumulative retrieved context coverage changes over retrieval steps. Each row is a step, and the columns measure file, block, line, and symbol-level overlap with the gold context.";
const instanceResourceMetricDefinitions: MetricDefinition[] = resourceMetricDefinitions.map((metric) => {
  if (metric.key === "averageSteps") {
    return {
      ...metric,
      label: "Steps",
      explanation: "Inferred retrieval steps for this run.",
    };
  }
  if (metric.key === "averageDuration") {
    return {
      ...metric,
      label: "Scored Task Duration",
      explanation: "Scored agent task runtime for this run; excludes setup, bootstrap, and validation.",
    };
  }
  if (metric.key === "totalTokens") {
    return {
      ...metric,
      label: "Tokens",
      explanation: "Tokens consumed by this run.",
    };
  }
  if (metric.key === "estimatedCost") {
    return {
      ...metric,
      explanation: "Inference cost for this run when metadata is available.",
    };
  }
  return metric;
});

const overviewMetricDefinitions = [
  ...resolutionMetricDefinitions.filter((metric) => ["fixOverlapVsGoldF1"].includes(metric.key)),
  ...contextRetrievalMetricDefinitions.filter((metric) => ["contextF1", "fileF1", "spanF1"].includes(metric.key)),
  ...instanceResourceMetricDefinitions.filter((metric) => ["averageSteps", "averageDuration", "totalTokens", "estimatedCost"].includes(metric.key)),
];

export function ComparisonInstanceDetailPage({
  comparison,
  instanceId,
  detail,
  detailError,
}: {
  comparison: ComparisonCard;
  instanceId: string;
  detail: ComparisonInstanceDetail | null | undefined;
  detailError?: string | null;
}) {
  const row = useMemo(
    () => buildInstanceRows(comparison).find((instanceRow) => instanceRow.instanceId === instanceId) ?? null,
    [comparison, instanceId],
  );
  const instanceComparison = row ? buildInstanceComparison(comparison, row) : null;
  const [viewMode, setViewMode] = useState<ComparisonResultsViewMode>("treatment-delta");
  const [deltaDisplayMode, setDeltaDisplayMode] = useState<DeltaDisplayMode>("percent");
  const [activeTab, setActiveTab] = useState<InstanceDetailTab>("overview");

  if (!row || !instanceComparison) {
    return <section className="rounded-lg border bg-background p-6 text-sm text-muted-foreground">Instance detail not found in the current comparison snapshot.</section>;
  }

  return (
    <TooltipProvider>
      <div className="space-y-4 pb-40">
        <InstanceHeader row={row} />
        <InstanceDetailTabs activeTab={activeTab} onChange={setActiveTab} />
        {renderInstanceDetailTab({
          activeTab,
          row,
          instanceComparison,
          viewMode,
          deltaDisplayMode,
          detail,
          detailError,
        })}
        <DetailControls
          viewMode={viewMode}
          deltaDisplayMode={deltaDisplayMode}
          onViewModeChange={setViewMode}
          onDeltaDisplayModeChange={setDeltaDisplayMode}
        />
      </div>
    </TooltipProvider>
  );
}

function renderInstanceDetailTab({
  activeTab,
  row,
  instanceComparison,
  viewMode,
  deltaDisplayMode,
  detail,
  detailError,
}: {
  activeTab: InstanceDetailTab;
  row: ReturnType<typeof buildInstanceRows>[number];
  instanceComparison: ComparisonCard;
  viewMode: ComparisonResultsViewMode;
  deltaDisplayMode: DeltaDisplayMode;
  detail: ComparisonInstanceDetail | null | undefined;
  detailError?: string | null;
}) {
  if (activeTab === "overview") {
    return (
      <InstanceOverview
        row={row}
        instanceComparison={instanceComparison}
        viewMode={viewMode}
      />
    );
  }

  if (activeTab === "resolution") {
    return (
      <div className="space-y-6">
        <ResolutionMetricSection
          comparison={instanceComparison}
          viewMode={viewMode}
          deltaDisplayMode={deltaDisplayMode}
          treatmentDeltaDisplay="versus"
          nonGraphDisplay
        />
        {detail ? <PatchOverlapBetweenVariantsSection overlap={detail.fixOverlapBetweenVariants} /> : null}
      </div>
    );
  }

  if (activeTab === "context") {
    return <ContextRetrievalMetricSection comparison={instanceComparison} viewMode={viewMode} deltaDisplayMode={deltaDisplayMode} treatmentDeltaDisplay="versus" />;
  }

  if (activeTab === "resources") {
    return (
      <ResourceUsageMetricSection
        comparison={instanceComparison}
        viewMode={viewMode}
        deltaDisplayMode={deltaDisplayMode}
        treatmentDeltaDisplay="versus"
        metricDefinitions={instanceResourceMetricDefinitions}
        nonGraphDisplay
      />
    );
  }

  if (activeTab === "skills") {
    return <SkillUseSection comparison={instanceComparison} viewMode={viewMode} />;
  }

  return renderDetailBackedTab(activeTab, detail, detailError, viewMode);
}

function renderDetailBackedTab(
  activeTab: InstanceDetailTab,
  detail: ComparisonInstanceDetail | null | undefined,
  detailError: string | null | undefined,
  viewMode: ComparisonResultsViewMode,
) {
  if (detailError) {
    return <StatusPanel tone="danger" message={`Unable to load detailed trajectory and trace data: ${detailError}`} />;
  }
  if (detail === undefined) {
    return <StatusPanel message="Loading detailed trajectory and trace data..." />;
  }
  if (detail === null) {
    return <StatusPanel message="Detailed trajectory and trace data is not available for this instance." />;
  }

  const variants = detailVariantsForViewMode(detail, viewMode);
  if (activeTab === "mcp") return <McpUseSection variants={detail.variants} />;
  if (activeTab === "answer") return <FinalAnswerSection variants={detail.variants} />;
  if (activeTab === "trajectory") return <TrajectorySection variants={viewMode === "treatment-delta" ? detail.variants : variants} viewMode={viewMode} />;
  if (activeTab === "patch") return <ModelPatchSection variants={detail.variants} />;
  if (activeTab === "trace") return <TraceSection variants={detail.variants} />;

  return null;
}

function detailVariantsForViewMode(detail: ComparisonInstanceDetail, viewMode: ComparisonResultsViewMode): ComparisonInstanceDetail["variants"] {
  if (viewMode === "treatment-delta" && detail.variants.length > 1) {
    return [detail.variants[1]];
  }
  return detail.variants;
}

function StatusPanel({ message, tone = "muted" }: { message: string; tone?: "muted" | "danger" }) {
  return (
    <section className={cn("rounded-lg bg-background p-6 text-sm", tone === "danger" ? "text-rose-700" : "text-muted-foreground")}>
      {message}
    </section>
  );
}

function SectionTitleWithHelp({ title, explanation }: { title: string; explanation: string }) {
  return (
    <div className="flex items-center gap-2">
      <h2 className="text-xl font-semibold tracking-tight">{title}</h2>
      <HelpIcon label={title} explanation={explanation} />
    </div>
  );
}

function InstanceHeader({
  row,
}: {
  row: ReturnType<typeof buildInstanceRows>[number];
}) {
  return (
    <section className="rounded-lg bg-background px-4 py-3">
      <h1 className="text-2xl font-semibold tracking-tight [overflow-wrap:anywhere] sm:text-3xl">{row.instanceId}</h1>
      <p className="mt-1 text-sm text-muted-foreground">
        {row.bench} / {formatLanguageLabel(row.language)}
        {row.originalInstanceId ? ` / ${row.originalInstanceId}` : ""}
      </p>
    </section>
  );
}

function InstanceDetailTabs({ activeTab, onChange }: { activeTab: InstanceDetailTab; onChange: (tab: InstanceDetailTab) => void }) {
  return (
    <div className="overflow-x-auto border-b" role="tablist" aria-label="Instance detail sections">
      <div className="flex min-w-max gap-1">
        {instanceDetailTabs.map((tab) => {
          const selected = tab.id === activeTab;
          return (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={selected}
              className={`border-b-2 px-4 py-3 text-sm font-medium transition-colors ${
                selected
                  ? "border-primary text-foreground"
                  : "border-transparent text-muted-foreground hover:border-border hover:text-foreground"
              }`}
              onClick={() => onChange(tab.id)}
            >
              {tab.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function InstanceOverview({
  row,
  instanceComparison,
  viewMode,
}: {
  row: ReturnType<typeof buildInstanceRows>[number];
  instanceComparison: ComparisonCard;
  viewMode: ComparisonResultsViewMode;
}) {
  return (
    <section className="space-y-6">
      <OverviewRunSummary comparison={instanceComparison} viewMode={viewMode} />
      <SetupDatasetSection
        row={row}
        instanceComparison={instanceComparison}
        viewMode={viewMode}
      />
    </section>
  );
}

function OverviewRunSummary({
  comparison,
  viewMode,
}: {
  comparison: ComparisonCard;
  viewMode: ComparisonResultsViewMode;
}) {
  const visibleMetrics = overviewMetricDefinitions.filter((metric) => comparison.variants.some((variant) => metric.value(variant) !== "—"));
  const comparisonPair = getComparisonPair(comparison);

  if (viewMode === "treatment-delta" && comparisonPair) {
    return (
      <section className="space-y-4">
        <h2 className="text-xl font-semibold tracking-tight">Run Summary</h2>
        <div className="rounded-lg bg-background p-5">
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <OverviewResolutionComparisonCard baseline={comparisonPair.baseline} treatment={comparisonPair.treatment} />
            {visibleMetrics.map((metric) => (
              <OverviewMetricComparisonCard
                key={metric.key}
                metric={metric}
                baseline={comparisonPair.baseline}
                treatment={comparisonPair.treatment}
              />
            ))}
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="space-y-4">
      <h2 className="text-xl font-semibold tracking-tight">Run Summary</h2>
      <div className="grid gap-5 lg:grid-cols-2">
        {comparison.variants.map((variant) => (
          <div key={variant.label} className="rounded-lg bg-background p-5">
            <div className="mb-4 text-sm font-medium text-muted-foreground">{variant.name}</div>
            <div className="grid gap-3 sm:grid-cols-2">
              <OverviewResolutionCard variant={variant} />
              {visibleMetrics.map((metric) => (
                <OverviewMetricCard
                  key={`${variant.label}-${metric.key}`}
                  label={overviewMetricLabel(metric)}
                  explanation={overviewMetricExplanation(metric)}
                  direction={metric.direction}
                  value={metric.value(variant)}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function OverviewResolutionCard({ variant }: { variant: ComparisonCard["variants"][number] }) {
  const status = variant.instances?.[0]?.artifacts?.resolutionStatus;

  return (
    <SummaryCard
      label="Pass@1"
      value={formatResolutionStatus(status)}
      className={resolutionStatusClassName(status)}
    />
  );
}

function OverviewResolutionComparisonCard({
  baseline,
  treatment,
}: {
  baseline: ComparisonCard["variants"][number];
  treatment: ComparisonCard["variants"][number];
}) {
  const baselineStatus = baseline.instances?.[0]?.artifacts?.resolutionStatus;
  const treatmentStatus = treatment.instances?.[0]?.artifacts?.resolutionStatus;
  const baselineValue = formatResolutionStatus(baselineStatus);
  const treatmentValue = formatResolutionStatus(treatmentStatus);
  return (
    <div className="rounded-md border p-4">
      <div className="text-xs uppercase tracking-wide text-muted-foreground">Pass@1</div>
      <div className="mt-3 grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-center gap-2">
        <OverviewComparisonValue
          value={baselineValue}
          className={resolutionStatusClassName(baselineStatus)}
        />
        <OverviewDirectionSeparator
          direction="higher"
          baselineValue={baselineValue}
          treatmentValue={treatmentValue}
          baselineNumericValue={resolutionStatusScore(baselineStatus)}
          treatmentNumericValue={resolutionStatusScore(treatmentStatus)}
        />
        <OverviewComparisonValue
          value={treatmentValue}
          className={resolutionStatusClassName(treatmentStatus)}
        />
      </div>
    </div>
  );
}

function OverviewMetricComparisonCard({
  metric,
  baseline,
  treatment,
}: {
  metric: (typeof overviewMetricDefinitions)[number];
  baseline: ComparisonCard["variants"][number];
  treatment: ComparisonCard["variants"][number];
}) {
  const baselineValue = metric.value(baseline);
  const treatmentValue = metric.value(treatment);
  return (
    <div className="rounded-md border p-4">
      <div className="flex flex-wrap items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground">
        <span>{overviewMetricLabel(metric)}</span>
        <MetricDirectionBadge direction={metric.direction} />
        <HelpIcon label={overviewMetricLabel(metric)} explanation={overviewMetricExplanation(metric)} />
      </div>
      <div className="mt-3 grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-center gap-2">
        <OverviewComparisonValue value={baselineValue} />
        <OverviewDirectionSeparator
          direction={metric.direction}
          baselineValue={baselineValue}
          treatmentValue={treatmentValue}
          baselineNumericValue={metric.parse(baselineValue)}
          treatmentNumericValue={metric.parse(treatmentValue)}
        />
        <OverviewComparisonValue value={treatmentValue} />
      </div>
    </div>
  );
}

function OverviewDirectionSeparator({
  direction,
  baselineValue,
  treatmentValue,
  baselineNumericValue,
  treatmentNumericValue,
  className,
  iconClassName,
}: {
  direction: (typeof overviewMetricDefinitions)[number]["direction"];
  baselineValue: string;
  treatmentValue: string;
  baselineNumericValue?: number | null;
  treatmentNumericValue?: number | null;
  className?: string;
  iconClassName?: string;
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
    <div
      className={cn("flex h-8 w-8 items-center justify-center", tone, className)}
      aria-label="Baseline to treatment"
    >
      <Icon className={cn("h-5 w-5", iconClassName)} />
    </div>
  );
}

function resolutionStatusScore(status: string | undefined): number | null {
  const normalized = (status ?? "").trim().toLowerCase();
  if (normalized === "resolved") return 1;
  if (["unresolved", "error", "missing"].includes(normalized)) return 0;
  return null;
}

function OverviewComparisonValue({
  value,
  className,
}: {
  value: string;
  className?: string;
}) {
  return (
    <div className="min-w-0 rounded-md bg-muted/30 px-3 py-2 text-center">
      <div className={cn("truncate text-sm font-medium tabular-nums", className)} title={value}>{value}</div>
    </div>
  );
}

function OverviewMetricCard({
  label,
  explanation,
  direction,
  value,
}: {
  label: string;
  explanation: string;
  direction: (typeof overviewMetricDefinitions)[number]["direction"];
  value: string;
}) {
  return (
    <div className="rounded-md border p-4">
      <div className="flex flex-wrap items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground">
        <span>{label}</span>
        <MetricDirectionBadge direction={direction} />
        <HelpIcon label={label} explanation={explanation} />
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-3">
        <span className="font-medium tabular-nums">{value}</span>
      </div>
    </div>
  );
}

function overviewMetricLabel(metric: (typeof overviewMetricDefinitions)[number]): string {
  if (metric.key === "spanF1") return "Block F1";
  return metric.label;
}

function overviewMetricExplanation(metric: (typeof overviewMetricDefinitions)[number]): string {
  if (metric.key === "contextF1") return "Balanced file/symbol/block F1 score.";
  if (metric.key === "spanF1") return "Block-level retrieval F1.";
  return metric.explanation;
}

function SetupDatasetSection({
  row,
  instanceComparison,
  viewMode,
}: {
  row: ReturnType<typeof buildInstanceRows>[number];
  instanceComparison: ComparisonCard;
  viewMode: ComparisonResultsViewMode;
}) {
  const setupVariants = setupVariantsForViewMode(instanceComparison, viewMode);
  const comparisonPair = getComparisonPair(instanceComparison);
  const setupBaseline = viewMode === "treatment-delta" && comparisonPair ? comparisonPair.baseline : undefined;
  const name = compactInstanceName(row);

  return (
    <section className="space-y-4">
      <h2 className="text-xl font-semibold tracking-tight">Setup & Dataset</h2>
      <div className="rounded-lg bg-background p-5">
        <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
          <div className="min-w-0">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Task</div>
            <div className="mt-2 font-mono text-2xl font-semibold tabular-nums text-foreground">{name.shortId}</div>
            <div className="mt-1 text-lg font-semibold [overflow-wrap:anywhere]">{name.repo}</div>
            <div className="mt-3 flex flex-wrap items-center gap-1.5 text-xs uppercase tracking-wide text-muted-foreground">
              <span>{compactBenchLabel(row.bench)}</span>
              <span aria-hidden="true">/</span>
              <span>{formatLanguageLabel(row.language)}</span>
              {name.taskTypeParts.map((part) => (
                <Fragment key={part}>
                  <span aria-hidden="true">/</span>
                  <span>{part}</span>
                </Fragment>
              ))}
            </div>
          </div>
          <div className="grid gap-3 text-sm lg:min-w-[24rem]">
            <TaskIdentityRow
              label="Original Issue"
              value={row.originalInstanceId ?? "—"}
              links={[
                ...(name.pullRequestUrl ? [{ label: `PR #${name.pullRequestNumber}`, href: name.pullRequestUrl }] : []),
                ...(name.relatedIssuesUrl ? [{ label: "Related Issues", href: name.relatedIssuesUrl }] : []),
              ]}
            />
            <TaskIdentityRow label="Instance ID" value={row.instanceId} />
          </div>
        </div>
      </div>
      <RunSetupSection variants={setupVariants} baseline={setupBaseline} />
    </section>
  );
}

function TaskIdentityRow({
  label,
  value,
  links = [],
}: {
  label: string;
  value: string;
  links?: Array<{ label: string; href: string }>;
}) {
  return (
    <div className="min-w-0 rounded-md border bg-muted/20 px-3 py-2">
      <div className="text-[11px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-1 font-medium [overflow-wrap:anywhere]">{value}</div>
      {links.length > 0 ? (
        <div className="mt-2 flex flex-wrap gap-2">
          {links.map((link) => (
            <a
              key={link.label}
              href={link.href}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1.5 rounded-md border bg-background px-2 py-1 text-xs font-medium text-primary transition-colors hover:bg-muted"
            >
              {link.label}
              <ExternalLink className="h-3.5 w-3.5" />
            </a>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function SummaryCard({ label, value, note, className }: { label: string; value: string; note?: string; className?: string }) {
  return (
    <div className="rounded-md border p-4">
      <div className="text-xs uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className={cn("mt-2 text-sm font-medium [overflow-wrap:anywhere]", className)}>{value}</div>
      {note ? <div className="mt-1.5 break-words text-xs leading-5 text-muted-foreground">{note}</div> : null}
    </div>
  );
}

function DetailControls({
  viewMode,
  deltaDisplayMode,
  onViewModeChange,
  onDeltaDisplayModeChange,
}: {
  viewMode: ComparisonResultsViewMode;
  deltaDisplayMode: DeltaDisplayMode;
  onViewModeChange: (value: ComparisonResultsViewMode) => void;
  onDeltaDisplayModeChange: (value: DeltaDisplayMode) => void;
}) {
  return (
    <section className="flex w-full flex-col items-end gap-2 sm:fixed sm:bottom-4 sm:right-4 sm:z-40 sm:w-auto">
      <div
        className={`overflow-hidden transition-all duration-200 ${
          viewMode === "treatment-delta"
            ? "max-h-10 translate-y-0 opacity-100"
            : "max-h-0 translate-y-1 opacity-0"
        }`}
        aria-hidden={viewMode !== "treatment-delta"}
      >
        <ToggleGroup
          type="single"
          variant="outline"
          value={deltaDisplayMode}
          onValueChange={(value) => value && onDeltaDisplayModeChange(value as DeltaDisplayMode)}
          className={segmentedControlClassName}
        >
          <ToggleGroupItem
            value="percent"
            aria-label="Percent diff"
            title="Percent diff"
            className={`${segmentedControlItemClassName} rounded-l-md border-r-0`}
          >
            <Percent className="h-4 w-4" />
          </ToggleGroupItem>
          <ToggleGroupItem
            value="absolute"
            aria-label="Numerical diff"
            title="Numerical diff"
            className={`${segmentedControlItemClassName} rounded-r-md`}
          >
            <span className="font-semibold tabular-nums">1.2→</span>
          </ToggleGroupItem>
        </ToggleGroup>
      </div>
      <ToggleGroup
        type="single"
        variant="outline"
        value={viewMode}
        onValueChange={(value) => value && onViewModeChange(value as ComparisonResultsViewMode)}
        className={segmentedControlClassName}
      >
        <ToggleGroupItem
          value="treatment-delta"
          aria-label="Treatment delta view"
          title="Treatment delta view"
          className={`${segmentedControlItemClassName} rounded-l-md border-r-0`}
        >
          <TrendingUpDown className="h-4 w-4" />
        </ToggleGroupItem>
        <ToggleGroupItem
          value="side-by-side"
          aria-label="Side by side view"
          title="Side by side view"
          className={`${segmentedControlItemClassName} rounded-r-md`}
        >
          <Columns2 className="h-4 w-4" />
        </ToggleGroupItem>
      </ToggleGroup>
    </section>
  );
}

function RunSetupSection({ variants, baseline }: { variants: ComparisonCard["variants"]; baseline?: ComparisonCard["variants"][number] }) {
  return (
    <div className={variants.length > 1 ? "grid gap-5 lg:grid-cols-2" : "grid gap-5"}>
      {variants.map((variant) => (
        <div key={variant.label} className="rounded-lg bg-background p-5">
          <h3 className="text-sm font-medium text-muted-foreground">{variant.name}</h3>
          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            <SetupSummaryCard label="Model" value={variant.model ?? "—"} baselineValue={baseline?.model} />
            <SetupSummaryCard label="Effort" value={variant.effort ?? "—"} baselineValue={baseline?.effort} />
            <SetupSummaryCard
              label="Mounted Resources"
              value={setupParameterValue(variant, "Mounted Resources")}
              baselineValue={baseline ? setupParameterValue(baseline, "Mounted Resources") : undefined}
            />
            <SetupSummaryCard
              label="Additional Prompt"
              value={setupParameterValue(variant, "Additional Prompt", "Setup Prompt")}
              baselineValue={baseline ? setupParameterValue(baseline, "Additional Prompt", "Setup Prompt") : undefined}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function SetupSummaryCard({
  label,
  value,
  baselineValue,
}: {
  label: string;
  value: string;
  baselineValue?: string;
}) {
  return (
    <SummaryCard
      label={label}
      value={value}
      note={baselineValue === undefined ? undefined : baselineValue === value ? "Matches baseline" : `Baseline: ${baselineValue}`}
    />
  );
}

function setupVariantsForViewMode(
  instanceComparison: ComparisonCard,
  viewMode: ComparisonResultsViewMode,
): ComparisonCard["variants"] {
  const comparisonPair = getComparisonPair(instanceComparison);
  return viewMode === "treatment-delta" && comparisonPair ? [comparisonPair.treatment] : instanceComparison.variants;
}

function setupParameterValue(variant: ComparisonCard["variants"][number], ...labels: string[]): string {
  for (const label of labels) {
    const value = variant.parameters.find((parameter) => parameter.label.toLowerCase() === label.toLowerCase())?.value;
    if (value && value.trim()) return value;
  }
  return "None";
}
function TrajectorySection({
  variants,
  viewMode,
}: {
  variants: ComparisonInstanceDetail["variants"];
  viewMode: ComparisonResultsViewMode;
}) {
  if (viewMode === "treatment-delta" && variants.length >= 2) {
    return <TrajectoryVersusSection baseline={variants[0]} treatment={variants[1]} />;
  }

  return (
    <section className="space-y-4">
      <SectionTitleWithHelp title="Cumulative Evaluated Trajectory" explanation={trajectoryExplanation} />
      <div className={variants.length > 1 ? "grid gap-6 xl:grid-cols-2" : "grid gap-6"}>
        {variants.map((variant) => (
          <div key={variant.label} className="h-full rounded-lg bg-background p-5">
            <h3 className="text-lg font-semibold">{variant.name}</h3>
            {(variant.evaluatedTrajectory?.steps?.length ?? 0) > 0 ? (
              <Table className="mt-4">
                <TableHeader>
                  <TableRow>
                    <TrajectoryTableHead label="Step" explanation="Cumulative retrieval step number." />
                    <TrajectoryTableHead label="File" explanation="Cumulative file-level gold-context coverage." />
                    <TrajectoryTableHead label="Block" explanation="Cumulative block-level gold-context coverage." />
                    <TrajectoryTableHead label="Line" explanation="Cumulative line-level gold-context coverage." />
                    <TrajectoryTableHead label="Symbol" explanation="Cumulative symbol-level gold-context coverage." />
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {variant.evaluatedTrajectory?.steps?.map((step) => (
                    <TableRow key={`${variant.label}-coverage-${step.step}`}>
                      <TableCell>{step.step}</TableCell>
                      <TableCell>{formatInstanceMetric(step.coverage.file ?? null)}</TableCell>
                      <TableCell>{formatInstanceMetric(step.coverage.span ?? null)}</TableCell>
                      <TableCell>{formatInstanceMetric(step.coverage.line ?? null)}</TableCell>
                      <TableCell>{formatInstanceMetric(step.coverage.symbol ?? null)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            ) : <p className="mt-4 text-sm text-muted-foreground">No evaluated trajectory coverage data was recorded.</p>}
          </div>
        ))}
      </div>
    </section>
  );
}

function McpUseSection({ variants }: { variants: ComparisonInstanceDetail["variants"] }) {
  const hasMcpData = variants.some((variant) => {
    const usage = variant.mcpUse;
    return (usage?.availableTools?.length ?? 0) > 0 || (usage?.toolCalls ?? 0) > 0 || (usage?.calls?.length ?? 0) > 0;
  });
  if (!hasMcpData) {
    return <StatusPanel message="No MCP tool use data was exported for this instance." />;
  }

  return (
    <section className="space-y-4">
      <SectionTitleWithHelp
        title="MCP Tool Use"
        explanation="Summarizes MCP tools exposed to the run, tool-call counts, returned context, and whether returned paths were later inspected or used."
      />
      <div className={variants.length > 1 ? "grid gap-6 xl:grid-cols-2" : "grid gap-6"}>
        {variants.map((variant) => {
          const usage = variant.mcpUse;
          const calls = usage?.calls ?? [];
          return (
            <div key={variant.label} className="h-full rounded-lg bg-background p-5">
              <h3 className="text-lg font-semibold">{variant.name}</h3>
              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                <SummaryCard label="MCP Calls" value={String(usage?.toolCalls ?? 0)} />
                <SummaryCard label="Successful" value={String(usage?.successfulToolCalls ?? 0)} />
                <SummaryCard label="With Results" value={String(usage?.callsWithResults ?? 0)} />
                <SummaryCard label="Meaningful" value={String(usage?.meaningfulCalls ?? 0)} />
              </div>
              {(usage?.byTool?.length ?? 0) > 0 ? (
                <div className="mt-4 grid gap-2">
                  {usage?.byTool?.map((entry) => (
                    <div key={entry.name} className="flex items-center justify-between gap-3 rounded-md border px-3 py-2 text-sm">
                      <span className="min-w-0 break-all">{entry.name}</span>
                      <span className="font-medium tabular-nums">{entry.calls}</span>
                    </div>
                  ))}
                </div>
              ) : null}
              {calls.length > 0 ? (
                <div className="mt-5 space-y-3">
                  {calls.map((call, index) => (
                    <details key={`${variant.label}-mcp-${index}`} className="rounded-md border p-4">
                      <summary className="cursor-pointer list-none font-medium [overflow-wrap:anywhere]">
                        {call.toolName || "MCP tool"} · {call.resultCount ?? 0} result{call.resultCount === 1 ? "" : "s"}
                        {call.meaningful ? <span className="ml-2 rounded bg-emerald-50 px-2 py-0.5 text-xs text-emerald-800">followed</span> : null}
                      </summary>
                      {call.query ? <div className="mt-3 text-sm text-muted-foreground [overflow-wrap:anywhere]">Query: {call.query}</div> : null}
                      <McpList label="Top Paths" values={call.topPaths} />
                      <McpList label="Final Context Overlap" values={call.overlapFinalContextFiles} />
                      <McpList label="Patch Overlap" values={call.overlapPatchFiles} />
                      <McpList label="Later Inspected" values={call.followedReturnedPaths} />
                    </details>
                  ))}
                </div>
              ) : (
                <p className="mt-4 text-sm text-muted-foreground">
                  No MCP calls were recorded for this variant on this instance.
                </p>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}

function SkillUseSection({
  comparison,
  viewMode,
}: {
  comparison: ComparisonCard;
  viewMode: ComparisonResultsViewMode;
}) {
  const hasSkillData = comparison.variants.some((variant) => {
    const usage = variant.results.skills;
    return (usage?.totalInvocations ?? 0) > 0 || (usage?.byType?.length ?? 0) > 0;
  });
  if (!hasSkillData) {
    return <StatusPanel message="No skill invocation data was exported for this instance." />;
  }

  const comparisonPair = getComparisonPair(comparison);
  if (viewMode === "treatment-delta" && comparisonPair) {
    return <SkillUseVersusSection baseline={comparisonPair.baseline} treatment={comparisonPair.treatment} />;
  }

  return (
    <section className="space-y-4">
      <SectionTitleWithHelp
        title="Skill Usage"
        explanation="Summarizes skill file invocations detected during this instance run, including total count and per-skill breakdown."
      />
      <div className={comparison.variants.length > 1 ? "grid gap-6 xl:grid-cols-2" : "grid gap-6"}>
        {comparison.variants.map((variant) => (
          <div key={variant.label} className="h-full rounded-lg bg-background p-5">
            <h3 className="text-lg font-semibold">{variant.name}</h3>
            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              <SummaryCard label="Skill Invocations" value={formatSkillCount(variant.results.skills?.totalInvocations ?? 0)} />
              <SummaryCard label="Skill Types" value={String(variant.results.skills?.byType?.length ?? 0)} />
            </div>
            <SkillBreakdownList entries={variant.results.skills?.byType ?? []} />
          </div>
        ))}
      </div>
    </section>
  );
}

function SkillUseVersusSection({
  baseline,
  treatment,
}: {
  baseline: ComparisonCard["variants"][number];
  treatment: ComparisonCard["variants"][number];
}) {
  const rows = mergedSkillRows(baseline, treatment);
  return (
    <section className="space-y-4">
      <SectionTitleWithHelp
        title="Skill Usage"
        explanation="Compares skill file invocations detected for this instance. The values show baseline to treatment, not an aggregate benchmark delta."
      />
      <div className="rounded-lg bg-background p-5">
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="rounded-md border p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Skill Invocations</div>
            <SkillVersusValue
              baselineValue={baseline.results.skills?.totalInvocations ?? 0}
              treatmentValue={treatment.results.skills?.totalInvocations ?? 0}
            />
          </div>
          <div className="rounded-md border p-4">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Skill Types</div>
            <SkillVersusValue
              baselineValue={baseline.results.skills?.byType?.length ?? 0}
              treatmentValue={treatment.results.skills?.byType?.length ?? 0}
            />
          </div>
        </div>
        {rows.length > 0 ? (
          <div className="mt-4 divide-y rounded-md border">
            {rows.map((row) => (
              <div key={row.name} className="grid gap-3 px-3 py-2 text-sm sm:grid-cols-[minmax(0,1fr)_auto] sm:items-center">
                <span className="min-w-0 [overflow-wrap:anywhere]">{row.name}</span>
                <SkillVersusValue baselineValue={row.baselineValue} treatmentValue={row.treatmentValue} compact />
              </div>
            ))}
          </div>
        ) : (
          <p className="mt-4 text-sm text-muted-foreground">No per-skill invocation breakdown available.</p>
        )}
      </div>
    </section>
  );
}

function SkillBreakdownList({ entries }: { entries: Array<{ name: string; averagePerRun: number }> }) {
  if (entries.length === 0) {
    return <p className="mt-4 text-sm text-muted-foreground">No per-skill invocation breakdown available.</p>;
  }

  return (
    <div className="mt-4 divide-y rounded-md border">
      {[...entries]
        .sort((left, right) => right.averagePerRun - left.averagePerRun || left.name.localeCompare(right.name))
        .map((entry) => (
          <div key={entry.name} className="flex items-center justify-between gap-3 px-3 py-2 text-sm">
            <span className="min-w-0 [overflow-wrap:anywhere]">{entry.name}</span>
            <span className="font-medium tabular-nums">{formatSkillCount(entry.averagePerRun)}</span>
          </div>
        ))}
    </div>
  );
}

function SkillVersusValue({
  baselineValue,
  treatmentValue,
  compact = false,
}: {
  baselineValue: number;
  treatmentValue: number;
  compact?: boolean;
}) {
  const baselineLabel = formatSkillCount(baselineValue);
  const treatmentLabel = formatSkillCount(treatmentValue);
  return (
    <div className={cn("grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-center gap-2", compact ? "mt-0 min-w-32" : "mt-3")}>
      <span className="truncate text-right font-medium tabular-nums">{baselineLabel}</span>
      <OverviewDirectionSeparator
        direction="neutral"
        baselineValue={baselineLabel}
        treatmentValue={treatmentLabel}
        baselineNumericValue={baselineValue}
        treatmentNumericValue={treatmentValue}
        className={compact ? "h-6 w-6" : undefined}
        iconClassName={compact ? "h-4 w-4" : undefined}
      />
      <span className="truncate font-medium tabular-nums">{treatmentLabel}</span>
    </div>
  );
}

function mergedSkillRows(
  baseline: ComparisonCard["variants"][number],
  treatment: ComparisonCard["variants"][number],
) {
  const baselineByName = new Map((baseline.results.skills?.byType ?? []).map((entry) => [entry.name, entry.averagePerRun]));
  const treatmentByName = new Map((treatment.results.skills?.byType ?? []).map((entry) => [entry.name, entry.averagePerRun]));
  return Array.from(new Set([...baselineByName.keys(), ...treatmentByName.keys()]))
    .map((name) => ({
      name,
      baselineValue: baselineByName.get(name) ?? 0,
      treatmentValue: treatmentByName.get(name) ?? 0,
    }))
    .sort((left, right) => Math.max(right.baselineValue, right.treatmentValue) - Math.max(left.baselineValue, left.treatmentValue) || left.name.localeCompare(right.name));
}

function formatSkillCount(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(2);
}

function McpList({ label, values }: { label: string; values?: string[] }) {
  if (!values || values.length === 0) return null;
  return (
    <div className="mt-3">
      <div className="text-xs uppercase tracking-wide text-muted-foreground">{label}</div>
      <ul className="mt-1 space-y-1 text-sm">
        {values.map((value) => (
          <li key={value} className="break-all rounded bg-muted/20 px-2 py-1 font-mono text-xs">{value}</li>
        ))}
      </ul>
    </div>
  );
}

function TrajectoryVersusSection({
  baseline,
  treatment,
}: {
  baseline: ComparisonInstanceDetail["variants"][number];
  treatment: ComparisonInstanceDetail["variants"][number];
}) {
  const rows = mergedTrajectorySteps(baseline, treatment);
  return (
    <section className="space-y-4">
      <SectionTitleWithHelp title="Cumulative Evaluated Trajectory" explanation={trajectoryExplanation} />
      <div className="rounded-lg bg-background p-5">
        {rows.length > 0 ? (
          <Table className="min-w-[42rem] table-fixed">
            <TableHeader>
              <TableRow>
                <TrajectoryMajorTableHead label="Step" explanation="Cumulative retrieval step number." className="w-16 border-l-0" />
                <TrajectoryMajorTableHead label="File" explanation="Cumulative file-level gold-context coverage." />
                <TrajectoryMajorTableHead label="Block" explanation="Cumulative block-level gold-context coverage." />
                <TrajectoryMajorTableHead label="Line" explanation="Cumulative line-level gold-context coverage." />
                <TrajectoryMajorTableHead label="Symbol" explanation="Cumulative symbol-level gold-context coverage." />
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((row) => (
                <TableRow key={`trajectory-versus-${row.step}`}>
                  <TableCell className="w-16 px-3 py-3">{row.step}</TableCell>
                  <TrajectoryMajorTableCell><CoverageVersusValue baselineValue={row.baseline?.coverage.file} treatmentValue={row.treatment?.coverage.file} /></TrajectoryMajorTableCell>
                  <TrajectoryMajorTableCell><CoverageVersusValue baselineValue={row.baseline?.coverage.span} treatmentValue={row.treatment?.coverage.span} /></TrajectoryMajorTableCell>
                  <TrajectoryMajorTableCell><CoverageVersusValue baselineValue={row.baseline?.coverage.line} treatmentValue={row.treatment?.coverage.line} /></TrajectoryMajorTableCell>
                  <TrajectoryMajorTableCell><CoverageVersusValue baselineValue={row.baseline?.coverage.symbol} treatmentValue={row.treatment?.coverage.symbol} /></TrajectoryMajorTableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        ) : (
          <p className="text-sm text-muted-foreground">No evaluated trajectory coverage data was recorded.</p>
        )}
      </div>
    </section>
  );
}

function TrajectoryMajorTableHead({
  label,
  explanation,
  className,
}: {
  label: string;
  explanation: string;
  className?: string;
}) {
  return (
    <TableHead className={cn("border-l px-3 text-center", className)}>
      <div className="flex items-center justify-center gap-2">
        <span>{label}</span>
        <HelpIcon label={label} explanation={explanation} />
      </div>
    </TableHead>
  );
}

function TrajectoryMajorTableCell({ children }: { children: ReactNode }) {
  return (
    <TableCell className="border-l px-3 py-3 text-center">
      {children}
    </TableCell>
  );
}

function mergedTrajectorySteps(
  baseline: ComparisonInstanceDetail["variants"][number],
  treatment: ComparisonInstanceDetail["variants"][number],
) {
  const baselineSteps = new Map((baseline.evaluatedTrajectory?.steps ?? []).map((step) => [step.step, step]));
  const treatmentSteps = new Map((treatment.evaluatedTrajectory?.steps ?? []).map((step) => [step.step, step]));
  return Array.from(new Set([...baselineSteps.keys(), ...treatmentSteps.keys()]))
    .sort((left, right) => left - right)
    .map((step) => ({
      step,
      baseline: baselineSteps.get(step),
      treatment: treatmentSteps.get(step),
    }));
}

function CoverageVersusValue({
  baselineValue,
  treatmentValue,
}: {
  baselineValue: number | null | undefined;
  treatmentValue: number | null | undefined;
}) {
  return (
    <div className="inline-grid grid-cols-[max-content_auto_max-content] items-center gap-1.5">
      <span className="whitespace-nowrap text-xs tabular-nums">{formatInstanceMetric(baselineValue ?? null)}</span>
      <OverviewDirectionSeparator
        direction="higher"
        baselineValue={formatInstanceMetric(baselineValue ?? null)}
        treatmentValue={formatInstanceMetric(treatmentValue ?? null)}
        baselineNumericValue={baselineValue ?? null}
        treatmentNumericValue={treatmentValue ?? null}
        className="h-6 w-5"
        iconClassName="h-4 w-4"
      />
      <span className="whitespace-nowrap text-xs tabular-nums">{formatInstanceMetric(treatmentValue ?? null)}</span>
    </div>
  );
}
