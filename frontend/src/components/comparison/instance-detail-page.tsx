
import { useMemo, useState } from "react";
import { Percent } from "lucide-react";

import type { ComparisonCard, ComparisonInstanceDetail } from "@/data/comparisons";
import { Table, TableBody, TableCell, TableHeader, TableRow } from "@/components/ui/table";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { TooltipProvider } from "@/components/ui/tooltip";
import { DetailSection, TrajectoryTableHead } from "@/components/comparison/detail-section";
import { buildInstanceComparison, buildInstanceRows } from "@/components/comparison/instance-data";
import { MarkdownText } from "@/components/comparison/markdown-text";
import { ComparisonMetricSections, PatchOverlapBetweenVariantsSection } from "@/components/comparison/metric-sections";
import {
  formatDurationMs,
  formatInstanceMetric,
  formatResolutionStatus,
  resolutionStatusClassName,
} from "@/components/comparison/format";
import type { ComparisonResultsViewMode, DeltaDisplayMode } from "@/components/comparison/types";
import { cn } from "@/lib/utils";

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
  const [deltaDisplayMode, setDeltaDisplayMode] = useState<DeltaDisplayMode>("absolute");

  if (!row || !instanceComparison) {
    return <section className="rounded-lg border bg-background p-6 text-sm text-muted-foreground">Instance detail not found in the current comparison snapshot.</section>;
  }

  return (
    <TooltipProvider>
      <div className="space-y-6">
        <InstanceHeader row={row} instanceComparison={instanceComparison} />
        <DetailControls
          viewMode={viewMode}
          deltaDisplayMode={deltaDisplayMode}
          onViewModeChange={setViewMode}
          onDeltaDisplayModeChange={setDeltaDisplayMode}
        />
        <ComparisonMetricSections comparison={instanceComparison} viewMode={viewMode} deltaDisplayMode={deltaDisplayMode} />
        {detail ? <PatchOverlapBetweenVariantsSection overlap={detail.fixOverlapBetweenVariants} collapsible /> : null}
        {renderDetailContent(detail, detailError)}
      </div>
    </TooltipProvider>
  );
}

function renderDetailContent(detail: ComparisonInstanceDetail | null | undefined, detailError?: string | null) {
  if (detail) {
    return <DetailedRunSections detail={detail} />;
  }
  if (detailError) {
    return (
      <section className="rounded-lg border bg-background p-6 text-sm text-rose-700">
        Unable to load detailed trajectory and trace data: {detailError}
      </section>
    );
  }
  if (detail === undefined) {
    return (
      <section className="rounded-lg border bg-background p-6 text-sm text-muted-foreground">
        Loading detailed trajectory and trace data...
      </section>
    );
  }
  return (
    <section className="rounded-lg border bg-background p-6 text-sm text-muted-foreground">
      Detailed trajectory and trace data is not available for this instance.
    </section>
  );
}

function InstanceHeader({
  row,
  instanceComparison,
}: {
  row: ReturnType<typeof buildInstanceRows>[number];
  instanceComparison: ComparisonCard;
}) {
  return (
    <section className="rounded-lg bg-background p-6">
      <h1 className="text-3xl font-semibold tracking-tight">{row.instanceId}</h1>
      {row.originalInstanceId ? <p className="mt-2 text-sm text-muted-foreground">{row.originalInstanceId}</p> : null}
      <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <SummaryCard label="Dataset" value={row.bench} />
        <SummaryCard label="Language" value={row.language} />
        <SummaryCard
          label={`${instanceComparison.variants[0].name} Pass@1`}
          value={formatResolutionStatus(row.baseline?.artifacts?.resolutionStatus)}
          className={resolutionStatusClassName(row.baseline?.artifacts?.resolutionStatus)}
        />
        {instanceComparison.variants[1] ? (
          <SummaryCard
            label={`${instanceComparison.variants[1].name} Pass@1`}
            value={formatResolutionStatus(row.treatment?.artifacts?.resolutionStatus)}
            className={resolutionStatusClassName(row.treatment?.artifacts?.resolutionStatus)}
          />
        ) : null}
      </div>
    </section>
  );
}

function SummaryCard({ label, value, className }: { label: string; value: string; className?: string }) {
  return (
    <div className="rounded-md border p-4">
      <div className="text-xs uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className={cn("mt-2 text-sm font-medium", className)}>{value}</div>
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
    <section className="rounded-lg border bg-background px-4 py-4">
      <div className="space-y-3">
        <div className="text-xs uppercase tracking-wide text-muted-foreground">Detail Controls</div>
        <div className="flex justify-center">
          <ToggleGroup type="single" variant="outline" value={viewMode} onValueChange={(value) => value && onViewModeChange(value as ComparisonResultsViewMode)}>
            <ToggleGroupItem value="treatment-delta">Treatment Delta</ToggleGroupItem>
            <ToggleGroupItem value="side-by-side">Side by Side</ToggleGroupItem>
          </ToggleGroup>
        </div>
        {viewMode === "treatment-delta" ? (
          <div className="flex justify-center">
            <ToggleGroup type="single" variant="outline" value={deltaDisplayMode} onValueChange={(value) => value && onDeltaDisplayModeChange(value as DeltaDisplayMode)}>
              <ToggleGroupItem value="absolute" aria-label="Numerical Diff"><span className="font-semibold tabular-nums">1.2→</span></ToggleGroupItem>
              <ToggleGroupItem value="percent" aria-label="Percent Diff"><Percent className="h-4 w-4" /></ToggleGroupItem>
            </ToggleGroup>
          </div>
        ) : null}
      </div>
    </section>
  );
}

function DetailedRunSections({ detail }: { detail: ComparisonInstanceDetail }) {
  return (
    <div className="space-y-6">
      <DetailSection title="Run Detail" variants={detail.variants} render={(variant) => (
        <div className="h-full rounded-lg border bg-background p-6">
          <h3 className="text-lg font-semibold">{variant.name}</h3>
          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            <SummaryCard label="Model" value={variant.model ?? "—"} />
            <SummaryCard label="Effort" value={variant.effort ?? "—"} />
            <SummaryCard label="Duration" value={typeof variant.durationMs === "number" ? formatDurationMs(variant.durationMs) : "—"} />
            <SummaryCard label="Status" value={variant.status ?? "—"} />
          </div>
        </div>
      )} />
      <DetailSection title="Final Answer" variants={detail.variants} render={(variant) => (
        <MarkdownTextPanel title={variant.name} text={variant.finalOutput?.finalAnswer || "No final answer recorded."} />
      )} />
      <TrajectorySection detail={detail} />
      {detail.variants.some((variant) => variant.modelPatch) ? (
        <DetailSection title="Model Patch" variants={detail.variants} render={(variant) => (
          <DiffCodePanel title={variant.name} text={variant.modelPatch || "No model patch recorded for this run."} />
        )} />
      ) : null}
      <DetailSection title="Reasoning & Conversation Trace" variants={detail.variants} render={(variant) => (
        <TracePanel variant={variant} />
      )} />
    </div>
  );
}

function TrajectorySection({ detail }: { detail: ComparisonInstanceDetail }) {
  return (
    <DetailSection title="Cumulative Evaluated Trajectory" variants={detail.variants} render={(variant) => (
      <div className="h-full rounded-lg border bg-background p-6">
        <h3 className="text-lg font-semibold">{variant.name}</h3>
        {(variant.evaluatedTrajectory?.steps?.length ?? 0) > 0 ? (
          <Table className="mt-4">
            <TableHeader>
              <TableRow>
                <TrajectoryTableHead label="Step" explanation="Cumulative retrieval step number." />
                <TrajectoryTableHead label="File" explanation="Cumulative file-level gold-context coverage." />
                <TrajectoryTableHead label="Symbol" explanation="Cumulative symbol-level gold-context coverage." />
                <TrajectoryTableHead label="Span" explanation="Cumulative span-level gold-context coverage." />
                <TrajectoryTableHead label="Line" explanation="Cumulative line-level gold-context coverage." />
              </TableRow>
            </TableHeader>
            <TableBody>
              {variant.evaluatedTrajectory?.steps?.map((step) => (
                <TableRow key={`${variant.label}-coverage-${step.step}`}>
                  <TableCell>{step.step}</TableCell>
                  <TableCell>{formatInstanceMetric(step.coverage.file ?? null)}</TableCell>
                  <TableCell>{formatInstanceMetric(step.coverage.symbol ?? null)}</TableCell>
                  <TableCell>{formatInstanceMetric(step.coverage.span ?? null)}</TableCell>
                  <TableCell>{formatInstanceMetric(step.coverage.line ?? null)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        ) : <p className="mt-4 text-sm text-muted-foreground">No evaluated trajectory coverage data was recorded.</p>}
      </div>
    )} />
  );
}

function TextPanel({ title, text }: { title: string; text: string }) {
  return (
    <div className="h-full rounded-lg border bg-background p-6">
      <h3 className="text-lg font-semibold">{title}</h3>
      <div className="mt-4 whitespace-pre-wrap text-sm leading-6">{text}</div>
    </div>
  );
}

function MarkdownTextPanel({ title, text }: { title: string; text: string }) {
  return (
    <div className="h-full rounded-lg border bg-background p-6">
      <h3 className="text-lg font-semibold">{title}</h3>
      <MarkdownText text={text} className="mt-4" />
    </div>
  );
}

function DiffCodePanel({ title, text }: { title: string; text: string }) {
  const lines = text.replace(/\r\n?/g, "\n").split("\n");

  return (
    <div className="h-full rounded-lg border bg-background p-6">
      <h3 className="text-lg font-semibold">{title}</h3>
      <pre className="mt-4 max-h-[28rem] overflow-auto rounded-md border bg-muted/20 py-3 text-xs leading-6">
        <code>
          {lines.map((line, index) => (
            <span
              key={`${title}-patch-line-${index}`}
              className={cn("block min-h-6 whitespace-pre px-4", diffLineClassName(line))}
            >
              {line || " "}
            </span>
          ))}
        </code>
      </pre>
    </div>
  );
}

function diffLineClassName(line: string): string {
  if (line.startsWith("diff --git ") || line.startsWith("index ")) {
    return "bg-muted/50 font-semibold text-muted-foreground";
  }
  if (line.startsWith("@@")) {
    return "border-l-2 border-sky-500 bg-sky-50 text-sky-900";
  }
  if (line.startsWith("+++") || line.startsWith("---")) {
    return "bg-muted/40 font-medium text-muted-foreground";
  }
  if (line.startsWith("+")) {
    return "border-l-2 border-emerald-500 bg-emerald-50 text-emerald-950";
  }
  if (line.startsWith("-")) {
    return "border-l-2 border-rose-500 bg-rose-50 text-rose-950";
  }
  if (line.startsWith("\\ No newline at end of file")) {
    return "text-muted-foreground";
  }
  return "";
}

function TracePanel({ variant }: { variant: ComparisonInstanceDetail["variants"][number] }) {
  if ((variant.traceEntries?.length ?? 0) === 0) {
    return <TextPanel title={variant.name} text="No structured conversation or reasoning trace was exported for this run." />;
  }
  return (
    <div className="h-full rounded-lg border bg-background p-6">
      <h3 className="text-lg font-semibold">{variant.name}</h3>
      <div className="mt-4 space-y-3">
        {variant.traceEntries?.map((entry, index) => (
          <details key={`${variant.label}-trace-${index}`} className="rounded-md border p-4">
            <summary className="cursor-pointer list-none font-medium">{entry.command || entry.kind.replace("_", " ")}</summary>
            <pre className="mt-3 max-h-80 overflow-auto rounded-md bg-muted/20 p-3 text-xs leading-6">
              {entry.text || entry.output || (entry.payload ? JSON.stringify(entry.payload, null, 2) : "")}
            </pre>
          </details>
        ))}
      </div>
    </div>
  );
}
