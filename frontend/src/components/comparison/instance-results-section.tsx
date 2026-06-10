
import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { Check, ChevronDown, ChevronRight, Eye, Filter } from "lucide-react";

import type { ComparisonCard, ComparisonInstance } from "@/data/comparisons";
import { Button, buttonVariants } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { buildInstanceRows, instanceContextF1 } from "@/components/comparison/instance-data";
import { compactInstanceName } from "@/components/comparison/instance-naming";
import {
  coveragePrecision,
  f1,
  formatDurationMs,
  formatLanguageLabel,
  formatMetric,
  formatPatternMetric,
  formatResolutionStatus,
  formatTokens,
  resolutionStatusClassName,
} from "@/components/comparison/format";
import type { ComparisonResultsViewMode, DeltaDisplayMode, InstanceRow } from "@/components/comparison/types";
import { cn } from "@/lib/utils";

const INSTANCE_PAGE_SIZE = 20;
const RESOLUTION_STATUS_ORDER = ["resolved", "unresolved", "error", "missing"];

type SortKey = "contextF1" | "fileF1" | "blockF1" | "steps";
type SortDirection = "asc" | "desc";
type SortState = {
  key: SortKey;
  direction: SortDirection;
} | null;

function resolutionStatusValue(instance: ComparisonInstance | undefined): string {
  return (instance?.artifacts?.resolutionStatus ?? "missing").trim().toLowerCase() || "missing";
}

function sortResolutionStatuses(left: string, right: string): number {
  const leftIndex = RESOLUTION_STATUS_ORDER.indexOf(left);
  const rightIndex = RESOLUTION_STATUS_ORDER.indexOf(right);
  if (leftIndex >= 0 || rightIndex >= 0) {
    return (leftIndex >= 0 ? leftIndex : Number.MAX_SAFE_INTEGER) - (rightIndex >= 0 ? rightIndex : Number.MAX_SAFE_INTEGER);
  }
  return left.localeCompare(right);
}

function resolutionFilterOptions(statuses: string[]): Array<{ value: string; label: string }> {
  return statuses.map((status) => ({ value: status, label: formatResolutionStatus(status) }));
}

function InlineHeaderFilter({
  label,
  ariaLabel,
  values,
  options,
  onChange,
}: {
  label: string;
  ariaLabel: string;
  values: string[];
  options: Array<{ value: string; label: string }>;
  onChange: (nextValue: string[]) => void;
}) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const allValues = options.map((option) => option.value);
  const allSelected = values.length === allValues.length;

  useEffect(() => {
    function handlePointerDown(event: MouseEvent) {
      if (!containerRef.current || containerRef.current.contains(event.target as Node)) return;
      setOpen(false);
    }
    document.addEventListener("mousedown", handlePointerDown);
    return () => document.removeEventListener("mousedown", handlePointerDown);
  }, []);

  return (
    <div ref={containerRef} className="relative inline-flex items-center gap-1">
      <span>{label}</span>
      <button
        type="button"
        aria-label={ariaLabel}
        aria-expanded={open}
        aria-haspopup="menu"
        onClick={() => setOpen((currentOpen) => !currentOpen)}
        className={cn(
          "inline-flex h-7 w-7 items-center justify-center rounded-md border transition-colors",
          allSelected ? "border-transparent text-muted-foreground hover:bg-accent/60" : "border-primary/30 bg-accent/60 text-foreground",
        )}
      >
        <Filter className="h-3.5 w-3.5" />
      </button>
      {open ? (
        <div role="menu" className="absolute left-0 top-full z-20 mt-2 min-w-40 rounded-md border bg-background p-1 shadow-md">
          <FilterOption label="All" selected={allSelected} onClick={() => onChange(allValues)} />
          {options.map((option) => {
            const selected = values.includes(option.value);
            return (
              <FilterOption
                key={option.value}
                label={option.label}
                selected={selected}
                onClick={() => {
                  const nextValues = selected ? values.filter((value) => value !== option.value) : [...values, option.value];
                  onChange(nextValues.length > 0 ? nextValues : allValues);
                }}
              />
            );
          })}
        </div>
      ) : null}
    </div>
  );
}

function FilterOption({ label, selected, onClick }: { label: string; selected: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      role="menuitemcheckbox"
      aria-checked={selected}
      onClick={onClick}
      className={cn(
        "flex w-full items-center justify-between gap-3 rounded-sm px-3 py-2 text-left text-xs transition-colors",
        selected ? "bg-accent text-foreground" : "text-muted-foreground hover:bg-accent/60 hover:text-foreground",
      )}
    >
      <span>{label}</span>
      {selected ? <Check className="h-3.5 w-3.5" /> : <span className="h-3.5 w-3.5" />}
    </button>
  );
}

function SortableHeader({
  label,
  sortKey,
  sort,
  onSort,
}: {
  label: string;
  sortKey: SortKey;
  sort: SortState;
  onSort: (sortKey: SortKey) => void;
}) {
  const active = sort?.key === sortKey;
  const ariaSort = active ? (sort.direction === "asc" ? "ascending" : "descending") : "none";

  return (
    <TableHead className="text-right" aria-sort={ariaSort}>
      <button
        type="button"
        onClick={() => onSort(sortKey)}
        className="ml-auto inline-flex items-center gap-1 rounded-sm text-right transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        <span>{label}</span>
        <ChevronDown
          className={cn(
            "h-3.5 w-3.5 transition-transform",
            active ? "opacity-100" : "opacity-35",
            active && sort.direction === "asc" ? "rotate-180" : "",
          )}
        />
      </button>
    </TableHead>
  );
}

export function InstanceResultsSection({
  comparison,
  deltaDisplayMode,
}: {
  comparison: ComparisonCard;
  viewMode: ComparisonResultsViewMode;
  deltaDisplayMode: DeltaDisplayMode;
}) {
  const rows = useMemo(() => buildInstanceRows(comparison), [comparison]);
  const comparisonPair = useMemo(
    () => (comparison.variants.length >= 2 ? { baseline: comparison.variants[0], treatment: comparison.variants[1] } : null),
    [comparison.variants],
  );
  const availableBaselineStatuses = useMemo(
    () => Array.from(new Set(rows.map((row) => resolutionStatusValue(row.baseline)))).sort(sortResolutionStatuses),
    [rows],
  );
  const availableTreatmentStatuses = useMemo(
    () => Array.from(new Set(rows.map((row) => resolutionStatusValue(row.treatment)))).sort(sortResolutionStatuses),
    [rows],
  );
  const [selectedBaselineStatuses, setSelectedBaselineStatuses] = useState<string[]>(availableBaselineStatuses);
  const [selectedTreatmentStatuses, setSelectedTreatmentStatuses] = useState<string[]>(availableTreatmentStatuses);
  const [sort, setSort] = useState<SortState>(null);
  const [page, setPage] = useState(1);
  const [expandedRowId, setExpandedRowId] = useState<string | null>(null);
  const filteredRows = useMemo(
    () =>
      rows.filter(
        (row) =>
          selectedBaselineStatuses.includes(resolutionStatusValue(row.baseline)) &&
          (!comparisonPair || selectedTreatmentStatuses.includes(resolutionStatusValue(row.treatment))),
      ),
    [comparisonPair, rows, selectedBaselineStatuses, selectedTreatmentStatuses],
  );
  const sortedRows = useMemo(
    () => (sort ? [...filteredRows].sort((left, right) => compareRowsBySortKey(left, right, sort, Boolean(comparisonPair))) : filteredRows),
    [comparisonPair, filteredRows, sort],
  );
  const pageCount = Math.max(1, Math.ceil(sortedRows.length / INSTANCE_PAGE_SIZE));
  const pageStart = (page - 1) * INSTANCE_PAGE_SIZE;
  const visibleRows = sortedRows.slice(pageStart, pageStart + INSTANCE_PAGE_SIZE);

  useEffect(() => {
    setSelectedBaselineStatuses(availableBaselineStatuses);
    setSelectedTreatmentStatuses(availableTreatmentStatuses);
    setPage(1);
    setExpandedRowId(null);
  }, [comparison.id, availableBaselineStatuses, availableTreatmentStatuses]);

  useEffect(() => {
    setPage(1);
    setExpandedRowId((currentId) => currentId && filteredRows.some((row) => row.instanceId === currentId) ? currentId : null);
  }, [filteredRows]);

  const tableViewMode: ComparisonResultsViewMode = "side-by-side";
  const columnCount = comparisonPair ? 8 : 7;
  const handleSort = (sortKey: SortKey) => {
    setSort((currentSort) => {
      if (currentSort?.key === sortKey) {
        return { key: sortKey, direction: currentSort.direction === "asc" ? "desc" : "asc" };
      }
      return { key: sortKey, direction: sortKey === "steps" ? "asc" : "desc" };
    });
  };

  return (
    <section className="space-y-4">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <h2 className="text-xl font-semibold tracking-tight">Issue Results</h2>
        <div className="text-sm text-muted-foreground">
          Showing {sortedRows.length === 0 ? 0 : pageStart + 1}-{Math.min(pageStart + INSTANCE_PAGE_SIZE, sortedRows.length)} of {sortedRows.length}
        </div>
      </div>
      <div className="overflow-x-auto rounded-lg bg-background">
        <Table className="min-h-[24rem] min-w-[72rem]">
          <TableHeader>
            <TableRow>
              <TableHead>Instance</TableHead>
              <TableHead>
                <InlineHeaderFilter
                  label={comparisonPair ? comparisonPair.baseline.name : comparison.variants[0]?.name}
                  ariaLabel="Filter issue results by baseline resolution status"
                  values={selectedBaselineStatuses}
                  onChange={setSelectedBaselineStatuses}
                  options={resolutionFilterOptions(availableBaselineStatuses)}
                />
                <div className="text-[11px] font-normal text-muted-foreground">Pass@1</div>
              </TableHead>
              {comparisonPair ? (
                <TableHead>
                  <InlineHeaderFilter
                    label={comparisonPair.treatment.name}
                    ariaLabel="Filter issue results by treatment resolution status"
                    values={selectedTreatmentStatuses}
                    onChange={setSelectedTreatmentStatuses}
                    options={resolutionFilterOptions(availableTreatmentStatuses)}
                  />
                  <div className="text-[11px] font-normal text-muted-foreground">Pass@1</div>
                </TableHead>
              ) : null}
              <SortableHeader label="Context F1" sortKey="contextF1" sort={sort} onSort={handleSort} />
              <SortableHeader label="File F1" sortKey="fileF1" sort={sort} onSort={handleSort} />
              <SortableHeader label="Block F1" sortKey="blockF1" sort={sort} onSort={handleSort} />
              <SortableHeader label="Steps" sortKey="steps" sort={sort} onSort={handleSort} />
              <TableHead className="w-[6rem]">Open</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {visibleRows.map((row) => {
              const isExpanded = expandedRowId === row.instanceId;
              const name = compactInstanceName(row);
              return (
                <Fragment key={row.instanceId}>
                  <TableRow key={row.instanceId}>
                    <TableCell>
                      <div className="flex items-start gap-3">
                        <button type="button" aria-expanded={isExpanded} onClick={() => setExpandedRowId(isExpanded ? null : row.instanceId)} className="mt-0.5 inline-flex h-6 w-6 items-center justify-center rounded-md border text-muted-foreground">
                          {isExpanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                        </button>
                        <div className="min-w-0" title={name.fullId}>
                          <div className="font-mono text-sm font-semibold tabular-nums text-foreground">{name.shortId}</div>
                          <div className="mt-1 truncate text-sm text-muted-foreground">{name.repo}</div>
                          <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[11px] uppercase tracking-wide text-muted-foreground/80">
                            <span>{row.bench}</span>
                            <span aria-hidden="true">/</span>
                            <span>{formatLanguageLabel(row.language)}</span>
                            {name.taskType ? (
                              <>
                                <span aria-hidden="true">/</span>
                                <span>{name.taskType}</span>
                              </>
                            ) : null}
                          </div>
                        </div>
                      </div>
                    </TableCell>
                    <TableCell className={cn("font-medium", resolutionStatusClassName(resolutionStatusValue(row.baseline)))}>{formatResolutionStatus(resolutionStatusValue(row.baseline))}</TableCell>
                    {comparisonPair ? <TableCell className={cn("font-medium", resolutionStatusClassName(resolutionStatusValue(row.treatment)))}>{formatResolutionStatus(resolutionStatusValue(row.treatment))}</TableCell> : null}
                    <SortableMetricCell value={rowMetricValue(row, "contextF1", Boolean(comparisonPair))} metricKey="contextF1" />
                    <SortableMetricCell value={rowMetricValue(row, "fileF1", Boolean(comparisonPair))} metricKey="fileF1" />
                    <SortableMetricCell value={rowMetricValue(row, "blockF1", Boolean(comparisonPair))} metricKey="blockF1" />
                    <SortableMetricCell value={rowMetricValue(row, "steps", Boolean(comparisonPair))} metricKey="steps" />
                    <TableCell>
                      <a
                        href={`#/comparisons/${comparison.id}/instances/${encodeURIComponent(row.instanceId)}`}
                        target="_blank"
                        rel="noreferrer"
                        className={cn(buttonVariants({ variant: "outline", size: "icon" }), "h-8 w-8")}
                        aria-label={`View details for ${row.instanceId}`}
                      >
                        <Eye className="h-4 w-4" />
                      </a>
                    </TableCell>
                  </TableRow>
                  {isExpanded ? (
                    <TableRow key={`${row.instanceId}-expanded`}>
                      <TableCell colSpan={columnCount} className="bg-muted/20 p-0">
                        <CompactInstanceExpansion
                          comparison={comparison}
                          row={row}
                          comparisonPair={comparisonPair}
                          viewMode={tableViewMode}
                          deltaDisplayMode={deltaDisplayMode}
                        />
                      </TableCell>
                    </TableRow>
                  ) : null}
                </Fragment>
              );
            })}
          </TableBody>
        </Table>
      </div>
      <div className="flex items-center justify-end gap-2">
        <Button variant="outline" onClick={() => setPage((currentPage) => Math.max(1, currentPage - 1))} disabled={page <= 1}>Previous</Button>
        <div className="text-sm text-muted-foreground">Page {page} of {pageCount}</div>
        <Button variant="outline" onClick={() => setPage((currentPage) => Math.min(pageCount, currentPage + 1))} disabled={page >= pageCount}>Next</Button>
      </div>
    </section>
  );
}

function SortableMetricCell({ value, metricKey }: { value: number | null; metricKey: SortKey }) {
  return <TableCell className="text-right font-mono text-sm tabular-nums">{formatSortableMetricValue(value, metricKey)}</TableCell>;
}

function compareRowsBySortKey(left: InstanceRow, right: InstanceRow, sort: Exclude<SortState, null>, preferTreatment: boolean): number {
  const leftValue = rowMetricValue(left, sort.key, preferTreatment);
  const rightValue = rowMetricValue(right, sort.key, preferTreatment);
  if (leftValue === null && rightValue === null) return left.instanceId.localeCompare(right.instanceId);
  if (leftValue === null) return 1;
  if (rightValue === null) return -1;
  const delta = leftValue - rightValue;
  if (delta === 0) return left.instanceId.localeCompare(right.instanceId);
  return sort.direction === "asc" ? delta : -delta;
}

function rowMetricValue(row: InstanceRow, sortKey: SortKey, preferTreatment: boolean): number | null {
  const instance = preferTreatment ? row.treatment ?? row.baseline : row.baseline;
  if (!instance) return null;
  if (sortKey === "contextF1") return instanceContextF1(instance);
  if (sortKey === "fileF1") return instanceLevelF1(instance, "file");
  if (sortKey === "blockF1") return instanceLevelF1(instance, "span");
  return typeof instance.trajectory.steps === "number" && Number.isFinite(instance.trajectory.steps) ? instance.trajectory.steps : null;
}

function instanceLevelF1(instance: ComparisonInstance, key: "file" | "span"): number | null {
  if (instance.artifacts && instance.artifacts.evaluationStatus !== "valid") return null;
  const metric = instance.quality[key];
  const values = coveragePrecision(metric.predSize, metric.goldSize, metric.intersection);
  return f1(values.coverage, values.precision);
}

function formatSortableMetricValue(value: number | null, metricKey: SortKey): string {
  if (value === null) return "-";
  return metricKey === "steps" ? formatPatternMetric(value) : formatMetric(value);
}

function CompactInstanceExpansion({
  comparison,
  row,
  comparisonPair,
  viewMode,
  deltaDisplayMode,
}: {
  comparison: ComparisonCard;
  row: ReturnType<typeof buildInstanceRows>[number];
  comparisonPair: { baseline: ComparisonCard["variants"][number]; treatment: ComparisonCard["variants"][number] } | null;
  viewMode: ComparisonResultsViewMode;
  deltaDisplayMode: DeltaDisplayMode;
}) {
  const showTreatmentDelta = viewMode === "treatment-delta" && comparisonPair;
  const variants = showTreatmentDelta
    ? [{ variant: comparisonPair.treatment, instance: row.treatment, baselineInstance: row.baseline }]
    : comparisonPair
      ? [
          { variant: comparisonPair.baseline, instance: row.baseline, baselineInstance: undefined },
          { variant: comparisonPair.treatment, instance: row.treatment, baselineInstance: undefined },
        ]
      : [{ variant: comparison.variants[0], instance: row.baseline, baselineInstance: undefined }];

  return (
    <div className="space-y-3 p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="text-xs text-muted-foreground">
          {row.bench} / {formatLanguageLabel(row.language)}
          {row.originalInstanceId ? <span> / {row.originalInstanceId}</span> : null}
        </div>
        <Button
          variant="outline"
          className="h-8 gap-2"
          onClick={() => { window.location.hash = `#/comparisons/${comparison.id}/instances/${encodeURIComponent(row.instanceId)}`; }}
        >
          <Eye className="h-3.5 w-3.5" />
          Full detail
        </Button>
      </div>
      <div className={showTreatmentDelta ? "grid gap-2" : "grid gap-2 lg:grid-cols-2"}>
        {variants.map(({ variant, instance, baselineInstance }) => (
          <CompactVariantSummary
            key={variant.label}
            name={variant.name}
            instance={instance}
            baselineInstance={baselineInstance}
            deltaDisplayMode={deltaDisplayMode}
          />
        ))}
      </div>
    </div>
  );
}

function CompactVariantSummary({
  name,
  instance,
  baselineInstance,
  deltaDisplayMode,
}: {
  name: string;
  instance?: ComparisonInstance;
  baselineInstance?: ComparisonInstance;
  deltaDisplayMode: DeltaDisplayMode;
}) {
  const contextF1 = instanceContextF1(instance);
  const baselineContextF1 = instanceContextF1(baselineInstance);
  const status = instance?.artifacts?.resolutionStatus;
  return (
    <div className="rounded-md border bg-background px-3 py-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="min-w-0 truncate text-sm font-medium">{name}</div>
        <div className="flex items-center gap-2">
          <div className={cn("text-xs font-medium", resolutionStatusClassName(status))}>{formatResolutionStatus(status)}</div>
          {baselineInstance ? <BaselineNote value={`Baseline: ${formatResolutionStatus(baselineInstance.artifacts?.resolutionStatus)}`} /> : null}
        </div>
      </div>
      <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs sm:grid-cols-3">
        <CompactStat
          label="Context F1"
          value={contextF1 === null ? "-" : formatMetric(contextF1)}
          delta={formatNumericDelta(contextF1, baselineContextF1, deltaDisplayMode, "higher", (value) => value.toFixed(3))}
        />
        <CompactStat
          label="Steps"
          value={formatNullableNumber(instance?.trajectory.steps)}
          delta={formatNumericDelta(instance?.trajectory.steps, baselineInstance?.trajectory.steps, deltaDisplayMode, "lower", formatStepDeltaValue)}
        />
        <CompactStat
          label="Runtime"
          value={formatNullableDuration(instance?.resources.durationMs)}
          delta={formatNumericDelta(instance?.resources.durationMs, baselineInstance?.resources.durationMs, deltaDisplayMode, "lower", formatDurationDeltaValue)}
        />
        <CompactStat
          label="Tokens"
          value={formatNullableTokens(instance?.resources.totalTokens)}
          delta={formatNumericDelta(instance?.resources.totalTokens, baselineInstance?.resources.totalTokens, deltaDisplayMode, "lower", formatTokenDeltaValue)}
        />
        <CompactStat
          label="Patch"
          value={instance ? (instance.artifacts?.hasModelPatch ? "Yes" : "No") : "-"}
          note={baselineInstance ? `Baseline: ${baselineInstance.artifacts?.hasModelPatch ? "Yes" : "No"}` : undefined}
        />
        <CompactStat
          label="Eval"
          value={formatEvaluationStatus(instance?.artifacts?.evaluationStatus)}
          note={baselineInstance ? `Baseline: ${formatEvaluationStatus(baselineInstance.artifacts?.evaluationStatus)}` : undefined}
        />
      </dl>
    </div>
  );
}

type CompactDelta = {
  label: string;
  tone: "success" | "danger" | "neutral";
};

function CompactStat({ label, value, delta, note }: { label: string; value: string; delta?: CompactDelta; note?: string }) {
  return (
    <div className="min-w-0">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="flex min-w-0 items-baseline gap-2">
        <span className="truncate font-medium text-foreground">{value}</span>
        {delta ? <span className={cn("shrink-0 text-[11px] font-medium", compactDeltaClassName(delta.tone))}>{delta.label}</span> : null}
      </dd>
      {!delta && note ? <dd className="truncate text-[11px] text-muted-foreground">{note}</dd> : null}
    </div>
  );
}

function BaselineNote({ value }: { value: string }) {
  return <span className="text-[11px] font-normal text-muted-foreground">{value}</span>;
}

function formatNullableNumber(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  return Number.isInteger(value) ? value.toFixed(0) : value.toFixed(1);
}

function formatNullableDuration(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? formatDurationMs(value) : "-";
}

function formatNullableTokens(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? formatTokens(value) : "-";
}

function formatEvaluationStatus(status: string | undefined): string {
  if (!status) return "-";
  return status.charAt(0).toUpperCase() + status.slice(1);
}

function formatNumericDelta(
  value: number | null | undefined,
  baseline: number | null | undefined,
  displayMode: DeltaDisplayMode,
  direction: "higher" | "lower",
  formatAbsoluteValue: (value: number) => string,
): CompactDelta | undefined {
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    typeof baseline !== "number" ||
    !Number.isFinite(baseline)
  ) {
    return undefined;
  }

  const delta = value - baseline;
  const tone = compactDeltaTone(delta, direction);
  if (displayMode === "percent") {
    if (baseline === 0) return undefined;
    return { label: `${formatSignedNumber((delta / Math.abs(baseline)) * 100, 1)}%`, tone };
  }
  return { label: formatSignedDelta(delta, formatAbsoluteValue), tone };
}

function compactDeltaTone(delta: number, direction: "higher" | "lower"): CompactDelta["tone"] {
  if (delta === 0) return "neutral";
  const improved = direction === "higher" ? delta > 0 : delta < 0;
  return improved ? "success" : "danger";
}

function compactDeltaClassName(tone: CompactDelta["tone"]): string {
  if (tone === "success") return "text-emerald-700";
  if (tone === "danger") return "text-rose-700";
  return "text-muted-foreground";
}

function formatSignedDelta(value: number, formatAbsoluteValue: (value: number) => string): string {
  if (value === 0) return formatAbsoluteValue(0);
  return `${value > 0 ? "+" : "-"}${formatAbsoluteValue(Math.abs(value))}`;
}

function formatSignedNumber(value: number, decimals: number): string {
  if (value === 0) return value.toFixed(decimals);
  return `${value > 0 ? "+" : "-"}${Math.abs(value).toFixed(decimals)}`;
}

function formatStepDeltaValue(value: number): string {
  return Number.isInteger(value) ? value.toFixed(0) : value.toFixed(1);
}

function formatDurationDeltaValue(value: number): string {
  return formatDurationMs(value);
}

function formatTokenDeltaValue(value: number): string {
  return formatTokens(value);
}
