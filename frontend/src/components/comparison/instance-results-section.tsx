
import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { Check, ChevronDown, ChevronRight, Eye, Filter } from "lucide-react";

import type { ComparisonCard } from "@/data/comparisons";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { buildInstanceComparison, buildInstanceRows } from "@/components/comparison/instance-data";
import { ComparisonMetricSections } from "@/components/comparison/metric-sections";
import { formatLanguageLabel, formatResolutionStatus, resolutionStatusClassName, sortBench } from "@/components/comparison/format";
import type { ComparisonResultsViewMode, DeltaDisplayMode } from "@/components/comparison/types";
import { cn } from "@/lib/utils";

const INSTANCE_PAGE_SIZE = 20;

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

export function InstanceResultsSection({
  comparison,
  viewMode,
  deltaDisplayMode,
}: {
  comparison: ComparisonCard;
  viewMode: ComparisonResultsViewMode;
  deltaDisplayMode: DeltaDisplayMode;
}) {
  const rows = useMemo(() => buildInstanceRows(comparison), [comparison]);
  const availableBenches = useMemo(() => Array.from(new Set(rows.map((row) => row.bench))).sort(sortBench), [rows]);
  const availableLanguages = useMemo(() => Array.from(new Set(rows.map((row) => row.language))).sort(), [rows]);
  const [selectedBenches, setSelectedBenches] = useState<string[]>(availableBenches);
  const [selectedLanguages, setSelectedLanguages] = useState<string[]>(availableLanguages);
  const [page, setPage] = useState(1);
  const [expandedRowId, setExpandedRowId] = useState<string | null>(null);
  const filteredRows = useMemo(
    () => rows.filter((row) => selectedBenches.includes(row.bench) && selectedLanguages.includes(row.language)),
    [rows, selectedBenches, selectedLanguages],
  );
  const pageCount = Math.max(1, Math.ceil(filteredRows.length / INSTANCE_PAGE_SIZE));
  const pageStart = (page - 1) * INSTANCE_PAGE_SIZE;
  const visibleRows = filteredRows.slice(pageStart, pageStart + INSTANCE_PAGE_SIZE);

  useEffect(() => {
    setSelectedBenches(availableBenches);
    setSelectedLanguages(availableLanguages);
    setPage(1);
    setExpandedRowId(null);
  }, [comparison.id, availableBenches, availableLanguages]);

  useEffect(() => {
    setPage(1);
    setExpandedRowId((currentId) => currentId && filteredRows.some((row) => row.instanceId === currentId) ? currentId : null);
  }, [filteredRows, selectedBenches, selectedLanguages]);

  const comparisonPair = comparison.variants.length >= 2 ? { baseline: comparison.variants[0], treatment: comparison.variants[1] } : null;
  const columnCount = comparisonPair ? 6 : 5;

  return (
    <section className="space-y-4">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <h2 className="text-xl font-semibold tracking-tight">Issue Results</h2>
        <div className="text-sm text-muted-foreground">
          Showing {filteredRows.length === 0 ? 0 : pageStart + 1}-{Math.min(pageStart + INSTANCE_PAGE_SIZE, filteredRows.length)} of {filteredRows.length}
        </div>
      </div>
      <div className="rounded-lg border bg-background">
        <Table className="min-h-[24rem]">
          <TableHeader>
            <TableRow>
              <TableHead>Instance</TableHead>
              <TableHead>
                <InlineHeaderFilter label="Dataset" ariaLabel="Filter issue results by dataset" values={selectedBenches} onChange={setSelectedBenches} options={availableBenches.map((bench) => ({ value: bench, label: bench }))} />
              </TableHead>
              <TableHead>
                <InlineHeaderFilter label="Language" ariaLabel="Filter issue results by language" values={selectedLanguages} onChange={setSelectedLanguages} options={availableLanguages.map((language) => ({ value: language, label: formatLanguageLabel(language) }))} />
              </TableHead>
              <TableHead>{comparisonPair ? comparisonPair.baseline.name : comparison.variants[0]?.name}<div className="text-[11px] font-normal text-muted-foreground">Pass@1</div></TableHead>
              {comparisonPair ? <TableHead>{comparisonPair.treatment.name}<div className="text-[11px] font-normal text-muted-foreground">Pass@1</div></TableHead> : null}
              <TableHead className="w-[6rem]">Open</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {visibleRows.map((row) => {
              const isExpanded = expandedRowId === row.instanceId;
              const instanceComparison = buildInstanceComparison(comparison, row);
              return (
                <Fragment key={row.instanceId}>
                  <TableRow key={row.instanceId}>
                    <TableCell>
                      <div className="flex items-start gap-3">
                        <button type="button" aria-expanded={isExpanded} onClick={() => setExpandedRowId(isExpanded ? null : row.instanceId)} className="mt-0.5 inline-flex h-6 w-6 items-center justify-center rounded-md border text-muted-foreground">
                          {isExpanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                        </button>
                        <div>
                          <div className="font-medium">{row.instanceId}</div>
                          {row.originalInstanceId ? <div className="mt-1 text-xs text-muted-foreground">{row.originalInstanceId}</div> : null}
                        </div>
                      </div>
                    </TableCell>
                    <TableCell>{row.bench}</TableCell>
                    <TableCell>{formatLanguageLabel(row.language)}</TableCell>
                    <TableCell className={cn("font-medium", resolutionStatusClassName(row.baseline?.artifacts?.resolutionStatus))}>{formatResolutionStatus(row.baseline?.artifacts?.resolutionStatus)}</TableCell>
                    {comparisonPair ? <TableCell className={cn("font-medium", resolutionStatusClassName(row.treatment?.artifacts?.resolutionStatus))}>{formatResolutionStatus(row.treatment?.artifacts?.resolutionStatus)}</TableCell> : null}
                    <TableCell>
                      <Button variant="outline" size="icon" className="h-8 w-8" aria-label={`View details for ${row.instanceId}`} onClick={() => { window.location.hash = `#/comparisons/${comparison.id}/instances/${encodeURIComponent(row.instanceId)}`; }}>
                        <Eye className="h-4 w-4" />
                      </Button>
                    </TableCell>
                  </TableRow>
                  {isExpanded ? (
                    <TableRow key={`${row.instanceId}-expanded`}>
                      <TableCell colSpan={columnCount} className="bg-muted/20 p-0">
                        <div className="space-y-6 p-6">
                          <ComparisonMetricSections comparison={instanceComparison} viewMode={viewMode} deltaDisplayMode={deltaDisplayMode} showExecutionMetrics={false} />
                        </div>
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
