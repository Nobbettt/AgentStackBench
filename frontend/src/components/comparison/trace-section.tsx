// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
import { AlertCircle, CheckSquare, FilePenLine, MessageSquareText, Search, Terminal } from "lucide-react";

import type { ComparisonInstanceDetail } from "@/data/comparisons";
import { HelpIcon } from "@/components/comparison/shared";
import { cn } from "@/lib/utils";

type TraceEntry = NonNullable<ComparisonInstanceDetail["variants"][number]["traceEntries"]>[number];
type TraceFilter = "all" | TraceEntry["kind"] | "failures";

const traceFilters: Array<{ id: TraceFilter; label: string }> = [
  { id: "all", label: "All" },
  { id: "command_execution", label: "Commands" },
  { id: "assistant_message", label: "Assistant" },
  { id: "file_change", label: "File Changes" },
  { id: "todo_list", label: "Todos" },
  { id: "failures", label: "Failures" },
];

export function TraceSection({ variants }: { variants: ComparisonInstanceDetail["variants"] }) {
  const [activeFilter, setActiveFilter] = useState<TraceFilter>("all");
  const [searchQuery, setSearchQuery] = useState("");
  const maxTraceLength = Math.max(0, ...variants.map((variant) => variant.traceEntries?.length ?? 0));
  const visibleRows = Array.from({ length: maxTraceLength }, (_entry, index) => ({
    index,
    entries: variants.map((variant) => variant.traceEntries?.[index]),
  })).filter((row) =>
    row.entries.some((entry) => entry && traceEntryMatches(entry, activeFilter, searchQuery)),
  );

  return (
    <section className="space-y-4">
      <SectionTitleWithHelp
        title="Trace"
        explanation="Index-aligned trace of command executions, assistant messages, file changes, and todo events exported for each variant."
      />
      <TraceSummary variants={variants} />
      <TraceControls
        activeFilter={activeFilter}
        searchQuery={searchQuery}
        onFilterChange={setActiveFilter}
        onSearchChange={setSearchQuery}
      />
      {maxTraceLength === 0 ? (
        <section className="rounded-lg bg-background p-6 text-sm text-muted-foreground">
          No structured conversation or reasoning trace was exported for this instance.
        </section>
      ) : visibleRows.length === 0 ? (
        <section className="rounded-lg bg-background p-6 text-sm text-muted-foreground">
          No trace events match the current filter.
        </section>
      ) : (
        <TraceTimeline variants={variants} rows={visibleRows} activeFilter={activeFilter} searchQuery={searchQuery} />
      )}
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

function TraceSummary({ variants }: { variants: ComparisonInstanceDetail["variants"] }) {
  return (
    <div className={variants.length > 1 ? "grid gap-4 lg:grid-cols-2" : "grid gap-4"}>
      {variants.map((variant) => {
        const stats = traceStats(variant.traceEntries ?? []);
        return (
          <div key={variant.label} className="rounded-lg bg-background p-5">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h3 className="text-lg font-semibold">{variant.name}</h3>
                <p className="mt-1 text-sm text-muted-foreground">{stats.total} trace events</p>
              </div>
              {stats.failures > 0 ? (
                <span className="inline-flex items-center gap-1.5 rounded-full border border-rose-200 bg-rose-50 px-2 py-1 text-xs font-medium text-rose-700">
                  <AlertCircle className="h-3.5 w-3.5" />
                  {stats.failures} failed
                </span>
              ) : null}
            </div>
            <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
              <TraceStatCard label="Commands" value={stats.commands} />
              <TraceStatCard label="Assistant" value={stats.assistant} />
              <TraceStatCard label="Files" value={stats.fileChanges} />
              <TraceStatCard label="Todos" value={stats.todos} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function TraceStatCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-md border p-3">
      <div className="text-xs uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-2 text-xl font-semibold tabular-nums">{value}</div>
    </div>
  );
}

function TraceControls({
  activeFilter,
  searchQuery,
  onFilterChange,
  onSearchChange,
}: {
  activeFilter: TraceFilter;
  searchQuery: string;
  onFilterChange: (filter: TraceFilter) => void;
  onSearchChange: (query: string) => void;
}) {
  return (
    <div className="flex flex-col gap-3 rounded-lg bg-background p-4 lg:flex-row lg:items-center lg:justify-between">
      <div className="flex flex-wrap gap-2">
        {traceFilters.map((filter) => (
          <button
            key={filter.id}
            type="button"
            className={cn(
              "rounded-md border px-3 py-1.5 text-sm font-medium transition-colors",
              activeFilter === filter.id
                ? "border-primary bg-primary text-primary-foreground"
                : "border-border bg-background text-muted-foreground hover:text-foreground",
            )}
            onClick={() => onFilterChange(filter.id)}
          >
            {filter.label}
          </button>
        ))}
      </div>
      <label className="relative block min-w-0 lg:w-80">
        <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <input
          value={searchQuery}
          onChange={(event) => onSearchChange(event.target.value)}
          placeholder="Search trace"
          className="h-10 w-full rounded-md border bg-background pl-9 pr-3 text-sm outline-none transition-colors placeholder:text-muted-foreground focus:border-primary"
        />
      </label>
    </div>
  );
}

function TraceTimeline({
  variants,
  rows,
  activeFilter,
  searchQuery,
}: {
  variants: ComparisonInstanceDetail["variants"];
  rows: Array<{ index: number; entries: Array<TraceEntry | undefined> }>;
  activeFilter: TraceFilter;
  searchQuery: string;
}) {
  return (
    <div className="rounded-lg bg-background p-5">
      <div className="grid gap-3" style={{ gridTemplateColumns: `4rem repeat(${variants.length}, minmax(0, 1fr))` }}>
        <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Step</div>
        {variants.map((variant) => (
          <div key={variant.label} className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{variant.name}</div>
        ))}
        {rows.map((row) => (
          <TraceTimelineRow
            key={`trace-row-${row.index}`}
            row={row}
            variants={variants}
            activeFilter={activeFilter}
            searchQuery={searchQuery}
          />
        ))}
      </div>
    </div>
  );
}

function TraceTimelineRow({
  row,
  variants,
  activeFilter,
  searchQuery,
}: {
  row: { index: number; entries: Array<TraceEntry | undefined> };
  variants: ComparisonInstanceDetail["variants"];
  activeFilter: TraceFilter;
  searchQuery: string;
}) {
  return (
    <>
      <div className="flex min-h-16 items-start pt-4 text-sm font-medium tabular-nums text-muted-foreground">
        {row.index + 1}
      </div>
      {variants.map((variant, variantIndex) => {
        const entry = row.entries[variantIndex];
        const visible = entry ? traceEntryMatches(entry, activeFilter, searchQuery) : false;
        return (
          <div key={`${variant.label}-trace-row-${row.index}`} className="min-w-0 border-t pt-3">
            {entry && visible ? (
              <TraceEventCard entry={entry} index={row.index} />
            ) : entry ? (
              <div className="rounded-md border border-dashed p-4 text-sm text-muted-foreground">Filtered</div>
            ) : (
              <div className="rounded-md border border-dashed p-4 text-sm text-muted-foreground">No event</div>
            )}
          </div>
        );
      })}
    </>
  );
}

function TraceEventCard({ entry, index }: { entry: TraceEntry; index: number }) {
  const body = traceEntryBody(entry);
  const failure = traceEntryIsFailure(entry);

  return (
    <details className={cn("group rounded-md border p-4", failure ? "border-rose-200 bg-rose-50/50" : "bg-background")}>
      <summary className="cursor-pointer list-none">
        <div className="flex min-w-0 items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <TraceKindBadge entry={entry} />
              {entry.status ? <span className="text-xs text-muted-foreground">{entry.status}</span> : null}
              {typeof entry.exitCode === "number" ? <span className="text-xs text-muted-foreground">exit {entry.exitCode}</span> : null}
            </div>
            <div className="mt-2 line-clamp-2 break-words text-sm font-medium">{traceEntryTitle(entry)}</div>
          </div>
          <span className="shrink-0 text-xs tabular-nums text-muted-foreground">#{index + 1}</span>
        </div>
      </summary>
      {body ? (
        <pre className="mt-4 max-h-96 overflow-auto rounded-md bg-muted/20 p-3 text-xs leading-6">
          {body}
        </pre>
      ) : (
        <p className="mt-4 text-sm text-muted-foreground">No event body recorded.</p>
      )}
    </details>
  );
}

function TraceKindBadge({ entry }: { entry: TraceEntry }) {
  const Icon = traceKindIcon(entry.kind);
  const failure = traceEntryIsFailure(entry);
  return (
    <span className={cn(
      "inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs font-medium",
      failure ? "border-rose-200 bg-rose-50 text-rose-700" : "border-border bg-muted/40 text-muted-foreground",
    )}>
      <Icon className="h-3.5 w-3.5" />
      {traceKindLabel(entry.kind)}
    </span>
  );
}

function traceKindIcon(kind: TraceEntry["kind"]) {
  if (kind === "command_execution") return Terminal;
  if (kind === "assistant_message") return MessageSquareText;
  if (kind === "file_change") return FilePenLine;
  return CheckSquare;
}

function traceStats(entries: TraceEntry[]) {
  return {
    total: entries.length,
    commands: entries.filter((entry) => entry.kind === "command_execution").length,
    assistant: entries.filter((entry) => entry.kind === "assistant_message").length,
    fileChanges: entries.filter((entry) => entry.kind === "file_change").length,
    todos: entries.filter((entry) => entry.kind === "todo_list").length,
    failures: entries.filter(traceEntryIsFailure).length,
  };
}

function traceKindLabel(kind: TraceEntry["kind"]): string {
  if (kind === "command_execution") return "Command";
  if (kind === "assistant_message") return "Assistant";
  if (kind === "file_change") return "File Change";
  return "Todo";
}

function traceEntryTitle(entry: TraceEntry): string {
  if (entry.command) return entry.command;
  const text = entry.text ?? entry.output ?? payloadText(entry.payload);
  return firstNonEmptyLine(text) || traceKindLabel(entry.kind);
}

function traceEntryBody(entry: TraceEntry): string {
  return entry.text ?? entry.output ?? payloadText(entry.payload);
}

function payloadText(payload: Record<string, unknown> | undefined): string {
  return payload ? JSON.stringify(payload, null, 2) : "";
}

function firstNonEmptyLine(value: string): string {
  return value.split(/\r?\n/).map((line) => line.trim()).find(Boolean) ?? "";
}

function traceEntryIsFailure(entry: TraceEntry): boolean {
  return entry.status === "failed" || (typeof entry.exitCode === "number" && entry.exitCode !== 0);
}

function traceEntryMatches(entry: TraceEntry, filter: TraceFilter, searchQuery: string): boolean {
  if (filter === "failures" && !traceEntryIsFailure(entry)) return false;
  if (filter !== "all" && filter !== "failures" && entry.kind !== filter) return false;

  const normalizedQuery = searchQuery.trim().toLowerCase();
  if (!normalizedQuery) return true;

  return [
    entry.kind,
    entry.status ?? "",
    entry.command ?? "",
    entry.text ?? "",
    entry.output ?? "",
    payloadText(entry.payload),
  ].some((value) => value.toLowerCase().includes(normalizedQuery));
}
