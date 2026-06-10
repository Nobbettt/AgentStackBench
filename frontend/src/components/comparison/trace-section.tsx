// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
import { AlertCircle, CheckSquare, FilePenLine, MessageSquareText, Search, Terminal, Wrench } from "lucide-react";

import type { ComparisonInstanceDetail } from "@/data/comparisons";
import { HelpIcon } from "@/components/comparison/shared";
import { cn } from "@/lib/utils";

type TraceEntry = NonNullable<ComparisonInstanceDetail["variants"][number]["traceEntries"]>[number];
type TraceFilter = "all" | TraceEntry["kind"] | "failures";

const traceFilters: Array<{ id: TraceFilter; label: string }> = [
  { id: "all", label: "All" },
  { id: "command_execution", label: "Commands" },
  { id: "tool_use", label: "Tools" },
  { id: "tool_result", label: "Tool Results" },
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
        explanation="Index-aligned trace of command executions, tool calls/results, assistant messages, file changes, and todo events exported for each variant."
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
  const fileChangeItems = entry.kind === "file_change" ? fileChangeItemsFromEntry(entry) : [];

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
      {fileChangeItems.length > 0 ? (
        <FileChangeBody changes={fileChangeItems} />
      ) : body ? (
        <pre className="mt-4 max-h-96 overflow-auto rounded-md bg-muted/20 p-3 text-xs leading-6">
          {body}
        </pre>
      ) : (
        <p className="mt-4 text-sm text-muted-foreground">{traceEntryEmptyBodyLabel(entry)}</p>
      )}
    </details>
  );
}

function FileChangeBody({ changes }: { changes: Array<{ path: string; kind?: string; description?: string }> }) {
  return (
    <div className="mt-4 divide-y rounded-md border bg-muted/10">
      {changes.map((change, index) => {
        const action = humanizeSkillName(change.kind ?? "changed");
        return (
          <div key={`${change.path}-${index}`} className="grid gap-2 px-3 py-2 text-sm sm:grid-cols-[7rem_minmax(0,1fr)] sm:items-start">
            <span className="inline-flex w-fit items-center rounded border bg-background px-2 py-0.5 text-xs font-medium text-muted-foreground">
              {action}
            </span>
            <div className="min-w-0">
              <div className="break-all font-mono text-xs leading-5">{change.path}</div>
              {change.description ? <div className="mt-1 text-xs text-muted-foreground">{change.description}</div> : null}
            </div>
          </div>
        );
      })}
    </div>
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
  if (kind === "tool_use" || kind === "tool_result") return Wrench;
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
  if (kind === "tool_use") return "Tool";
  if (kind === "tool_result") return "Tool Result";
  if (kind === "assistant_message") return "Assistant";
  if (kind === "file_change") return "File Change";
  return "Todo";
}

function traceEntryTitle(entry: TraceEntry): string {
  if (entry.command) return commandTraceTitle(entry) ?? compactCommand(entry.command);
  if (entry.kind === "todo_list") return todoTraceTitle(entry) ?? traceKindLabel(entry.kind);
  if (entry.kind === "file_change") return fileChangeTraceTitle(entry);
  const text = entry.text ?? entry.output ?? payloadText(entry.payload);
  return firstNonEmptyLine(text) || traceKindLabel(entry.kind);
}

function traceEntryBody(entry: TraceEntry): string {
  if (entry.kind === "file_change" && !entry.text && !entry.output && (payloadIsEmpty(entry.payload) || fileChangeItemsFromEntry(entry).length > 0)) return "";
  return entry.text ?? entry.output ?? payloadText(entry.payload);
}

function payloadText(payload: Record<string, unknown> | undefined): string {
  return payload ? JSON.stringify(payload, null, 2) : "";
}

function payloadIsEmpty(payload: Record<string, unknown> | undefined): boolean {
  return !payload || Object.keys(payload).length === 0;
}

function traceEntryEmptyBodyLabel(entry: TraceEntry): string {
  if (entry.kind === "file_change") return "No file-change metadata was exported for this event.";
  return "No event body recorded.";
}

function firstNonEmptyLine(value: string): string {
  return value.split(/\r?\n/).map((line) => line.trim()).find(Boolean) ?? "";
}

function todoTraceTitle(entry: TraceEntry): string | null {
  const items = todoItemsFromEntry(entry);
  if (items.length === 0) return null;

  const completedCount = items.filter((item) => item.completed).length;
  const activeItem = items.find((item) => !item.completed);
  if (activeItem) {
    return `Todo: ${trimLongText(activeItem.text, 90)} (${completedCount}/${items.length} done)`;
  }

  return `Todo list complete (${completedCount}/${items.length} done)`;
}

function todoItemsFromEntry(entry: TraceEntry): Array<{ text: string; completed: boolean }> {
  const payloadItems = todoItemsFromValue(entry.payload);
  if (payloadItems.length > 0) return payloadItems;

  const rawText = entry.text ?? entry.output;
  if (!rawText) return [];

  try {
    return todoItemsFromValue(JSON.parse(rawText) as unknown);
  } catch {
    return [];
  }
}

function todoItemsFromValue(value: unknown): Array<{ text: string; completed: boolean }> {
  if (!isRecord(value)) return [];
  const items = value.items;
  if (!Array.isArray(items)) return [];

  return items
    .map((item) => {
      if (!isRecord(item) || typeof item.text !== "string") return null;
      return {
        text: item.text,
        completed: Boolean(item.completed) || item.status === "completed",
      };
    })
    .filter((item): item is { text: string; completed: boolean } => item !== null);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function fileChangeTraceTitle(entry: TraceEntry): string {
  const payload = entry.payload;
  if (!payload || payloadIsEmpty(payload)) return "File changed (details unavailable)";

  const changes = fileChangeItemsFromEntry(entry);
  if (changes.length > 1) {
    const verbs = Array.from(new Set(changes.map((change) => humanizeSkillName(change.kind ?? "changed"))));
    const action = verbs.length === 1 ? verbs[0] : "Changed";
    return `${action} ${changes.length} files`;
  }
  if (changes.length === 1) {
    const action = humanizeSkillName(changes[0].kind ?? "changed");
    return `File ${action.toLowerCase()}: ${changes[0].path}`;
  }

  const description = stringPayloadValue(payload, "description");
  if (description) return trimLongText(description, 100);

  const path = stringPayloadValue(payload, "path") ?? stringPayloadValue(payload, "file") ?? stringPayloadValue(payload, "source_path");
  const changeType = stringPayloadValue(payload, "change_type") ?? stringPayloadValue(payload, "kind");
  const action = changeType ? humanizeSkillName(changeType) : "Changed";
  if (path) return `File ${action.toLowerCase()}: ${path}`;

  return "File changed";
}

function fileChangeItemsFromEntry(entry: TraceEntry): Array<{ path: string; kind?: string; description?: string }> {
  const payload = entry.payload;
  if (!payload) return [];

  const changes = fileChangeItemsFromPayload(payload);
  if (changes.length > 0) return changes;

  const path = stringPayloadValue(payload, "path") ?? stringPayloadValue(payload, "file") ?? stringPayloadValue(payload, "source_path");
  if (!path) return [];

  const kind = stringPayloadValue(payload, "change_type") ?? stringPayloadValue(payload, "kind") ?? undefined;
  const description = stringPayloadValue(payload, "description") ?? undefined;
  return [{ path, kind, description }];
}

function fileChangeItemsFromPayload(payload: Record<string, unknown>): Array<{ path: string; kind?: string; description?: string }> {
  const rawChanges = payload.changes;
  if (!Array.isArray(rawChanges)) return [];
  return rawChanges
    .map((change): { path: string; kind?: string; description?: string } | null => {
      if (!isRecord(change)) return null;
      const path = stringPayloadValue(change, "path");
      if (!path) return null;
      const kind = stringPayloadValue(change, "change_type") ?? stringPayloadValue(change, "kind");
      const description = stringPayloadValue(change, "description");
      return {
        path,
        ...(kind ? { kind } : {}),
        ...(description ? { description } : {}),
      };
    })
    .filter((change): change is { path: string; kind?: string; description?: string } => change !== null);
}

function stringPayloadValue(payload: Record<string, unknown>, key: string): string | null {
  const value = payload[key];
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function commandTraceTitle(entry: TraceEntry): string | null {
  if (!entry.command) return null;

  const command = unwrapShellCommand(entry.command);
  const body = traceEntryBody(entry);
  const skillName = skillNameFromTraceBody(body);
  if (skillName && (command.includes("<agent-runtime>") || command.includes("SKILL.md"))) {
    return `Read skill: ${skillName}`;
  }

  const fileRead = fileReadFromCommand(command);
  if (fileRead) {
    const action = fileRead.numbered ? "Inspect numbered lines" : "Read file";
    return `${action}: ${fileRead.path}${fileRead.range ? ` (${fileRead.range})` : ""}`;
  }

  const searchPattern = searchPatternFromCommand(command);
  if (searchPattern) return `Search code for: ${searchPattern}`;

  const formatterTargets = formatterTargetsFromCommand(command);
  if (formatterTargets) return `Format files: ${formatterTargets}`;

  const testCommand = testTitleFromCommand(command);
  if (testCommand) return testCommand;

  if (/\bgit\s+status\b/.test(command)) return "Check git status";
  if (/\bgit\s+diff\s+--check\b/.test(command)) return "Check diff for whitespace errors";
  if (/\bgit\s+diff\b/.test(command)) return "Review git diff";
  if (/\bgit\s+show\b/.test(command)) return "Inspect git commit";
  if (/\b(?:python3?|[^ ]*\/python)\s+-m\s+pip\s+install\b|\b(?:pip|[^ ]*\/pip)\s+install\b/.test(command)) return "Install Python dependencies";
  if (/\b(?:python3?|[^ ]*\/python)\s+-m\s+venv\b/.test(command)) return "Create Python virtual environment";
  if (/\b(?:npm|yarn|pnpm)\s+(?:install|ci)\b/.test(command)) return "Install JavaScript dependencies";
  if (/\b(?:npm|yarn|pnpm)\s+run\s+build\b/.test(command)) return "Run frontend build";
  if (/\b(?:npm|yarn|pnpm)\s+run\b/.test(command)) return `Run script: ${scriptNameFromCommand(command) ?? compactCommand(command)}`;
  if (/^pwd\b/.test(command)) return "Check working directory";
  if (/^ls\b|\s+ls\b/.test(command)) return "List files";
  if (/\bwhich\s+\S+|\bcommand\s+-v\s+\S+/.test(command)) return "Check tool availability";
  if (/\brm\s+-rf\s+\.venv\b/.test(command)) return "Remove Python virtual environment";
  if (/\bcat\b/.test(command)) {
    const path = command.match(/\bcat\s+(.+)$/)?.[1]?.trim();
    return path ? `Read file: ${trimShellOperators(path)}` : "Read file";
  }

  return null;
}

function unwrapShellCommand(command: string): string {
  const trimmed = command.trim();
  const shellMatch = trimmed.match(/^\/bin\/(?:ba|z)?sh\s+-lc\s+([\s\S]+)$/);
  if (!shellMatch) return trimmed;

  const shellArgument = shellMatch[1].trim();
  const quote = shellArgument[0];
  if ((quote === "'" || quote === "\"") && shellArgument.endsWith(quote)) {
    return shellArgument.slice(1, -1);
  }
  return shellArgument;
}

function skillNameFromTraceBody(body: string): string | null {
  const frontmatterName = body.match(/^---\s*\n[\s\S]*?\bname:\s*([^\n]+)\n[\s\S]*?\n---/);
  if (frontmatterName) return humanizeSkillName(frontmatterName[1]);

  const heading = body.match(/^#\s+(.+)$/m);
  return heading ? heading[1].trim() : null;
}

function humanizeSkillName(value: string): string {
  const normalized = value.trim().replace(/^["']|["']$/g, "");
  if (!normalized.includes("-") && /[A-Z]/.test(normalized)) return normalized;
  return normalized
    .split(/[-_\s]+/)
    .filter(Boolean)
    .map((word) => word[0]?.toUpperCase() + word.slice(1))
    .join(" ");
}

function fileReadFromCommand(command: string): { path: string; range?: string; numbered?: boolean } | null {
  const sedMatch = command.match(/\bsed\s+-n\s+['"]?(\d+,\d+)p['"]?\s+([^;&|]+)/);
  if (sedMatch) {
    return { path: trimShellOperators(sedMatch[2]), range: sedMatch[1].replace(",", "-") };
  }

  const numberedMatch = command.match(/\bnl\s+-ba\s+([^;&|]+)\s*\|\s*sed\s+-n\s+['"]?(\d+,\d+)p['"]?/);
  if (numberedMatch) {
    return { path: trimShellOperators(numberedMatch[1]), range: numberedMatch[2].replace(",", "-"), numbered: true };
  }

  const headMatch = command.match(/\bhead\s+(?:-\d+\s+|-n\s+\d+\s+)?([^;&|]+)/);
  if (headMatch) return { path: trimShellOperators(headMatch[1]) };

  const tailMatch = command.match(/\btail\s+(?:-\d+\s+|-n\s+\d+\s+)?([^;&|]+)/);
  if (tailMatch) return { path: trimShellOperators(tailMatch[1]) };

  return null;
}

function searchPatternFromCommand(command: string): string | null {
  const rgMatch = command.match(/\brg(?:\s+-[^\s]+)*\s+((?:"(?:\\.|[^"])+")|(?:'(?:\\.|[^'])+')|[^\s|;&]+)/);
  if (!rgMatch) return null;
  return unquoteShellValue(rgMatch[1]);
}

function formatterTargetsFromCommand(command: string): string | null {
  const gofmtMatch = command.match(/\bgofmt\s+-w\s+(.+)$/);
  if (gofmtMatch) return trimLongText(gofmtMatch[1].trim(), 80);

  const prettierMatch = command.match(/\bprettier\s+(?:--write|-w)\s+(.+)$/);
  if (prettierMatch) return trimLongText(prettierMatch[1].trim(), 80);

  const blackMatch = command.match(/\bblack\s+(.+)$/);
  if (blackMatch) return trimLongText(blackMatch[1].trim(), 80);

  return null;
}

function testTitleFromCommand(command: string): string | null {
  if (/\bgo\s+test\b/.test(command)) return `Run Go tests: ${testTargetFromCommand(command, "go test")}`;
  if (/\bpytest\b/.test(command)) return `Run pytest: ${testTargetFromCommand(command, "pytest")}`;
  if (/\btests\/runtests\.py\b/.test(command)) return `Run Django tests: ${testTargetAfter(command, "tests/runtests.py")}`;
  if (/\b(?:npm|yarn|pnpm)\s+(?:test|run\s+test)\b/.test(command)) return "Run JavaScript tests";
  if (/\bcargo\s+test\b/.test(command)) return "Run Rust tests";
  if (/\bmvn\s+test\b/.test(command)) return "Run Maven tests";
  return null;
}

function testTargetFromCommand(command: string, marker: string): string {
  return trimLongText(command.slice(command.indexOf(marker) + marker.length).trim() || "all", 80);
}

function testTargetAfter(command: string, marker: string): string {
  return trimLongText(command.slice(command.indexOf(marker) + marker.length).trim() || "all", 80);
}

function scriptNameFromCommand(command: string): string | null {
  return command.match(/\b(?:npm|yarn|pnpm)\s+run\s+([^\s]+)/)?.[1] ?? null;
}

function compactCommand(command: string): string {
  return trimLongText(unwrapShellCommand(command), 110);
}

function trimShellOperators(value: string): string {
  return unquoteShellValue(value.trim().replace(/\s*(?:&&|\|\||[;|]).*$/, ""));
}

function unquoteShellValue(value: string): string {
  return value.trim().replace(/^["']|["']$/g, "").replace(/\\"/g, "\"").replace(/\\'/g, "'");
}

function trimLongText(value: string, maxLength: number): string {
  return value.length <= maxLength ? value : `${value.slice(0, maxLength - 1)}…`;
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
