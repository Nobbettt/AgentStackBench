
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, Columns2, Percent, TrendingUpDown } from "lucide-react";

import { ComparisonResults } from "@/components/comparison-results";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Button } from "@/components/ui/button";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { buildFilteredComparison, comparisonHasInstanceData, getAvailableBenches, getAvailableLanguages } from "@/data/comparison-aggregation";
import type { ComparisonCard } from "@/data/comparisons";
import { formatLanguageLabel } from "@/components/comparison/format";
import type { ComparisonResultsViewMode, DeltaDisplayMode } from "@/components/comparison/types";

function formatAgentName(agent: ComparisonCard["agent"]): string {
  return agent === "codex" ? "Codex" : "Claude Code";
}

function getComparisonModels(comparison: ComparisonCard): string[] {
  return Array.from(new Set(
    comparison.variants
      .map((variant) => variant.model ?? variant.parameters.find((parameter) => parameter.label.toLowerCase() === "model")?.value)
      .filter((value): value is string => Boolean(value?.trim())),
  ));
}

function formatComparisonRunDate(comparison: ComparisonCard): string | null {
  const timestamp = comparison.completedAt ?? comparison.startedAt;
  if (!timestamp) return null;
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return null;
  return new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" }).format(date);
}

function getDatasetSliceSummary(comparison: ComparisonCard): string | null {
  const benchCounts = comparison.taskSet?.benchCounts;
  if (!benchCounts || Object.keys(benchCounts).length === 0) return null;
  return Object.entries(benchCounts)
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .map(([bench, count]) => `${count} ${bench}`)
    .join(" / ");
}

function getComparisonPair(comparison: ComparisonCard) {
  if (comparison.variants.length < 2) return null;
  return { baseline: comparison.variants[0], treatment: comparison.variants[1] };
}

export function ComparisonPage({ comparison }: { comparison: ComparisonCard }) {
  const hasInstanceFilters = comparisonHasInstanceData(comparison);
  const availableLanguages = useMemo(() => (hasInstanceFilters ? getAvailableLanguages(comparison) : []), [comparison, hasInstanceFilters]);
  const availableBenches = useMemo(() => (hasInstanceFilters ? getAvailableBenches(comparison) : []), [comparison, hasInstanceFilters]);
  const [selectedLanguages, setSelectedLanguages] = useState<string[]>(availableLanguages);
  const [selectedBenches, setSelectedBenches] = useState<string[]>(availableBenches);
  const [viewMode, setViewMode] = useState<ComparisonResultsViewMode>("treatment-delta");
  const [deltaDisplayMode, setDeltaDisplayMode] = useState<DeltaDisplayMode>("absolute");
  const activeComparison = useMemo(
    () => hasInstanceFilters && selectedLanguages.length > 0 && selectedBenches.length > 0
      ? buildFilteredComparison(comparison, { languages: selectedLanguages, benches: selectedBenches })
      : comparison,
    [comparison, hasInstanceFilters, selectedBenches, selectedLanguages],
  );
  const models = getComparisonModels(activeComparison);
  const runDate = formatComparisonRunDate(comparison);
  const datasetSlice = getDatasetSliceSummary(activeComparison);
  const comparisonPair = getComparisonPair(activeComparison);
  const summaryCards = [
    { label: "Coding Agent", value: formatAgentName(activeComparison.agent) },
    { label: models.length > 1 ? "LLM Models" : "LLM Model", value: models.join(" / ") || "Unknown" },
    ...(runDate ? [{ label: "Run Date", value: runDate }] : []),
    ...(datasetSlice ? [{ label: "Dataset Slice", value: datasetSlice }] : []),
  ];

  useEffect(() => {
    if (!hasInstanceFilters) return;
    setSelectedLanguages(availableLanguages);
    setSelectedBenches(availableBenches);
  }, [comparison.id, hasInstanceFilters, availableLanguages, availableBenches]);

  return (
    <main className="mx-auto flex max-w-[88rem] flex-col gap-6 px-4 py-8">
      <a href="#/" className="inline-flex items-center gap-2 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground">
        <ArrowLeft className="h-4 w-4" /> Back to overview
      </a>
      <section className="rounded-lg bg-background p-6">
        <div className="flex items-center gap-3">
          <img src={comparison.icon} alt="" className="h-6 w-6 shrink-0" />
          <div>
            <h1 className="text-3xl font-semibold tracking-tight">{comparison.title}</h1>
            <p className="mt-2 text-sm text-muted-foreground">{comparison.summary}</p>
          </div>
        </div>
        <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {summaryCards.map((card) => <SummaryCard key={card.label} label={card.label} value={card.value} />)}
        </div>
      </section>
      {(hasInstanceFilters || comparisonPair) ? (
        <ComparisonControls
          comparisonPair={comparisonPair}
          viewMode={viewMode}
          deltaDisplayMode={deltaDisplayMode}
          setViewMode={setViewMode}
          setDeltaDisplayMode={setDeltaDisplayMode}
          languages={{ available: availableLanguages, selected: selectedLanguages, setSelected: setSelectedLanguages }}
          benches={{ available: availableBenches, selected: selectedBenches, setSelected: setSelectedBenches }}
          taskCount={activeComparison.tasks}
          hasInstanceFilters={hasInstanceFilters}
        />
      ) : null}
      <SetupParameters comparison={comparison} comparisonPair={comparisonPair} viewMode={viewMode} />
      {activeComparison.tasks > 0 ? (
        <ComparisonResults comparison={activeComparison} viewMode={viewMode} deltaDisplayMode={deltaDisplayMode} />
      ) : (
        <section className="rounded-lg border bg-background p-6 text-sm text-muted-foreground">No tasks match the selected language and dataset filters.</section>
      )}
    </main>
  );
}

function SummaryCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border p-4">
      <div className="text-xs uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-2 break-words text-sm font-medium">{value}</div>
    </div>
  );
}

function ComparisonControls({
  comparisonPair,
  viewMode,
  deltaDisplayMode,
  setViewMode,
  setDeltaDisplayMode,
  languages,
  benches,
  taskCount,
  hasInstanceFilters,
}: {
  comparisonPair: ReturnType<typeof getComparisonPair>;
  viewMode: ComparisonResultsViewMode;
  deltaDisplayMode: DeltaDisplayMode;
  setViewMode: (value: ComparisonResultsViewMode) => void;
  setDeltaDisplayMode: (value: DeltaDisplayMode) => void;
  languages: FilterState;
  benches: FilterState;
  taskCount: number;
  hasInstanceFilters: boolean;
}) {
  return (
    <section className="rounded-lg border bg-background px-4 py-4">
      <div className="space-y-4">
        <div className="text-xs uppercase tracking-wide text-muted-foreground">Comparison Controls</div>
        {comparisonPair ? (
          <div className="flex flex-col items-center gap-3">
            <ToggleGroup type="single" variant="outline" value={viewMode} onValueChange={(value) => value && setViewMode(value as ComparisonResultsViewMode)}>
              <ToggleGroupItem value="treatment-delta" className="gap-2"><TrendingUpDown className="h-4 w-4" />Treatment Delta</ToggleGroupItem>
              <ToggleGroupItem value="side-by-side" className="gap-2"><Columns2 className="h-4 w-4" />Side by Side</ToggleGroupItem>
            </ToggleGroup>
            {viewMode === "treatment-delta" ? (
              <ToggleGroup type="single" variant="outline" value={deltaDisplayMode} onValueChange={(value) => value && setDeltaDisplayMode(value as DeltaDisplayMode)}>
                <ToggleGroupItem value="absolute" aria-label="Numerical Diff"><span className="font-semibold tabular-nums">1.2→</span></ToggleGroupItem>
                <ToggleGroupItem value="percent" aria-label="Percent Diff"><Percent className="h-4 w-4" /></ToggleGroupItem>
              </ToggleGroup>
            ) : null}
          </div>
        ) : null}
        {hasInstanceFilters ? <FilterControls languages={languages} benches={benches} /> : null}
        <div className="flex justify-end text-sm text-muted-foreground">Showing <span className="mx-1 font-medium text-foreground">{taskCount}</span> matching tasks.</div>
      </div>
    </section>
  );
}

type FilterState = {
  available: string[];
  selected: string[];
  setSelected: (value: string[]) => void;
};

function FilterControls({ languages, benches }: { languages: FilterState; benches: FilterState }) {
  return (
    <div className="space-y-3">
      <FilterRow label="Languages" state={languages} formatter={formatLanguageLabel} />
      <FilterRow label="Datasets" state={benches} />
    </div>
  );
}

function FilterRow({ label, state, formatter = (value) => value }: { label: string; state: FilterState; formatter?: (value: string) => string }) {
  const allSelected = state.available.length > 0 && state.selected.length === state.available.length;
  const filterToggleClassName = "border border-input bg-background text-muted-foreground shadow-sm hover:border-primary/30 hover:bg-accent/60 hover:text-foreground data-[state=on]:border-primary data-[state=on]:bg-primary data-[state=on]:text-primary-foreground data-[state=on]:shadow";
  return (
    <div className="flex flex-col gap-2 lg:flex-row lg:items-start">
      <div className="w-full text-xs uppercase tracking-wide text-muted-foreground lg:w-24 lg:pt-2">
        {label}<div className="mt-1 text-[11px] normal-case tracking-normal text-muted-foreground/80">{state.selected.length}/{state.available.length} selected</div>
      </div>
      <div className="flex flex-1 flex-wrap gap-2">
        <Button variant={allSelected ? "default" : "outline"} className={!allSelected ? "text-muted-foreground" : undefined} onClick={() => state.setSelected(state.available)}>All</Button>
        <ToggleGroup type="multiple" variant="outline" value={state.selected} onValueChange={(value) => state.setSelected(value.length > 0 ? value : state.available)} className="flex flex-wrap justify-start gap-2">
          {state.available.map((value) => <ToggleGroupItem key={value} value={value} className={filterToggleClassName}>{formatter(value)}</ToggleGroupItem>)}
        </ToggleGroup>
      </div>
    </div>
  );
}

function SetupParameters({
  comparison,
  comparisonPair,
  viewMode,
}: {
  comparison: ComparisonCard;
  comparisonPair: ReturnType<typeof getComparisonPair>;
  viewMode: ComparisonResultsViewMode;
}) {
  return (
    <Accordion type="single" collapsible className="w-full rounded-lg border bg-background px-6">
      <AccordionItem value="setup-parameters" className="border-b-0">
        <AccordionTrigger className="text-xl font-semibold tracking-tight hover:no-underline">Setup Parameters</AccordionTrigger>
        <AccordionContent>
          {viewMode === "treatment-delta" && comparisonPair ? (
            <div className="grid gap-2.5 sm:grid-cols-2">
              {comparisonPair.treatment.parameters.map((parameter) => {
                const baselineValue = comparisonPair.baseline.parameters.find((baselineParameter) => baselineParameter.label === parameter.label)?.value;
                return <ParameterCard key={parameter.label} label={parameter.label} value={parameter.value} note={baselineValue === parameter.value ? "Matches baseline" : `Baseline: ${baselineValue ?? "—"}`} />;
              })}
            </div>
          ) : (
            <div className="grid gap-3 md:grid-cols-2">
              {comparison.variants.map((variant) => (
                <div key={variant.label} className="rounded-lg bg-background p-5">
                  <div className="mb-4 text-sm font-medium text-muted-foreground">{variant.name}</div>
                  <div className="space-y-3">{variant.parameters.map((parameter) => <ParameterCard key={parameter.label} label={parameter.label} value={parameter.value} />)}</div>
                </div>
              ))}
            </div>
          )}
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}

function ParameterCard({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="rounded-md border p-3">
      <div className="text-xs uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-1.5 whitespace-pre-wrap break-words text-sm font-medium leading-6">{value}</div>
      {note ? <div className="mt-1.5 break-words text-xs leading-5 text-muted-foreground">{note}</div> : null}
    </div>
  );
}
