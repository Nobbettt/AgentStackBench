
import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, Columns2, Percent, SlidersHorizontal, TrendingUpDown } from "lucide-react";
import { Label, Pie, PieChart } from "recharts";

import { ComparisonResults, type ComparisonResultsTab } from "@/components/comparison-results";
import { Button } from "@/components/ui/button";
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { buildFilteredComparison, comparisonHasInstanceData, getAvailableBenches, getAvailableLanguages } from "@/data/comparison-aggregation";
import { type ComparisonCard, type ComparisonInstanceBundle, type ComparisonInstancesPayload, withComparisonInstances } from "@/data/comparisons";
import { loadComparisonInstances } from "@/data/load-comparison-instances";
import { formatLanguageLabel } from "@/components/comparison/format";
import type { ComparisonResultsViewMode, DeltaDisplayMode } from "@/components/comparison/types";

function formatComparisonRunDate(comparison: ComparisonCard): string | null {
  const timestamp = comparison.completedAt ?? comparison.startedAt;
  if (!timestamp) return null;
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return null;
  return new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" }).format(date);
}

const sliceColors = [
  "hsl(var(--chart-1))",
  "hsl(var(--chart-2))",
  "hsl(var(--chart-3))",
  "hsl(var(--chart-4))",
  "hsl(var(--chart-5))",
];
const TOP_REPOSITORY_SLICE_LIMIT = 8;
const repositorySizeBuckets = [
  { id: "repo-size-lt-1k", label: "<1k files", min: 0, max: 1_000 },
  { id: "repo-size-1k-5k", label: "1k-5k files", min: 1_000, max: 5_000 },
  { id: "repo-size-5k-20k", label: "5k-20k files", min: 5_000, max: 20_000 },
  { id: "repo-size-20k-50k", label: "20k-50k files", min: 20_000, max: 50_000 },
  { id: "repo-size-50k-plus", label: "50k+ files", min: 50_000, max: Number.POSITIVE_INFINITY },
] as const;

type DistributionSliceEntry = {
  id: string;
  label: string;
  count: number;
  fill: string;
};

type DistributionSliceData = {
  slices: DistributionSliceEntry[];
  distinctCount: number;
};

function getDatasetSliceData(comparison: ComparisonCard): DistributionSliceEntry[] {
  const benchCounts = comparison.taskSet?.benchCounts;
  if (!benchCounts || Object.keys(benchCounts).length === 0) return [];
  return Object.entries(benchCounts)
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .map(([bench, count], index) => ({
      id: `bench-${bench}`,
      label: bench,
      count,
      fill: sliceColors[index % sliceColors.length],
    }));
}

function getLanguageSliceData(comparison: ComparisonCard): DistributionSliceEntry[] {
  const languagesByInstanceId = new Map<string, string>();
  for (const variant of comparison.variants) {
    for (const instance of variant.instances ?? []) {
      if (!languagesByInstanceId.has(instance.instanceId)) {
        languagesByInstanceId.set(instance.instanceId, instance.language);
      }
    }
  }

  const languageCounts = new Map<string, number>();
  for (const language of languagesByInstanceId.values()) {
    languageCounts.set(language, (languageCounts.get(language) ?? 0) + 1);
  }

  return Array.from(languageCounts.entries())
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .map(([language, count], index) => ({
      id: `language-${language}`,
      label: formatLanguageLabel(language),
      count,
      fill: sliceColors[index % sliceColors.length],
    }));
}

function getRepositorySliceData(comparison: ComparisonCard): DistributionSliceData {
  const repositoriesByInstanceId = new Map<string, string>();
  for (const variant of comparison.variants) {
    for (const instance of variant.instances ?? []) {
      if (!repositoriesByInstanceId.has(instance.instanceId)) {
        repositoriesByInstanceId.set(instance.instanceId, getRepositoryLabel(instance.originalInstanceId ?? instance.instanceId));
      }
    }
  }

  const repositoryCounts = new Map<string, number>();
  for (const repository of repositoriesByInstanceId.values()) {
    repositoryCounts.set(repository, (repositoryCounts.get(repository) ?? 0) + 1);
  }

  const sortedRepositories = Array.from(repositoryCounts.entries())
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]));
  const topRepositories = sortedRepositories.slice(0, TOP_REPOSITORY_SLICE_LIMIT);
  const otherCount = sortedRepositories
    .slice(TOP_REPOSITORY_SLICE_LIMIT)
    .reduce((sum, [, count]) => sum + count, 0);
  const entries = otherCount > 0 ? [...topRepositories, ["Other", otherCount] as [string, number]] : topRepositories;

  return {
    distinctCount: repositoryCounts.size,
    slices: entries.map(([repository, count], index) => ({
      id: repository === "Other" ? "repo-other" : `repo-${index}`,
      label: repository,
      count,
      fill: sliceColors[index % sliceColors.length],
    })),
  };
}

function getRepositorySizeSliceData(comparison: ComparisonCard): DistributionSliceEntry[] {
  const sizeByInstanceId = new Map<string, number | null>();
  for (const variant of comparison.variants) {
    for (const instance of variant.instances ?? []) {
      if (sizeByInstanceId.has(instance.instanceId)) continue;
      const trackedFiles = instance.repositorySize?.trackedFiles;
      sizeByInstanceId.set(instance.instanceId, typeof trackedFiles === "number" ? trackedFiles : null);
    }
  }

  const bucketCounts = new Map<string, number>(repositorySizeBuckets.map((bucket) => [bucket.id, 0]));
  let unavailableCount = 0;
  for (const trackedFiles of sizeByInstanceId.values()) {
    if (trackedFiles === null) {
      unavailableCount += 1;
      continue;
    }

    const bucket = repositorySizeBuckets.find((candidate) => trackedFiles >= candidate.min && trackedFiles < candidate.max);
    if (bucket) {
      bucketCounts.set(bucket.id, (bucketCounts.get(bucket.id) ?? 0) + 1);
    }
  }

  const bucketEntries = repositorySizeBuckets
    .map((bucket, index) => ({
      id: bucket.id,
      label: bucket.label,
      count: bucketCounts.get(bucket.id) ?? 0,
      fill: sliceColors[index % sliceColors.length],
    }))
    .filter((entry) => entry.count > 0);

  return unavailableCount > 0
    ? [
        ...bucketEntries,
        {
          id: "repo-size-unavailable",
          label: "No size data",
          count: unavailableCount,
          fill: sliceColors[bucketEntries.length % sliceColors.length],
        },
      ]
    : bucketEntries;
}

function getRepositoryLabel(instanceId: string): string {
  const [ownerRaw, repoRaw] = instanceId.split("__");
  if (!ownerRaw || !repoRaw) return instanceId;
  const owner = ownerRaw.replace(/^instance_/, "");
  const repo = repoRaw
    .replace(/-[0-9a-f]{40}(?:-v[0-9a-z]+)?$/i, "")
    .replace(/-v[0-9a-z]+$/i, "")
    .replace(/-\d+$/i, "");
  return `${owner}/${repo}`;
}

function getComparisonPair(comparison: ComparisonCard) {
  if (comparison.variants.length < 2) return null;
  return { baseline: comparison.variants[0], treatment: comparison.variants[1] };
}

const segmentedControlClassName = "gap-0 rounded-md shadow-lg backdrop-blur";
const segmentedControlItemClassName = "w-14 rounded-none bg-background px-0 data-[state=on]:bg-primary data-[state=on]:text-primary-foreground";
export type ComparisonPageTab = "setup" | ComparisonResultsTab;
const comparisonTabs: Array<{ id: ComparisonPageTab; label: string }> = [
  { id: "overview", label: "Overview" },
  { id: "setup", label: "Setup & Dataset" },
  { id: "execution", label: "Resource Usage" },
  { id: "resolution", label: "Resolution" },
  { id: "context", label: "Context" },
  { id: "correlations", label: "Correlations" },
  { id: "languages", label: "Metric Breakdowns" },
  { id: "usage", label: "Skills" },
  { id: "tools", label: "Tools" },
  { id: "issues", label: "All Tasks" },
];

const comparisonBundleLoadOrder: ComparisonInstanceBundle[] = ["index", "metrics", "trajectory"];

export function bundlesForTab(tab: ComparisonPageTab): ComparisonInstanceBundle[] {
  if (tab === "overview") return [];
  if (tab === "setup") return ["index"];
  if (tab === "context" || tab === "correlations") return ["index", "metrics", "trajectory"];
  return ["index", "metrics"];
}

function buildFilteredTaskIndexComparison(
  comparison: ComparisonCard,
  filters: { languages: string[]; benches: string[] },
): ComparisonCard {
  if (!comparisonHasInstanceData(comparison)) return comparison;

  const idSets = comparison.variants.map((variant) =>
    new Set(
      (variant.instances ?? [])
        .filter((instance) => filters.languages.includes(instance.language) && filters.benches.includes(instance.bench))
        .map((instance) => instance.instanceId),
    ),
  );
  const [firstSet, ...restSets] = idSets;
  const selectedIds = new Set(firstSet ?? []);
  for (const idSet of restSets) {
    for (const instanceId of Array.from(selectedIds)) {
      if (!idSet.has(instanceId)) selectedIds.delete(instanceId);
    }
  }

  const filteredVariants = comparison.variants.map((variant) => ({
    ...variant,
    instances: (variant.instances ?? []).filter((instance) => selectedIds.has(instance.instanceId)),
  }));
  const benchCounts = Object.fromEntries(
    Object.entries(
      filteredVariants[0]?.instances?.reduce<Record<string, number>>((counts, instance) => {
        counts[instance.bench] = (counts[instance.bench] ?? 0) + 1;
        return counts;
      }, {}) ?? {},
    ).sort((left, right) => left[0].localeCompare(right[0])),
  );

  return {
    ...comparison,
    tasks: selectedIds.size,
    taskSet: {
      count: selectedIds.size,
      benchCounts,
      hash: selectedIds.size === comparison.tasks ? comparison.taskSet?.hash : undefined,
      sourceDatasetCount: selectedIds.size === comparison.tasks ? comparison.taskSet?.sourceDatasetCount : undefined,
      selectionKind: selectedIds.size === comparison.tasks ? comparison.taskSet?.selectionKind : undefined,
    },
    variants: filteredVariants,
  };
}

export function ComparisonPage({ comparison }: { comparison: ComparisonCard }) {
  const [instancePayloads, setInstancePayloads] = useState<Partial<Record<ComparisonInstanceBundle, ComparisonInstancesPayload>>>({});
  const [instanceLoadError, setInstanceLoadError] = useState<string | null>(null);
  const loadedBundlesRef = useRef<Set<ComparisonInstanceBundle>>(new Set());
  const loadingBundlesRef = useRef<Set<ComparisonInstanceBundle>>(new Set());
  const [selectedLanguages, setSelectedLanguages] = useState<string[]>([]);
  const [selectedBenches, setSelectedBenches] = useState<string[]>([]);
  const [viewMode, setViewMode] = useState<ComparisonResultsViewMode>("treatment-delta");
  const [deltaDisplayMode, setDeltaDisplayMode] = useState<DeltaDisplayMode>("absolute");
  const [activeTab, setActiveTab] = useState<ComparisonPageTab>("overview");
  const loadingPromisesRef = useRef<Map<ComparisonInstanceBundle, Promise<ComparisonInstancesPayload>>>(new Map());
  const requiredBundles = useMemo(() => bundlesForTab(activeTab), [activeTab]);
  const requiredBundleKey = requiredBundles.join(",");
  const comparisonWithInstances = useMemo(
    () =>
      comparisonBundleLoadOrder.reduce(
        (nextComparison, bundle) => {
          const payload = instancePayloads[bundle];
          return payload ? withComparisonInstances(nextComparison, payload) : nextComparison;
        },
        comparison,
      ),
    [comparison, instancePayloads],
  );
  const hasEmbeddedInstanceData = comparisonHasInstanceData(comparison);
  const hasInstanceFilters = comparisonHasInstanceData(comparisonWithInstances);
  const metricsBundleLoaded = hasEmbeddedInstanceData || Boolean(instancePayloads.metrics);
  const instanceDataLoading = !hasEmbeddedInstanceData && requiredBundles.some((bundle) => !instancePayloads[bundle]) && !instanceLoadError;
  const availableLanguages = useMemo(
    () => (hasInstanceFilters ? getAvailableLanguages(comparisonWithInstances) : []),
    [comparisonWithInstances, hasInstanceFilters],
  );
  const availableBenches = useMemo(
    () => (hasInstanceFilters ? getAvailableBenches(comparisonWithInstances) : []),
    [comparisonWithInstances, hasInstanceFilters],
  );
  const activeComparison = useMemo(
    () => {
      if (!hasInstanceFilters) return comparisonWithInstances;
      const filters = {
        languages: selectedLanguages.length > 0 ? selectedLanguages : availableLanguages,
        benches: selectedBenches.length > 0 ? selectedBenches : availableBenches,
      };
      return metricsBundleLoaded
        ? buildFilteredComparison(comparisonWithInstances, filters)
        : buildFilteredTaskIndexComparison(comparisonWithInstances, filters);
    },
    [availableBenches, availableLanguages, comparisonWithInstances, hasInstanceFilters, metricsBundleLoaded, selectedBenches, selectedLanguages],
  );
  const runDate = formatComparisonRunDate(comparison);
  const datasetSliceData = activeTab === "setup" ? getDatasetSliceData(activeComparison) : [];
  const languageSliceData = activeTab === "setup" ? getLanguageSliceData(activeComparison) : [];
  const repositorySliceData = activeTab === "setup" ? getRepositorySliceData(activeComparison) : { slices: [], distinctCount: 0 };
  const repositorySizeSliceData = activeTab === "setup" ? getRepositorySizeSliceData(activeComparison) : [];
  const comparisonPair = getComparisonPair(activeComparison);
  const showComparisonControls = hasInstanceFilters || Boolean(comparisonPair);

  useEffect(() => {
    setInstancePayloads({});
    setInstanceLoadError(null);
    loadedBundlesRef.current.clear();
    loadingBundlesRef.current.clear();
    loadingPromisesRef.current.clear();
    setSelectedLanguages([]);
    setSelectedBenches([]);
  }, [comparison.id]);

  useEffect(() => {
    let active = true;
    setInstanceLoadError(null);

    if (comparisonHasInstanceData(comparison) || requiredBundles.length === 0) {
      return () => {
        active = false;
      };
    }

    for (const bundle of requiredBundles) {
      if (instancePayloads[bundle] || loadedBundlesRef.current.has(bundle)) continue;
      let loadPromise = loadingPromisesRef.current.get(bundle);
      if (!loadPromise) {
        loadingBundlesRef.current.add(bundle);
        loadPromise = loadComparisonInstances(comparison.id, bundle);
        loadingPromisesRef.current.set(bundle, loadPromise);
      }
      void loadPromise
        .then((payload) => {
          if (!active) return;
          loadedBundlesRef.current.add(bundle);
          setInstancePayloads((currentPayloads) => ({ ...currentPayloads, [bundle]: payload }));
          setInstanceLoadError(null);
        })
        .catch((error: unknown) => {
          if (!active) return;
          setInstanceLoadError(error instanceof Error ? error.message : `Failed to load ${bundle} comparison data.`);
        })
        .finally(() => {
          loadingBundlesRef.current.delete(bundle);
          loadingPromisesRef.current.delete(bundle);
        });
    }

    return () => {
      active = false;
    };
  }, [comparison, instancePayloads, requiredBundleKey, requiredBundles]);

  useEffect(() => {
    if (!hasInstanceFilters) return;
    setSelectedLanguages(availableLanguages);
    setSelectedBenches(availableBenches);
  }, [comparison.id, hasInstanceFilters]);

  return (
    <main className="mx-auto flex max-w-[88rem] flex-col gap-4 px-4 pb-40 pt-4">
      <section className="rounded-lg bg-background px-4 py-3">
        <div className="flex items-center gap-3">
          <img src={comparison.icon} alt="" className="h-6 w-6 shrink-0" />
          <div>
            <h1 className="text-3xl font-semibold tracking-tight">{comparison.title}</h1>
            <p className="mt-1 text-sm text-muted-foreground">{comparison.summary}</p>
            {runDate ? <p className="mt-1 text-sm text-muted-foreground">Run Date: {runDate}</p> : null}
          </div>
        </div>
      </section>
      <ComparisonTabs activeTab={activeTab} onChange={setActiveTab} />
      {showComparisonControls ? (
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
      {instanceDataLoading ? (
        <InstanceDataStatus bundles={requiredBundles.filter((bundle) => !instancePayloads[bundle])} />
      ) : instanceLoadError ? (
        <InstanceDataStatus error={instanceLoadError} />
      ) : activeTab === "setup" ? (
        <SetupParameters
          comparison={activeComparison}
          comparisonPair={comparisonPair}
          viewMode={viewMode}
          datasetSliceData={datasetSliceData}
          languageSliceData={languageSliceData}
          repositorySliceData={repositorySliceData}
          repositorySizeSliceData={repositorySizeSliceData}
        />
      ) : activeComparison.tasks > 0 ? (
        <ComparisonResults comparison={activeComparison} viewMode={viewMode} deltaDisplayMode={deltaDisplayMode} activeTab={activeTab} />
      ) : (
        <section className="rounded-lg bg-background p-6 text-sm text-muted-foreground">No tasks match the selected language and dataset filters.</section>
      )}
    </main>
  );
}

function InstanceDataStatus({ error, bundles = [] }: { error?: string; bundles?: ComparisonInstanceBundle[] }) {
  const bundleLabel = bundles.length > 0 ? ` (${bundles.join(", ")})` : "";
  return (
    <section className="rounded-lg border bg-background p-4 text-sm text-muted-foreground">
      {error ? `Unable to load task-level comparison data: ${error}` : `Loading task-level comparison data${bundleLabel}…`}
    </section>
  );
}

function ComparisonTabs({ activeTab, onChange }: { activeTab: ComparisonPageTab; onChange: (tab: ComparisonPageTab) => void }) {
  return (
    <div className="overflow-x-auto border-b" role="tablist" aria-label="Comparison result sections">
      <div className="flex min-w-max gap-1">
        {comparisonTabs.map((tab) => {
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
  const [filtersOpen, setFiltersOpen] = useState(false);

  return (
    <section className="flex w-full flex-col items-end gap-2 sm:fixed sm:bottom-4 sm:right-4 sm:z-40 sm:w-[min(40rem,calc(100vw-2rem))]">
      {hasInstanceFilters ? (
        <div
          id="comparison-filter-controls"
          className={
            filtersOpen
              ? "max-h-[min(26rem,60vh)] w-full overflow-y-auto rounded-lg border bg-background/95 p-3 shadow-lg backdrop-blur"
              : "hidden"
          }
        >
          <FilterControls languages={languages} benches={benches} />
          <div className="mt-3 flex justify-end text-sm text-muted-foreground">
            Showing <span className="mx-1 font-medium text-foreground">{taskCount}</span> matching tasks.
          </div>
        </div>
      ) : null}
      <div className="flex flex-col items-end gap-2">
        {hasInstanceFilters ? (
          <Button
            variant="outline"
            className="h-9 gap-2 bg-background px-3 shadow-lg backdrop-blur"
            onClick={() => setFiltersOpen((isOpen) => !isOpen)}
            aria-expanded={filtersOpen}
            aria-controls="comparison-filter-controls"
          >
            <SlidersHorizontal className="h-4 w-4" />
            Filters
            <ChevronDown className={`h-4 w-4 transition-transform ${filtersOpen ? "rotate-180" : ""}`} />
          </Button>
        ) : null}
        {comparisonPair ? (
          <>
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
                onValueChange={(value) => value && setDeltaDisplayMode(value as DeltaDisplayMode)}
                className={segmentedControlClassName}
              >
                <ToggleGroupItem
                  value="absolute"
                  aria-label="Numerical diff"
                  title="Numerical diff"
                  className={`${segmentedControlItemClassName} rounded-l-md border-r-0`}
                >
                  <span className="font-semibold tabular-nums">1.2→</span>
                </ToggleGroupItem>
                <ToggleGroupItem
                  value="percent"
                  aria-label="Percent diff"
                  title="Percent diff"
                  className={`${segmentedControlItemClassName} rounded-r-md`}
                >
                  <Percent className="h-4 w-4" />
                </ToggleGroupItem>
              </ToggleGroup>
            </div>
            <ToggleGroup
              type="single"
              variant="outline"
              value={viewMode}
              onValueChange={(value) => value && setViewMode(value as ComparisonResultsViewMode)}
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
          </>
        ) : null}
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
  const toggleAll = () => state.setSelected(allSelected ? [] : state.available);
  const filterToggleClassName = "border border-input bg-background text-muted-foreground shadow-sm hover:border-primary/30 hover:bg-accent/60 hover:text-foreground data-[state=on]:border-primary data-[state=on]:bg-primary data-[state=on]:text-primary-foreground data-[state=on]:shadow";
  return (
    <div className="flex flex-col gap-2 lg:flex-row lg:items-start">
      <div className="w-full text-xs uppercase tracking-wide text-muted-foreground lg:w-24 lg:pt-2">
        {label}<div className="mt-1 text-[11px] normal-case tracking-normal text-muted-foreground/80">{state.selected.length}/{state.available.length} selected</div>
      </div>
      <div className="flex flex-1 flex-wrap gap-2">
        <Button
          variant={allSelected ? "default" : "outline"}
          className={!allSelected ? "text-muted-foreground" : undefined}
          onClick={toggleAll}
          aria-pressed={allSelected}
          title={allSelected ? `Deselect all ${label.toLowerCase()}` : `Select all ${label.toLowerCase()}`}
        >
          All
        </Button>
        <ToggleGroup type="multiple" variant="outline" value={state.selected} onValueChange={state.setSelected} className="flex flex-wrap justify-start gap-2">
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
  datasetSliceData,
  languageSliceData,
  repositorySliceData,
  repositorySizeSliceData,
}: {
  comparison: ComparisonCard;
  comparisonPair: ReturnType<typeof getComparisonPair>;
  viewMode: ComparisonResultsViewMode;
  datasetSliceData: DistributionSliceEntry[];
  languageSliceData: DistributionSliceEntry[];
  repositorySliceData: DistributionSliceData;
  repositorySizeSliceData: DistributionSliceEntry[];
}) {
  return (
    <section className="space-y-6">
      <RunSummary
        datasetSliceData={datasetSliceData}
        languageSliceData={languageSliceData}
        repositorySliceData={repositorySliceData}
        repositorySizeSliceData={repositorySizeSliceData}
      />
      <div className="space-y-4">
        <h2 className="text-xl font-semibold tracking-tight">Setup Parameters</h2>
        <div className="rounded-lg bg-background p-5">
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
        </div>
      </div>
    </section>
  );
}

function RunSummary({
  datasetSliceData,
  languageSliceData,
  repositorySliceData,
  repositorySizeSliceData,
}: {
  datasetSliceData: DistributionSliceEntry[];
  languageSliceData: DistributionSliceEntry[];
  repositorySliceData: DistributionSliceData;
  repositorySizeSliceData: DistributionSliceEntry[];
}) {
  if (
    datasetSliceData.length === 0 &&
    languageSliceData.length === 0 &&
    repositorySliceData.slices.length === 0 &&
    repositorySizeSliceData.length === 0
  ) return null;

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold tracking-tight">Dataset</h2>
      <div className="grid gap-6 lg:grid-cols-2 xl:grid-cols-4">
        {datasetSliceData.length > 0 ? (
          <DistributionDonutChart
            title="Dataset Slice"
            description="Task distribution by benchmark"
            data={datasetSliceData}
            variant="donut"
          />
        ) : null}
        {languageSliceData.length > 0 ? (
          <DistributionDonutChart
            title="Languages"
            description="Task distribution by language"
            data={languageSliceData}
            variant="label"
          />
        ) : null}
        {repositorySliceData.slices.length > 0 ? (
          <DistributionDonutChart
            title="Repositories"
            description="Top repos plus Other"
            data={repositorySliceData.slices}
            variant="donut"
            centerValue={repositorySliceData.distinctCount}
            centerLabel="repos"
          />
        ) : null}
        {repositorySizeSliceData.length > 0 ? (
          <DistributionDonutChart
            title="Repository Size"
            description="Tracked file count buckets"
            data={repositorySizeSliceData}
            variant="label"
          />
        ) : null}
      </div>
    </div>
  );
}

function DistributionDonutChart({
  title,
  description,
  data,
  variant,
  centerValue,
  centerLabel,
}: {
  title: string;
  description: string;
  data: DistributionSliceEntry[];
  variant: "donut" | "label";
  centerValue?: number;
  centerLabel?: string;
}) {
  const total = data.reduce((sum, entry) => sum + entry.count, 0);
  const displayedCenterValue = centerValue ?? total;
  const displayedCenterLabel = centerLabel ?? "tasks";
  const chartConfig = data.reduce<ChartConfig>((config, entry) => {
    config[entry.id] = { label: entry.label, color: entry.fill };
    return config;
  }, {});
  const chartSize = variant === "label"
    ? {
        width: 320,
        height: 208,
        className: "mx-auto h-[13rem] w-full max-w-[20rem] overflow-visible",
        outerRadius: 54,
        innerRadius: 0,
        labelOffset: 22,
      }
    : {
        width: 208,
        height: 208,
        className: "mx-auto h-[13rem] w-[13rem] overflow-visible",
        outerRadius: 58,
        innerRadius: 40,
        labelOffset: 0,
      };

  return (
    <div className="flex min-w-0 flex-col p-4">
      <div className="text-center">
        <div className="text-xs uppercase tracking-wide text-muted-foreground">{title}</div>
        <div className="mt-1 text-sm text-muted-foreground">{description}</div>
      </div>
      <div className="mt-3 flex flex-1 flex-col items-center">
        <ChartContainer config={chartConfig} className={chartSize.className}>
          <PieChart width={chartSize.width} height={chartSize.height} style={{ overflow: "visible" }}>
            <ChartTooltip
              content={
                <ChartTooltipContent
                  hideLabel
                  nameKey="id"
                  formatter={(value) => {
                    const count = Number(value);
                    const percent = total > 0 ? (count / total) * 100 : 0;
                    return `${count.toLocaleString()} tasks (${percent.toFixed(1)}%)`;
                  }}
                />
              }
            />
            <Pie
              data={data}
              dataKey="count"
              nameKey="id"
              innerRadius={variant === "donut" ? chartSize.innerRadius : 0}
              outerRadius={chartSize.outerRadius}
              cx={chartSize.width / 2}
              cy={chartSize.height / 2}
              stroke="none"
              strokeWidth={0}
              labelLine={false}
              label={variant === "label" ? (props) => renderPieLabel(props, chartSize.labelOffset) : false}
            >
              {variant === "donut" ? (
                <Label
                  content={({ viewBox }) => {
                    if (!viewBox || !("cx" in viewBox) || !("cy" in viewBox)) return null;
                    return (
                      <text x={viewBox.cx} y={viewBox.cy} textAnchor="middle" dominantBaseline="middle">
                        <tspan x={viewBox.cx} y={viewBox.cy} className="fill-foreground text-2xl font-semibold">
                          {displayedCenterValue.toLocaleString()}
                        </tspan>
                        <tspan x={viewBox.cx} y={(viewBox.cy ?? 0) + 20} className="fill-muted-foreground text-xs">
                          {displayedCenterLabel}
                        </tspan>
                      </text>
                    );
                  }}
                />
              ) : null}
            </Pie>
          </PieChart>
        </ChartContainer>
        <div className="mx-auto mt-3 flex max-w-xl flex-wrap justify-center gap-x-4 gap-y-2 text-xs text-muted-foreground">
          {data.map((entry) => {
            return (
              <div key={entry.id} className="inline-flex items-center gap-2">
                <span className="h-2 w-2 shrink-0 rounded-full" style={{ backgroundColor: entry.fill }} />
                <span>{entry.label}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function renderPieLabel({
  cx,
  cy,
  midAngle,
  outerRadius,
  payload,
}: {
  cx?: number;
  cy?: number;
  midAngle?: number;
  outerRadius?: number;
  payload?: DistributionSliceEntry;
}, labelOffset: number) {
  if (
    !payload ||
    typeof cx !== "number" ||
    typeof cy !== "number" ||
    typeof midAngle !== "number" ||
    typeof outerRadius !== "number"
  ) {
    return null;
  }

  const radius = outerRadius + labelOffset;
  const angle = -midAngle * (Math.PI / 180);
  const x = cx + radius * Math.cos(angle);
  const y = cy + radius * Math.sin(angle);

  return (
    <text
      x={x}
      y={y}
      textAnchor={typeof cx === "number" && x > cx ? "start" : "end"}
      dominantBaseline="central"
      className="fill-foreground text-xs"
    >
      {payload.label}
    </text>
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
