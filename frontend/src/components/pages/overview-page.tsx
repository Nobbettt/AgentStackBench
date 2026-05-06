
import { useEffect, useState } from "react";

import { type CarouselApi, Carousel, CarouselContent, CarouselItem } from "@/components/ui/carousel";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { cardsForFilter, type ComparisonCard, type ComparisonData, type FilterMode } from "@/data/comparisons";

function formatAgentName(agent: ComparisonCard["agent"]): string {
  return agent === "codex" ? "Codex" : "Claude Code";
}

function ComparisonCardLink({ comparison }: { comparison: ComparisonCard }) {
  return (
    <a
      href={`#/comparisons/${comparison.id}`}
      className="block rounded-lg border bg-background p-6 transition-colors hover:bg-accent/40 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
    >
      <div className="flex items-center gap-3">
        <img src={comparison.icon} alt="" className="h-5 w-5 shrink-0" />
        <div className="text-sm font-medium">{comparison.title}</div>
      </div>
      <p className="mt-2 text-sm text-muted-foreground">{comparison.summary}</p>
      <div className="mt-4 text-xs uppercase tracking-wide text-muted-foreground">
        {formatAgentName(comparison.agent)} / {comparison.suite}
      </div>
    </a>
  );
}

export function OverviewPage({ data }: { data: ComparisonData }) {
  const [filter, setFilter] = useState<FilterMode>("all");
  const [carouselApi, setCarouselApi] = useState<CarouselApi>();

  useEffect(() => {
    if (!carouselApi) return;
    const targetIndex = data.filterOrder.indexOf(filter);
    if (targetIndex >= 0 && carouselApi.selectedScrollSnap() !== targetIndex) {
      carouselApi.scrollTo(targetIndex);
    }
  }, [carouselApi, data.filterOrder, filter]);

  useEffect(() => {
    if (!carouselApi) return;
    const syncFilterFromCarousel = () => {
      const selectedFilter = data.filterOrder[carouselApi.selectedScrollSnap()];
      if (selectedFilter) setFilter(selectedFilter);
    };
    syncFilterFromCarousel();
    carouselApi.on("select", syncFilterFromCarousel);
    carouselApi.on("reInit", syncFilterFromCarousel);
    return () => {
      carouselApi.off("select", syncFilterFromCarousel);
      carouselApi.off("reInit", syncFilterFromCarousel);
    };
  }, [carouselApi, data.filterOrder]);

  return (
    <main className="mx-auto flex max-w-7xl flex-col gap-8 px-4 py-8">
      <section className="space-y-4">
        <h2 className="text-2xl font-semibold tracking-tight">Comparisons</h2>
        <div className="flex justify-center">
          <ToggleGroup type="single" variant="outline" value={filter} onValueChange={(value) => value && setFilter(value as FilterMode)}>
            {data.filterOrder.map((mode) => (
              <ToggleGroupItem key={mode} value={mode} className={mode === "all" ? undefined : "gap-2"}>
                {mode === "all" ? "All" : (
                  <>
                    <img src={data.leaderboardRows.find((row) => row.agent === mode)?.icon} alt="" className="h-4 w-4" />
                    {mode === "codex" ? "Codex" : "Claude Code"}
                  </>
                )}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
        </div>
        <Carousel setApi={setCarouselApi} opts={{ align: "start" }} className="w-full touch-pan-y">
          <CarouselContent>
            {data.filterOrder.map((mode) => (
              <CarouselItem key={mode}>
                <div className="grid gap-4 md:grid-cols-2">
                  {cardsForFilter(data, mode).map((item) => (
                    <ComparisonCardLink key={`${mode}-${item.id}`} comparison={item} />
                  ))}
                </div>
              </CarouselItem>
            ))}
          </CarouselContent>
        </Carousel>
      </section>
      <section className="space-y-3">
        <h2 className="text-2xl font-semibold tracking-tight">Leaderboard</h2>
        <div className="rounded-md border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Model</TableHead>
                <TableHead>Suite</TableHead>
                <TableHead>Effort Level</TableHead>
                <TableHead>Tasks</TableHead>
                <TableHead>Pass@1</TableHead>
                <TableHead>Context F1</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.leaderboardRows.map((row) => (
                <TableRow key={`${row.agent}-${row.model}-${row.suite}`}>
                  <TableCell><div className="flex items-center gap-2"><img src={row.icon} alt="" className="h-4 w-4 shrink-0" /><span>{row.model}</span></div></TableCell>
                  <TableCell>{row.suite}</TableCell>
                  <TableCell>{row.effort}</TableCell>
                  <TableCell>{row.tasks}</TableCell>
                  <TableCell>{row.officialPassAt1 ?? row.completedRunRate ?? row.passAt1 ?? "—"}</TableCell>
                  <TableCell>{row.contextF1 ?? row.score ?? "—"}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
        <p className="text-sm text-muted-foreground">
          Pass@1 is computed here through the SWE-bench harness on generated patches. Completed Run Rate remains available in the comparison detail view as a separate execution-status metric.
        </p>
      </section>
    </main>
  );
}
