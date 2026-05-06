// Fork note: Modified by Norbert Laszlo on 2026-04-24 from upstream ContextBench.
// Summary of changes: split frontend page and comparison components into smaller modules.

import { useEffect, useState } from "react";
import { ArrowLeft } from "lucide-react";

import { ComparisonInstanceDetailPage } from "@/components/comparison-results";
import { ComparisonPage } from "@/components/pages/comparison-page";
import { OverviewPage } from "@/components/pages/overview-page";
import { type ComparisonData, type ComparisonInstanceDetail, findComparisonById } from "@/data/comparisons";
import { loadComparisonData } from "@/data/load-comparison-data";
import { loadInstanceDetail } from "@/data/load-instance-detail";
import { parseRoute, type Route } from "@/routes";

export default function App() {
  const [data, setData] = useState<ComparisonData | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [route, setRoute] = useState<Route>(() =>
    typeof window === "undefined" ? { page: "overview" } : parseRoute(window.location.hash),
  );
  const [instanceDetail, setInstanceDetail] = useState<ComparisonInstanceDetail | null | undefined>(undefined);
  const [instanceDetailError, setInstanceDetailError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    void loadComparisonData()
      .then((nextData) => {
        if (!active) return;
        setData(nextData);
        setLoadError(null);
      })
      .catch((error: unknown) => {
        if (active) setLoadError(error instanceof Error ? error.message : "Failed to load comparison data.");
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    const handleHashChange = () => setRoute(parseRoute(window.location.hash));
    window.addEventListener("hashchange", handleHashChange);
    return () => window.removeEventListener("hashchange", handleHashChange);
  }, []);

  useEffect(() => {
    let active = true;
    if (route.page !== "instanceDetail") {
      setInstanceDetail(undefined);
      setInstanceDetailError(null);
      return () => {
        active = false;
      };
    }

    setInstanceDetail(undefined);
    setInstanceDetailError(null);
    void loadInstanceDetail(route.comparisonId, route.instanceId)
      .then((nextDetail) => {
        if (active) setInstanceDetail(nextDetail);
      })
      .catch((error: unknown) => {
        if (!active) return;
        setInstanceDetail(null);
        setInstanceDetailError(error instanceof Error ? error.message : "Failed to load instance detail.");
      });
    return () => {
      active = false;
    };
  }, [route]);

  if (loadError) {
    return <StatusPage title="Unable to load comparison data" message={loadError} />;
  }

  if (!data) {
    return <StatusPage title="Loading comparison data" />;
  }

  const comparison = route.page === "comparison" ? findComparisonById(data, route.id) : undefined;
  const detailComparison = route.page === "instanceDetail" ? findComparisonById(data, route.comparisonId) : undefined;

  return (
    <div className="min-h-screen bg-background text-foreground">
      {route.page === "comparison" && comparison ? (
        <ComparisonPage key={comparison.id} comparison={comparison} />
      ) : route.page === "instanceDetail" ? (
        <main className="mx-auto flex max-w-[96rem] flex-col gap-6 px-4 py-8">
          <a
            href={detailComparison ? `#/comparisons/${detailComparison.id}` : "#/"}
            className="inline-flex items-center gap-2 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground"
          >
            <ArrowLeft className="h-4 w-4" />
            {detailComparison ? "Back to comparison" : "Back to overview"}
          </a>
          {detailComparison ? (
            <ComparisonInstanceDetailPage
              comparison={detailComparison}
              instanceId={route.instanceId}
              detail={instanceDetail}
              detailError={instanceDetailError}
            />
          ) : (
            <section className="rounded-lg border bg-background p-6 text-sm text-muted-foreground">
              Loading instance detail…
            </section>
          )}
        </main>
      ) : (
        <OverviewPage data={data} />
      )}
    </div>
  );
}

function StatusPage({ title, message }: { title: string; message?: string }) {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <main className="mx-auto flex max-w-3xl flex-col gap-3 px-4 py-8">
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        {message ? <p className="text-sm text-muted-foreground">{message}</p> : null}
      </main>
    </div>
  );
}
