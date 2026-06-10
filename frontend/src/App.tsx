// SPDX-License-Identifier: Apache-2.0
// Fork note: Modified by Norbert Laszlo on 2026-05-26 from upstream ContextBench.
// Summary of changes: split frontend modules and added global navigation.

import { type ReactNode, useEffect, useState } from "react";
import { Github } from "lucide-react";

import { ComparisonInstanceDetailPage } from "@/components/comparison-results";
import { ComparisonPage } from "@/components/pages/comparison-page";
import { OverviewPage } from "@/components/pages/overview-page";
import { type ComparisonData, type ComparisonInstanceDetail, type ComparisonInstancesPayload, findComparisonById, withComparisonInstances } from "@/data/comparisons";
import { loadComparisonData } from "@/data/load-comparison-data";
import { loadComparisonInstances } from "@/data/load-comparison-instances";
import { loadInstanceDetail } from "@/data/load-instance-detail";
import { parseRoute, type Route } from "@/routes";

const GITHUB_REPO_URL = "https://github.com/Nobbettt/AgentStackBench";

export default function App() {
  const [data, setData] = useState<ComparisonData | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [route, setRoute] = useState<Route>(() =>
    typeof window === "undefined" ? { page: "overview" } : parseRoute(window.location.hash),
  );
  const [instanceDetail, setInstanceDetail] = useState<ComparisonInstanceDetail | null | undefined>(undefined);
  const [instanceDetailError, setInstanceDetailError] = useState<string | null>(null);
  const [instanceComparisonPayload, setInstanceComparisonPayload] = useState<ComparisonInstancesPayload | null | undefined>(undefined);
  const [instanceComparisonPayloadError, setInstanceComparisonPayloadError] = useState<string | null>(null);

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
      setInstanceComparisonPayload(undefined);
      setInstanceComparisonPayloadError(null);
      return () => {
        active = false;
      };
    }

    setInstanceDetail(undefined);
    setInstanceDetailError(null);
    setInstanceComparisonPayload(undefined);
    setInstanceComparisonPayloadError(null);
    void loadComparisonInstances(route.comparisonId, "metrics")
      .then((payload) => {
        if (active) setInstanceComparisonPayload(payload);
      })
      .catch((error: unknown) => {
        if (!active) return;
        setInstanceComparisonPayload(null);
        setInstanceComparisonPayloadError(error instanceof Error ? error.message : "Failed to load instance comparison metrics.");
      });
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
    return (
      <AppShell>
        <StatusPage title="Unable to load comparison data" message={loadError} />
      </AppShell>
    );
  }

  if (!data) {
    return (
      <AppShell>
        <StatusPage title="Loading comparison data" />
      </AppShell>
    );
  }

  const comparison = route.page === "comparison" ? findComparisonById(data, route.id) : undefined;
  const detailComparison = route.page === "instanceDetail" ? findComparisonById(data, route.comparisonId) : undefined;
  const detailComparisonWithInstances =
    detailComparison && instanceComparisonPayload
      ? withComparisonInstances(detailComparison, instanceComparisonPayload)
      : detailComparison;

  return (
    <AppShell>
      {route.page === "comparison" && comparison ? (
        <ComparisonPage key={comparison.id} comparison={comparison} />
      ) : route.page === "instanceDetail" ? (
        <main className="mx-auto flex max-w-[96rem] flex-col gap-4 px-4 pb-8 pt-4">
          {detailComparison ? (
            instanceComparisonPayloadError ? (
              <section className="rounded-lg border bg-background p-6 text-sm text-rose-700">
                Unable to load instance comparison metrics: {instanceComparisonPayloadError}
              </section>
            ) : instanceComparisonPayload === undefined ? (
              <section className="rounded-lg border bg-background p-6 text-sm text-muted-foreground">
                Loading instance comparison metrics…
              </section>
            ) : (
              <ComparisonInstanceDetailPage
                comparison={detailComparisonWithInstances ?? detailComparison}
                instanceId={route.instanceId}
                detail={instanceDetail}
                detailError={instanceDetailError}
              />
            )
          ) : (
            <section className="rounded-lg border bg-background p-6 text-sm text-muted-foreground">
              Loading instance detail…
            </section>
          )}
        </main>
      ) : (
        <OverviewPage data={data} />
      )}
    </AppShell>
  );
}

function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="sticky top-0 z-50 border-b bg-background/95 backdrop-blur">
        <nav className="mx-auto flex h-14 max-w-[96rem] items-center justify-between px-4" aria-label="Primary">
          <a
            href="#/"
            className="inline-flex min-h-9 items-center text-lg font-semibold tracking-tight text-foreground transition-colors hover:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          >
            AgentStackBench
          </a>
          <a
            href={GITHUB_REPO_URL}
            target="_blank"
            rel="noreferrer"
            className="inline-flex h-9 items-center gap-2 rounded-md border border-input bg-background px-3 text-sm font-medium shadow-sm transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            aria-label="Open AgentStackBench on GitHub"
          >
            <Github className="h-4 w-4" />
            <span>GitHub</span>
          </a>
        </nav>
      </header>
      {children}
    </div>
  );
}

function StatusPage({ title, message }: { title: string; message?: string }) {
  return (
    <main className="mx-auto flex max-w-3xl flex-col gap-3 px-4 py-8">
      <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
      {message ? <p className="text-sm text-muted-foreground">{message}</p> : null}
    </main>
  );
}
