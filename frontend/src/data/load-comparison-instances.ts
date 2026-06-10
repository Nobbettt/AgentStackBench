// SPDX-License-Identifier: Apache-2.0

import type { ComparisonInstanceBundle, ComparisonInstancesPayload } from "@/data/comparisons";

export async function loadComparisonInstances(
  comparisonId: string,
  bundle: ComparisonInstanceBundle,
): Promise<ComparisonInstancesPayload> {
  // "no-cache" revalidates with the server (via ETag/Last-Modified) before
  // using a cached copy; "force-cache" would serve stale bundles after a
  // redeploy and silently disagree with the freshly fetched comparison.json.
  const response = await fetch(
    `${import.meta.env.BASE_URL}comparison-data/${encodeURIComponent(comparisonId)}/${bundle}.json`,
    { cache: import.meta.env.DEV ? "no-store" : "no-cache" },
  );

  if (!response.ok) {
    throw new Error(`Comparison ${bundle} payload request failed with ${response.status}`);
  }

  const payload = (await response.json()) as ComparisonInstancesPayload;
  if (
    payload.comparisonId !== comparisonId ||
    payload.bundle !== bundle ||
    !Array.isArray(payload.variants) ||
    !payload.variants.every((variant) => (variant.label === "A" || variant.label === "B") && Array.isArray(variant.instances))
  ) {
    throw new Error(`Comparison ${bundle} payload does not match the expected data shape`);
  }

  return payload;
}
