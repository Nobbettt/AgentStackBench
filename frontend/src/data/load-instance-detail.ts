
import type { ComparisonInstanceDetail } from "@/data/comparisons";

export async function loadInstanceDetail(
  comparisonId: string,
  instanceId: string,
): Promise<ComparisonInstanceDetail | null> {
  const response = await fetch(
    `${import.meta.env.BASE_URL}instances/${encodeURIComponent(comparisonId)}/${encodeURIComponent(instanceId)}.json`,
    { cache: "no-store" },
  );

  if (response.status === 404) {
    return null;
  }

  if (!response.ok) {
    throw new Error(`Instance detail request failed with ${response.status}`);
  }

  return (await response.json()) as ComparisonInstanceDetail;
}
