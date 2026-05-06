
import { type ComparisonData, withResolvedIcons } from "@/data/comparisons";

export async function loadComparisonData(): Promise<ComparisonData> {
  const response = await fetch(`${import.meta.env.BASE_URL}comparison.json`, {
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(`comparison.json request failed with ${response.status}`);
  }

  const payload = (await response.json()) as ComparisonData;
  if (!Array.isArray(payload.filterOrder) || !Array.isArray(payload.comparisonCards) || !Array.isArray(payload.leaderboardRows)) {
    throw new Error("comparison.json does not match the expected comparison data shape");
  }

  return withResolvedIcons(payload);
}
