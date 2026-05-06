
export type Route =
  | { page: "overview" }
  | { page: "comparison"; id: string }
  | { page: "instanceDetail"; comparisonId: string; instanceId: string };

export function parseRoute(hash: string): Route {
  const normalized = hash.replace(/^#\/?/, "");
  const parts = normalized.split("/").filter(Boolean);

  if (parts[0] === "comparisons" && parts[1] && parts[2] === "instances" && parts[3]) {
    return {
      page: "instanceDetail",
      comparisonId: decodeURIComponent(parts[1]),
      instanceId: decodeURIComponent(parts.slice(3).join("/")),
    };
  }

  if (normalized.startsWith("comparisons/")) {
    const id = normalized.slice("comparisons/".length);
    if (id) {
      return { page: "comparison", id };
    }
  }

  return { page: "overview" };
}
