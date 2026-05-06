// Fork note: Modified by Norbert Laszlo on 2026-04-17 from upstream ContextBench.
// Summary of changes: distinguish fork-specific execution completion from official Pass@1 and add execution integrity metadata.

import claudeLogo from "@/assets/claude.svg";
import openaiLogo from "@/assets/openai.svg";

import type {
  ComparisonCard,
  ComparisonData,
  ComparisonInstance,
  ComparisonInstanceDetail,
  ComparisonInstanceDetailTraceEntry,
  ComparisonInstanceDetailVariant,
  FilterMode,
  LeaderboardRow,
} from "@/data/comparison-types";

export type {
  ComparisonCard,
  ComparisonData,
  ComparisonInstance,
  ComparisonInstanceDetail,
  ComparisonInstanceDetailTraceEntry,
  ComparisonInstanceDetailVariant,
  FilterMode,
  LeaderboardRow,
} from "@/data/comparison-types";


export const filterOrder: FilterMode[] = ["all", "claude", "codex"];

export const comparisonCards: ComparisonCard[] = [
  {
    id: "codex-bootstrap-vs-baseline",
    agent: "codex",
    icon: openaiLogo,
    title: "Bootstrap vs baseline",
    summary: "Codex retrieved wider context when the setup prompt was enabled.",
    suite: "superpowers-smoke",
    completedAt: "2026-03-24T02:29:10Z",
    taskSet: {
      count: 24,
      benchCounts: {
        Verified: 18,
        Poly: 6,
      },
    },
    effort: "High",
    tasks: 24,
    score: "71.4%",
    variants: [
      {
        model: "gpt-5.4",
        label: "A",
        name: "Baseline",
        effort: "High",
        score: "68.2%",
        parameters: [
          { label: "Setup Prompt", value: "None" },
          { label: "Reasoning", value: "High" },
          { label: "Network", value: "Disabled" },
        ],
        results: {
          outcome: {
            success: 13,
            partialSuccess: 5,
            failures: 6,
            completedTasks: 18,
            successRate: "54.2%",
          },
          quality: {
            fileCoverage: "68.4%",
            spanCoverage: "59.1%",
            precision: "63.8%",
            editSuccess: "54.0%",
          },
          efficiency: {
            avgDuration: "11m 42s",
            totalTokens: "1.18M",
            toolCalls: "46",
            cost: "$0.92",
          },
        },
      },
      {
        model: "gpt-5.4",
        label: "B",
        name: "Bootstrap",
        effort: "High",
        score: "71.4%",
        parameters: [
          { label: "Setup Prompt", value: "Superpowers bootstrap" },
          { label: "Reasoning", value: "High" },
          { label: "Network", value: "Disabled" },
        ],
        results: {
          outcome: {
            success: 15,
            partialSuccess: 5,
            failures: 4,
            completedTasks: 20,
            successRate: "62.5%",
          },
          quality: {
            fileCoverage: "72.7%",
            spanCoverage: "63.4%",
            precision: "67.9%",
            editSuccess: "60.8%",
          },
          efficiency: {
            avgDuration: "14m 05s",
            totalTokens: "1.31M",
            toolCalls: "52",
            cost: "$1.06",
          },
        },
      },
    ],
    notes: [
      "Bootstrap runs explored more repository context before producing patches.",
      "The stronger context coverage came with longer run durations.",
    ],
  },
  {
    id: "claude-verified-slice",
    agent: "claude",
    icon: claudeLogo,
    title: "Verified slice",
    summary: "Claude completed more tasks cleanly on the verified sample slice.",
    suite: "verified-comparison",
    completedAt: "2026-03-25T14:18:00Z",
    taskSet: {
      count: 24,
      benchCounts: {
        Verified: 24,
      },
    },
    effort: "Medium",
    tasks: 24,
    score: "68.9%",
    variants: [
      {
        model: "claude-sonnet-4-6",
        label: "A",
        name: "Verified baseline",
        effort: "Medium",
        score: "67.1%",
        parameters: [
          { label: "Prompt Variant", value: "Standard" },
          { label: "Reasoning", value: "Medium" },
          { label: "Network", value: "Disabled" },
        ],
        results: {
          outcome: {
            success: 14,
            partialSuccess: 4,
            failures: 6,
            completedTasks: 18,
            successRate: "58.3%",
          },
          quality: {
            fileCoverage: "69.2%",
            spanCoverage: "61.0%",
            precision: "66.5%",
            editSuccess: "57.8%",
          },
          efficiency: {
            avgDuration: "9m 18s",
            totalTokens: "842K",
            toolCalls: "34",
            cost: "$0.54",
          },
        },
      },
      {
        model: "claude-sonnet-4-6",
        label: "B",
        name: "Verified tuned",
        effort: "High",
        score: "68.9%",
        parameters: [
          { label: "Prompt Variant", value: "Verified tuned" },
          { label: "Reasoning", value: "High" },
          { label: "Network", value: "Disabled" },
        ],
        results: {
          outcome: {
            success: 15,
            partialSuccess: 4,
            failures: 5,
            completedTasks: 19,
            successRate: "62.5%",
          },
          quality: {
            fileCoverage: "70.9%",
            spanCoverage: "62.8%",
            precision: "68.1%",
            editSuccess: "59.6%",
          },
          efficiency: {
            avgDuration: "10m 47s",
            totalTokens: "901K",
            toolCalls: "37",
            cost: "$0.61",
          },
        },
      },
    ],
    notes: [
      "Claude stayed consistent on the verified benchmark subset.",
      "The stronger completion rate came with slightly narrower retrieval breadth.",
    ],
  },
  {
    id: "codex-network-enabled",
    agent: "codex",
    icon: openaiLogo,
    title: "Network-enabled run",
    summary: "Codex improved patch quality when network access was allowed.",
    suite: "net-enabled",
    completedAt: "2026-03-27T11:42:00Z",
    taskSet: {
      count: 16,
      benchCounts: {
        Verified: 9,
        Poly: 7,
      },
    },
    effort: "High",
    tasks: 16,
    score: "74.2%",
    variants: [
      {
        model: "gpt-5.4",
        label: "A",
        name: "Offline",
        effort: "Medium",
        score: "69.5%",
        parameters: [
          { label: "Network", value: "Disabled" },
          { label: "Validation", value: "Local only" },
          { label: "Reasoning", value: "Medium" },
        ],
        results: {
          outcome: {
            success: 9,
            partialSuccess: 4,
            failures: 3,
            completedTasks: 13,
            successRate: "56.3%",
          },
          quality: {
            fileCoverage: "70.1%",
            spanCoverage: "60.6%",
            precision: "65.4%",
            editSuccess: "58.2%",
          },
          efficiency: {
            avgDuration: "8m 55s",
            totalTokens: "716K",
            toolCalls: "29",
            cost: "$0.47",
          },
        },
      },
      {
        model: "gpt-5.4",
        label: "B",
        name: "Network enabled",
        effort: "High",
        score: "74.2%",
        parameters: [
          { label: "Network", value: "Enabled" },
          { label: "Validation", value: "Remote + local" },
          { label: "Reasoning", value: "High" },
        ],
        results: {
          outcome: {
            success: 11,
            partialSuccess: 3,
            failures: 2,
            completedTasks: 14,
            successRate: "68.8%",
          },
          quality: {
            fileCoverage: "74.8%",
            spanCoverage: "66.2%",
            precision: "70.4%",
            editSuccess: "64.1%",
          },
          efficiency: {
            avgDuration: "10m 12s",
            totalTokens: "804K",
            toolCalls: "35",
            cost: "$0.59",
          },
        },
      },
    ],
    notes: [
      "Allowing network access improved validation and patch confidence.",
      "This setup also increased environmental variability across runs.",
    ],
  },
  {
    id: "claude-prompt-variation",
    agent: "claude",
    icon: claudeLogo,
    title: "Prompt variation",
    summary: "Claude was steadier across prompt variants but slower overall.",
    suite: "prompt-variation",
    completedAt: "2026-03-28T09:05:00Z",
    taskSet: {
      count: 16,
      benchCounts: {
        Verified: 12,
        Poly: 4,
      },
    },
    effort: "Low",
    tasks: 16,
    score: "66.1%",
    variants: [
      {
        model: "claude-sonnet-4-6",
        label: "A",
        name: "Prompt A",
        effort: "Low",
        score: "65.4%",
        parameters: [
          { label: "Prompt Style", value: "Direct" },
          { label: "Reasoning", value: "Low" },
          { label: "Network", value: "Disabled" },
        ],
        results: {
          outcome: {
            success: 8,
            partialSuccess: 4,
            failures: 4,
            completedTasks: 12,
            successRate: "50.0%",
          },
          quality: {
            fileCoverage: "64.3%",
            spanCoverage: "57.6%",
            precision: "61.7%",
            editSuccess: "55.1%",
          },
          efficiency: {
            avgDuration: "7m 44s",
            totalTokens: "655K",
            toolCalls: "26",
            cost: "$0.39",
          },
        },
      },
      {
        model: "claude-sonnet-4-6",
        label: "B",
        name: "Prompt B",
        effort: "Medium",
        score: "66.1%",
        parameters: [
          { label: "Prompt Style", value: "Structured" },
          { label: "Reasoning", value: "Medium" },
          { label: "Network", value: "Disabled" },
        ],
        results: {
          outcome: {
            success: 9,
            partialSuccess: 4,
            failures: 3,
            completedTasks: 13,
            successRate: "56.3%",
          },
          quality: {
            fileCoverage: "65.9%",
            spanCoverage: "59.4%",
            precision: "63.1%",
            editSuccess: "56.8%",
          },
          efficiency: {
            avgDuration: "8m 31s",
            totalTokens: "701K",
            toolCalls: "30",
            cost: "$0.44",
          },
        },
      },
    ],
    notes: [
      "Prompt wording had less effect on Claude than on Codex in this slice.",
      "The tradeoff was lower peak score and slower run completion.",
    ],
  },
];

function fallbackModelName(agent: "claude" | "codex"): string {
  return agent === "codex" ? "Codex" : "Claude Code";
}

function variantModel(variant: ComparisonCard["variants"][number]): string | undefined {
  return variant.model ?? variant.parameters.find((parameter) => parameter.label.toLowerCase() === "model")?.value;
}

export function completedRunsForOutcome(outcome: ComparisonCard["variants"][number]["results"]["outcome"]): number {
  return outcome.completedRuns ?? outcome.success ?? 0;
}

export function partialRunsForOutcome(outcome: ComparisonCard["variants"][number]["results"]["outcome"]): number {
  return outcome.partialRuns ?? outcome.partialSuccess ?? 0;
}

export function finishedRunsForOutcome(outcome: ComparisonCard["variants"][number]["results"]["outcome"]): number {
  return outcome.finishedRuns ?? outcome.completedTasks ?? (completedRunsForOutcome(outcome) + partialRunsForOutcome(outcome));
}

export function completedRunRateForOutcome(outcome: ComparisonCard["variants"][number]["results"]["outcome"]): string | undefined {
  return outcome.completedRunRate ?? outcome.successRate ?? outcome.passAt1;
}

export const leaderboardRows: LeaderboardRow[] = comparisonCards.flatMap((item) =>
  item.variants.map((variant) => ({
    agent: item.agent,
    icon: item.icon,
    model: variantModel(variant) ?? fallbackModelName(item.agent),
    suite: variant.name,
    effort: variant.effort,
    tasks: item.tasks,
    completedRunRate: completedRunRateForOutcome(variant.results.outcome),
    officialPassAt1: variant.results.outcome.officialPassAt1 ?? null,
    contextF1: variant.contextF1 ?? variant.score,
    score: variant.score,
  })),
);

export const placeholderComparisonData: ComparisonData = {
  filterOrder,
  comparisonCards,
  leaderboardRows,
};

function resolveIcon(agent: "claude" | "codex"): string {
  return agent === "claude" ? claudeLogo : openaiLogo;
}

function resolveVariantLabel(label: unknown, index: number): "A" | "B" {
  if (label === "A" || label === "B") {
    return label;
  }
  return index === 0 ? "A" : "B";
}

export function withResolvedIcons(data: ComparisonData): ComparisonData {
  return {
    filterOrder: data.filterOrder,
    comparisonCards: data.comparisonCards.map((item) => ({
      ...item,
      icon: item.icon || resolveIcon(item.agent),
      variants: item.variants.map((variant, index) => ({
        ...variant,
        label: resolveVariantLabel(variant.label, index),
      })),
    })),
    leaderboardRows: data.leaderboardRows.map((row) => ({
      ...row,
      icon: row.icon || resolveIcon(row.agent),
    })),
  };
}

export function cardsForFilter(data: ComparisonData, filter: FilterMode): ComparisonCard[] {
  if (filter === "all") {
    return data.comparisonCards;
  }

  return data.comparisonCards.filter((item) => item.agent === filter);
}

export function findComparisonById(data: ComparisonData, id: string): ComparisonCard | undefined {
  return data.comparisonCards.find((item) => item.id === id);
}
