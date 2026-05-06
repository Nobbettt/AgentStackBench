
export type FilterMode = "all" | "claude" | "codex";

export type PatchOverlapVsGold = {
  status: string;
  reason?: string | null;
  recall?: number | null;
  precision?: number | null;
  f1?: number | null;
  intersection?: number;
  goldSize?: number;
  predSize?: number;
};

export type PatchOverlapSummary = {
  status: string;
  reason?: string | null;
  recall?: string | null;
  precision?: string | null;
  f1?: string | null;
  availableInstances?: number;
  unavailableInstances?: number;
};

export type PatchOverlapPair = {
  status: string;
  reason?: string | null;
  leftLabel: string;
  rightLabel: string;
  leftCoveredByRight?: string | number | null;
  rightCoveredByLeft?: string | number | null;
  f1?: string | number | null;
  intersection?: number;
  leftSize?: number;
  rightSize?: number;
  availableInstances?: number;
  unavailableInstances?: number;
};

export type ComparisonInstance = {
  instanceId: string;
  originalInstanceId?: string | null;
  bench: string;
  language: string;
  outcome: {
    status: string;
  };
  artifacts?: {
    hasModelPatch?: boolean;
    hasPrediction?: boolean;
    evaluationStatus?: "valid" | "error" | "missing";
    resolutionStatus?: "resolved" | "unresolved" | "error" | "missing";
  };
  quality: {
    file: {
      intersection: number;
      goldSize: number;
      predSize: number;
    };
    symbol: {
      intersection: number;
      goldSize: number;
      predSize: number;
    };
    span: {
      intersection: number;
      goldSize: number;
      predSize: number;
    };
    line: {
      intersection: number;
      goldSize: number;
      predSize: number;
    };
  };
  trajectory: {
    efficiency?: number | null;
    redundancy?: number | null;
    usageDrop?: number | null;
    steps?: number | null;
    linesPerStep?: number | null;
  };
  fixOverlap?: {
    vsGold?: PatchOverlapVsGold;
  };
  resources: {
    durationMs?: number | null;
    totalTokens?: number | null;
    toolCalls?: number | null;
    costUsd?: number | null;
  };
  skills?: {
    totalInvocations?: number;
    byType?: Array<{
      name: string;
      count: number;
    }>;
  };
  tools?: {
    totalInvocations?: number;
    byType?: Array<{
      name: string;
      count: number;
    }>;
  };
};

export type ComparisonInstanceDetailTraceEntry = {
  kind: "command_execution" | "todo_list" | "file_change" | "assistant_message";
  status?: string;
  command?: string;
  output?: string;
  exitCode?: number | null;
  text?: string;
  payload?: Record<string, unknown>;
};

export type ComparisonInstanceDetailVariant = {
  label: "A" | "B";
  name: string;
  model?: string;
  effort?: string;
  status?: string;
  evaluationStatus?: string;
  startedAt?: string;
  completedAt?: string;
  durationMs?: number;
  tokenUsage?: Record<string, unknown> | null;
  modelPatch?: string;
  finalOutput?: {
    status?: string;
    finalAnswer?: string;
    notes?: string;
    retrievedContextFiles?: string[];
    retrievedContextSpans?: Array<{
      file: string;
      start: number;
      end: number;
    }>;
    retrievedContextSymbols?: Array<{
      file: string;
      name: string;
    }>;
  };
  predTrajectory?: {
    predSteps?: Array<{
      files?: string[];
      spans?: Record<string, Array<{ start: number; end: number }>>;
      symbols?: Record<string, string[]>;
    }>;
    predFiles?: string[];
    predSpans?: Record<string, Array<{ start: number; end: number }>>;
    predSymbols?: Record<string, string[]>;
  };
  evaluatedTrajectory?: {
    steps?: Array<{
      step: number;
      coverage: {
        file?: number;
        symbol?: number;
        span?: number;
        line?: number;
      };
    }>;
    aucCoverage?: Record<string, number>;
    redundancy?: Record<string, number>;
  };
  fixOverlap?: {
    vsGold?: PatchOverlapVsGold;
  };
  traceEntries?: ComparisonInstanceDetailTraceEntry[];
};

export type ComparisonInstanceDetail = {
  comparisonId: string;
  instanceId: string;
  originalInstanceId?: string | null;
  bench: string;
  language: string;
  variants: ComparisonInstanceDetailVariant[];
  fixOverlapBetweenVariants?: PatchOverlapPair;
};

export type ComparisonCard = {
  id: string;
  agent: "claude" | "codex";
  icon: string;
  title: string;
  summary: string;
  suite: string;
  startedAt?: string;
  completedAt?: string;
  taskSet?: {
    count?: number;
    hash?: string;
    benchCounts?: Record<string, number>;
    sourceDatasetCount?: number;
    selectionKind?: string;
  };
  effort: "Low" | "Medium" | "High";
  tasks: number;
  contextF1?: string;
  score?: string;
  variants: Array<{
    slug?: string;
    model?: string;
    label: "A" | "B";
    name: string;
    effort: "Low" | "Medium" | "High";
    contextF1?: string;
    score?: string;
    parameters: Array<{
      label: string;
      value: string;
    }>;
    results: {
      outcome: {
        completedRuns?: number;
        partialRuns?: number;
        failures: number;
        finishedRuns?: number;
        expectedTasks?: number;
        attemptedTasks?: number;
        completedRunRate?: string;
        officialPassAt1?: string | null;
        officialPassAt1OnEvaluated?: string | null;
        metricType?: string;
        comparableToOfficialLeaderboard?: boolean;
        success?: number;
        partialSuccess?: number;
        completedTasks?: number;
        passAt1?: string;
        successRate?: string;
      };
      integrity?: {
        patchProducingRuns?: number;
        convertedPredictions?: number;
        validEvaluations?: number;
        resolvedTasks?: number;
        patchProductionRate?: string;
        convertedPredictionRate?: string;
        validEvaluationRate?: string;
      };
      quality: {
        contextF1?: string;
        fileF1?: string;
        symbolF1?: string;
        spanF1?: string;
        avgLineF1?: string;
        fixOverlapVsGold?: PatchOverlapSummary;
        fileCoverage?: string;
        spanCoverage?: string;
        precision?: string;
        editSuccess?: string;
      };
      efficiency: {
        efficiency?: string;
        redundancy?: string;
        usageDrop?: string;
        averageDuration?: string;
        averageSteps?: string;
        avgDuration?: string;
        avgLinesPerStep?: string;
        totalTokens?: string;
        toolCalls?: string;
        cost?: string;
      };
      skills?: {
        averageInvocationsPerRun?: number;
        totalInvocations?: number;
        byType?: Array<{
          name: string;
          averagePerRun: number;
        }>;
      };
      tools?: {
        averageInvocationsPerRun?: number;
        totalInvocations?: number;
        byType?: Array<{
          name: string;
          averagePerRun: number;
        }>;
      };
    };
    instances?: ComparisonInstance[];
  }>;
  fixOverlapBetweenVariants?: PatchOverlapPair;
  notes: string[];
};

export type LeaderboardRow = {
  agent: "claude" | "codex";
  icon: string;
  model: string;
  suite: string;
  effort: "Low" | "Medium" | "High";
  tasks: number;
  completedRunRate?: string;
  officialPassAt1?: string | null;
  passAt1?: string;
  contextF1?: string;
  score?: string;
};

export type ComparisonData = {
  filterOrder: FilterMode[];
  comparisonCards: ComparisonCard[];
  leaderboardRows: LeaderboardRow[];
};
