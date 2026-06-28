
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

export type PooledContextLevel = {
  recall?: string;
  precision?: string;
  f1?: string;
  intersection?: number;
  goldSize?: number;
  predSize?: number;
};

// `n` is the number of instances with gold context at this granularity;
// instances without gold are excluded from the macro averages.
export type ContextLevelSummary = {
  recall?: string;
  precision?: string;
  f1?: string;
  n?: number;
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
    predictedContextPathDiagnostics?: {
      missingFinalPaths?: string[];
      missingTrajectoryPaths?: string[];
      missingFinalPathCount?: number;
      missingTrajectoryPathCount?: number;
    };
    verificationQuality?: VerificationQuality;
    regressionTest?: RegressionTestDiagnostic;
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
  evaluatedTrajectory?: {
    steps?: Array<{
      step: number;
      isSkillRead?: boolean;
      coverage: {
        file?: number;
        symbol?: number;
        span?: number;
        line?: number;
      };
    }>;
  };
  fixOverlap?: {
    vsGold?: PatchOverlapVsGold;
  };
  resources: {
    durationMs?: number | null;
    durationStatus?: "available" | "unavailable";
    durationUnavailableReason?: "missing_duration" | "timed_out" | "exceeds_configured_timeout";
    rawDurationMs?: number | null;
    totalTokens?: number | null;
    inputTokens?: number | null;
    outputTokens?: number | null;
    cachedInputTokens?: number | null;
    nonCachedInputTokens?: number | null;
    toolCalls?: number | null;
    mcpToolCalls?: number | null;
    successfulMcpToolCalls?: number | null;
    commandExecutions?: number | null;
    readToolCalls?: number | null;
    editToolCalls?: number | null;
    rawTraceEvents?: number | null;
    rawAgentActions?: number | null;
    costUsd?: number | null;
    retryAttempts?: number;
    retried?: boolean;
    retrySuppressed?: boolean;
  };
  repositorySize?: {
    status?: "available" | "unavailable";
    reason?: string | null;
    repo?: string | null;
    commit?: string | null;
    trackedFiles?: number | null;
    lineCountStatus?: "available" | "unavailable";
    lineCountReason?: string | null;
    trackedTextLines?: number | null;
    trackedBytes?: number | null;
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
  mcp?: McpUseSummary;
};

export type VerificationQuality = {
  schemaVersion?: number;
  strongestVerification?: string | null;
  successfulRuntimeVerification?: boolean;
  successfulStaticVerification?: boolean;
  syntaxOnly?: boolean;
  dependencyBlocked?: boolean;
  environmentLimited?: boolean;
  environmentLimitationMatches?: string[];
  commandsTotal?: number | null;
  successfulCommandsTotal?: number | null;
  failedCommandsTotal?: number | null;
  commandCategories?: Record<string, number>;
  successfulCommandCategories?: Record<string, number>;
  verificationCommands?: Array<{
    command?: string;
    category?: string;
    succeeded?: boolean;
  }>;
};

export type RegressionTestDiagnostic = {
  schemaVersion?: number;
  addedRegressionTest?: boolean;
  regressionTestsRun?: boolean | null;
  addedTestFiles?: string[];
  coveringCommands?: string[];
  reason?: string | null;
};

export type McpToolCount = {
  name: string;
  calls: number;
  successfulCalls?: number;
};

export type McpToolCallDetail = {
  toolName: string;
  succeeded?: boolean;
  query?: string;
  topK?: number | null;
  resultCount?: number;
  totalCandidates?: number | null;
  topPaths?: string[];
  topSymbols?: string[];
  overlapFinalContextFiles?: string[];
  overlapPatchFiles?: string[];
  followedReturnedPaths?: string[];
  meaningful?: boolean;
};

export type McpUseSummary = {
  availableTools?: string[];
  toolCalls?: number;
  successfulToolCalls?: number;
  callsWithResults?: number;
  meaningfulCalls?: number;
  callsWithFinalContextOverlap?: number;
  callsWithPatchOverlap?: number;
  callsWithFollowupOnReturnedPath?: number;
  returnedPathCount?: number;
  instancesWithMcpCalls?: number;
  instancesWithMeaningfulMcpUse?: number;
  byTool?: McpToolCount[];
};

export type McpUseDetail = McpUseSummary & {
  calls?: McpToolCallDetail[];
};

export type ComparisonInstanceDetailTraceEntry = {
  kind: "command_execution" | "todo_list" | "file_change" | "assistant_message" | "tool_use" | "tool_result";
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
  predictedContextPathDiagnostics?: {
    missingFinalPaths?: string[];
    missingTrajectoryPaths?: string[];
    missingFinalPathCount?: number;
    missingTrajectoryPathCount?: number;
  };
  verificationQuality?: VerificationQuality;
  regressionTest?: RegressionTestDiagnostic;
  startedAt?: string;
  completedAt?: string;
  durationMs?: number | null;
  durationStatus?: "available" | "unavailable";
  durationUnavailableReason?: "missing_duration" | "timed_out" | "exceeds_configured_timeout";
  rawDurationMs?: number | null;
  retry?: {
    attempts?: number;
    maxAttempts?: number;
    retried?: boolean;
    suppressed?: boolean;
    suppressionReason?: string | null;
    events?: Array<Record<string, unknown>>;
  };
  tokenUsage?: Record<string, unknown> | null;
  traceCounters?: {
    toolCalls?: number;
    mcpToolCalls?: number;
    successfulMcpToolCalls?: number;
    commandExecutions?: number;
    readToolCalls?: number;
    editToolCalls?: number;
  };
  mcpUse?: McpUseDetail;
  persistedToolResults?: Array<{
    source_path?: string;
    artifact_path?: string | null;
    status?: string;
    size_bytes?: number | null;
    label?: string | null;
  }>;
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
      isSkillRead?: boolean;
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
        contextRecall?: string;
        contextPrecision?: string;
        trajectoryGoldFound?: string | null;
        fileF1?: string;
        symbolF1?: string;
        spanF1?: string;
        avgLineF1?: string;
        contextLevels?: {
          file?: ContextLevelSummary;
          symbol?: ContextLevelSummary;
          block?: ContextLevelSummary;
          line?: ContextLevelSummary;
        };
        pooledContextLevels?: {
          file?: PooledContextLevel;
          symbol?: PooledContextLevel;
          block?: PooledContextLevel;
          line?: PooledContextLevel;
        };
        trajectoryContextLevels?: {
          file?: {
            goldFound?: string | null;
          };
          symbol?: {
            goldFound?: string | null;
          };
          block?: {
            goldFound?: string | null;
          };
          line?: {
            goldFound?: string | null;
          };
        };
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
        excludedDurationValues?: number;
        averageSteps?: string;
        avgDuration?: string;
        avgLinesPerStep?: string;
        totalTokens?: string;
        inputTokens?: string;
        outputTokens?: string;
        cachedInputTokens?: string;
        nonCachedInputTokens?: string;
        cachedInputShare?: string | null;
        toolCalls?: string;
        mcpToolCalls?: string;
        successfulMcpToolCalls?: string;
        commandExecutions?: string;
        readToolCalls?: string;
        editToolCalls?: string;
        rawTraceEvents?: string;
        rawAgentActions?: string;
        cost?: string;
      };
      retries?: {
        totalAttempts?: number;
        retriedRuns?: number;
        suppressedRetries?: number;
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
      mcp?: McpUseSummary;
      verification?: {
        strongestVerificationCounts?: Record<string, number>;
        successfulRuntimeVerificationRuns?: number;
        syntaxOnlyRuns?: number;
        environmentLimitedRuns?: number;
        addedRegressionTestRuns?: number;
        addedRegressionTestNotRunRuns?: number;
        totalCommands?: number;
        failedCommands?: number;
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

export type ComparisonInstanceBundle = "index" | "metrics" | "trajectory";

export type PartialComparisonInstance = Partial<ComparisonInstance> & {
  instanceId: string;
  bench?: string;
  language?: string;
};

export type ComparisonInstancesPayload = {
  comparisonId: string;
  bundle?: ComparisonInstanceBundle;
  variants: Array<{
    label: "A" | "B";
    name?: string;
    instances: PartialComparisonInstance[];
  }>;
};
