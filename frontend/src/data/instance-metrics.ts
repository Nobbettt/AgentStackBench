// SPDX-License-Identifier: Apache-2.0

import type { ComparisonInstance } from "@/data/comparisons";

export type ContextLevel = "file" | "symbol" | "span" | "line";
export type ContextMeasure = "f1" | "recall" | "precision";

export const CONTEXT_LEVELS: ContextLevel[] = ["file", "span", "line", "symbol"];

export function mean(values: number[]): number | null {
  return values.length > 0 ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

export function f1(coverage: number, precision: number): number {
  const denominator = coverage + precision;
  return denominator === 0 ? 0 : (2 * coverage * precision) / denominator;
}

export function hasValidEvaluation(instance: ComparisonInstance): boolean {
  return !instance.artifacts?.evaluationStatus || instance.artifacts.evaluationStatus === "valid";
}

// Coverage/precision are undefined when an instance has no gold context at a
// granularity (a 0/0 convention would score 1.0 and inflate macro averages),
// so empty-gold instances yield null and are excluded from means.
export function coveragePrecisionOrNull(
  predSize: number,
  goldSize: number,
  intersection: number,
): { coverage: number; precision: number } | null {
  if (goldSize <= 0) return null;
  return {
    coverage: intersection / goldSize,
    precision: predSize > 0 ? intersection / predSize : 1,
  };
}

export function contextLevelMetric(
  instance: ComparisonInstance,
  level: ContextLevel,
  measure: ContextMeasure,
): number | null {
  if (!hasValidEvaluation(instance)) return null;
  const metric = instance.quality[level];
  const values = coveragePrecisionOrNull(metric.predSize, metric.goldSize, metric.intersection);
  if (!values) return null;
  if (measure === "recall") return values.coverage;
  if (measure === "precision") return values.precision;
  return f1(values.coverage, values.precision);
}

// Macro context aggregate per instance: mean over the granularities that are
// defined for this instance (null when none are).
export function instanceContextAggregate(
  instance: ComparisonInstance,
  measure: ContextMeasure,
): number | null {
  return mean(
    CONTEXT_LEVELS
      .map((level) => contextLevelMetric(instance, level, measure))
      .filter((value): value is number => value !== null),
  );
}

export function terminalTrajectoryCoverage(instance: ComparisonInstance, level: ContextLevel): number | null {
  if (!hasValidEvaluation(instance)) return null;
  if (!instance.evaluatedTrajectory) return null;
  const steps = (instance.evaluatedTrajectory.steps ?? []).filter((step) => !step.isSkillRead);
  const values = steps
    .map((step) => step.coverage[level])
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  if (values.length === 0) return 0;
  return Math.min(Math.max(values[values.length - 1], 0), 1);
}

export function instanceTrajectoryGoldFound(instance: ComparisonInstance): number | null {
  return mean(
    CONTEXT_LEVELS
      .map((level) => terminalTrajectoryCoverage(instance, level))
      .filter((value): value is number => value !== null),
  );
}
