// SPDX-License-Identifier: Apache-2.0

import type { InstanceRow } from "@/components/comparison/types";

export type CompactInstanceName = {
  fullId: string;
  shortId: string;
  repo: string;
  pullRequestUrl?: string;
  relatedIssuesUrl?: string;
  pullRequestNumber?: string;
  taskType?: string;
  taskTypeParts: string[];
};

export function compactInstanceName(row: InstanceRow): CompactInstanceName {
  const instanceParts = row.instanceId.split("__");
  const shortId = instanceParts[instanceParts.length - 1] || row.instanceId;
  const taskTypeParts = instanceParts.length >= 4 ? instanceParts.slice(2, -1).filter(Boolean) : [];
  const repo = repositoryLabelFromOriginalId(row.originalInstanceId);
  const githubRef = githubReferenceFromOriginalId(row.originalInstanceId);

  return {
    fullId: row.originalInstanceId ? `${row.instanceId} / ${row.originalInstanceId}` : row.instanceId,
    shortId,
    repo,
    pullRequestUrl: githubRef?.pullRequestUrl,
    relatedIssuesUrl: githubRef?.relatedIssuesUrl,
    pullRequestNumber: githubRef?.number,
    taskType: taskTypeParts.length > 0 ? taskTypeParts.join(" / ") : undefined,
    taskTypeParts,
  };
}

export function compactBenchLabel(bench: string): string {
  if (bench === "SWE-Bench-Verified") return "Verified";
  if (bench === "SWE-Bench-Pro") return "Pro";
  if (bench === "Multi-SWE-Bench") return "Multi";
  if (bench === "SWE-PolyBench") return "PolyBench";
  return bench;
}

function repositoryLabelFromOriginalId(originalInstanceId: string | null | undefined): string {
  const parsed = parseGithubIssueOriginalId(originalInstanceId);
  if (!parsed) {
    return "Unknown repo";
  }

  return `${parsed.owner}/${parsed.repo}`;
}

function githubReferenceFromOriginalId(
  originalInstanceId: string | null | undefined,
): { number: string; pullRequestUrl: string; relatedIssuesUrl: string } | undefined {
  const parsed = parseGithubIssueOriginalId(originalInstanceId);
  if (!parsed?.issueNumber) return undefined;
  const encodedQuery = encodeURIComponent(`is:issue ${parsed.issueNumber}`);
  return {
    number: parsed.issueNumber,
    pullRequestUrl: `https://github.com/${parsed.owner}/${parsed.repo}/pull/${parsed.issueNumber}`,
    relatedIssuesUrl: `https://github.com/${parsed.owner}/${parsed.repo}/issues?q=${encodedQuery}`,
  };
}

function parseGithubIssueOriginalId(originalInstanceId: string | null | undefined): { owner: string; repo: string; issueNumber?: string } | null {
  const match = String(originalInstanceId || "").match(/^(?:instance_)?([^_]+)__(.+)$/);
  if (!match) {
    return null;
  }

  const owner = match[1];
  const repoAndIssue = match[2];
  const issueMatch = repoAndIssue.match(/-(\d+)(?:-[0-9a-f]{7,40})?(?:-v[0-9a-z]+)?$/i);
  const repo = repoAndIssue
    .replace(/-[0-9a-f]{40}(?:-v[0-9a-z]+)?$/i, "")
    .replace(/-[0-9a-f]{7,12}$/i, "")
    .replace(/-v[0-9a-z]+$/i, "")
    .replace(/-\d+$/i, "");

  if (!repo) return { owner, repo: owner, issueNumber: issueMatch?.[1] };

  return { owner, repo, issueNumber: issueMatch?.[1] };
}
