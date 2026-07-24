import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const dataRoot = resolve(process.cwd(), 'public', 'data');

export interface LatestRun {
  windowEnd: string | null;
  fetched: number;
  accepted: number;
  rejected: number;
  rejectedByReason: Record<string, number>;
  inserted: number;
  updated: number;
  unchanged: number;
  failedEnrichments: number;
  topics: Record<string, { fetched: number; accepted: number; rejected: number }>;
}

export interface ArchiveStatus {
  updatedAt: string;
  staleAfterHours: number;
  uniquePapers: number;
  topicAssignments: number;
  monthlyAdditions: Record<string, number>;
  categoryDistribution: Record<string, number>;
  verifiedCode: number;
  verifiedCodeCoveragePercent: number;
  recordStatus: Record<string, number>;
  latestRun: LatestRun | null;
}

interface StatusDocument {
  updated_at?: string;
  stale_after_hours?: number;
  unique_papers?: number;
  record_status?: Record<string, number>;
  latest_run?: {
    window_end?: string | null;
    fetched?: number;
    accepted?: number;
    rejected?: number;
    rejected_by_reason?: Record<string, number>;
    inserted?: number;
    updated?: number;
    unchanged?: number;
    failed_enrichments?: number;
    topics?: Record<string, { fetched?: number; accepted?: number; rejected?: number }>;
  } | null;
}

interface StatisticsDocument {
  updated_at?: string;
  unique_papers?: number;
  topic_assignments?: number;
  monthly_additions?: Record<string, number>;
  category_distribution?: Record<string, number>;
  verified_code?: number;
  verified_code_coverage_percent?: number;
}

function readJson<T>(name: string): T | null {
  const path = resolve(dataRoot, name);
  if (!existsSync(path)) return null;
  return JSON.parse(readFileSync(path, 'utf8')) as T;
}

function normalizeRun(run: StatusDocument['latest_run']): LatestRun | null {
  if (!run) return null;
  const topics = Object.fromEntries(
    Object.entries(run.topics ?? {}).map(([topic, counts]) => [
      topic,
      {
        fetched: counts.fetched ?? 0,
        accepted: counts.accepted ?? 0,
        rejected: counts.rejected ?? 0,
      },
    ]),
  );
  return {
    windowEnd: run.window_end ?? null,
    fetched: run.fetched ?? 0,
    accepted: run.accepted ?? 0,
    rejected: run.rejected ?? 0,
    rejectedByReason: run.rejected_by_reason ?? {},
    inserted: run.inserted ?? 0,
    updated: run.updated ?? 0,
    unchanged: run.unchanged ?? 0,
    failedEnrichments: run.failed_enrichments ?? 0,
    topics,
  };
}

export function loadArchiveStatus(): ArchiveStatus {
  const status = readJson<StatusDocument>('status.json');
  const statistics = readJson<StatisticsDocument>('statistics.json');
  return {
    updatedAt: status?.updated_at ?? statistics?.updated_at ?? new Date(0).toISOString(),
    staleAfterHours: status?.stale_after_hours ?? 36,
    uniquePapers: status?.unique_papers ?? statistics?.unique_papers ?? 0,
    topicAssignments: statistics?.topic_assignments ?? 0,
    monthlyAdditions: statistics?.monthly_additions ?? {},
    categoryDistribution: statistics?.category_distribution ?? {},
    verifiedCode: statistics?.verified_code ?? 0,
    verifiedCodeCoveragePercent: statistics?.verified_code_coverage_percent ?? 0,
    recordStatus: status?.record_status ?? {},
    latestRun: normalizeRun(status?.latest_run ?? null),
  };
}
