import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { archiveSummary, previewPapers, topicCounts } from './preview-papers';
import type { PaperPreview, PaperRecord, PaperStatus } from '../types/paper';

interface CanonicalAuthor {
  name?: string;
  affiliations?: string[];
}

interface CanonicalMatch {
  topic?: string;
  score?: number | null;
  threshold?: number | null;
  matched_title_terms?: string[];
  matched_abstract_terms?: string[];
  matched_all_terms?: string[];
  evidence_complete?: boolean;
}

interface CanonicalPaper {
  id: string;
  title?: string | null;
  abstract?: string | null;
  authors?: CanonicalAuthor[];
  published?: string | null;
  updated?: string | null;
  topics?: string[];
  arxiv_categories?: string[];
  primary_category?: string | null;
  record_status?: 'complete' | 'partial' | 'unavailable';
  classification?: {
    classifier_version?: string;
    matches?: CanonicalMatch[];
  };
  source?: {
    origin?: string;
    fetched_at?: string | null;
    arxiv_version?: string | null;
  };
  code?: {
    status?: string;
    url?: string | null;
    source?: string | null;
    confidence?: number | null;
    checked_at?: string | null;
    evidence?: string[];
  };
  links?: {
    abstract?: string | null;
    pdf?: string | null;
  };
  doi?: string | null;
  journal_reference?: string | null;
}

const canonicalPath = resolve(process.cwd(), '.generated', 'canonical.json');
const siteCardPath = resolve(process.cwd(), 'public', 'data', 'site-card.json');
let cachedPapers: PaperRecord[] | undefined;

export interface ArchiveSummary {
  uniquePapers: number;
  topicAssignments: number;
  verifiedCode: number;
  updatedAt: string;
  topics: Record<string, number>;
}

function statusFor(record: CanonicalPaper): PaperStatus {
  if (record.published && record.updated && record.updated > record.published) {
    return 'revised';
  }
  return 'new';
}

function safeHttpUrl(value: string | null | undefined): string | undefined {
  if (!value) return undefined;
  try {
    const parsed = new URL(value);
    return parsed.protocol === 'https:' || parsed.protocol === 'http:'
      ? parsed.toString()
      : undefined;
  } catch {
    return undefined;
  }
}

function toPaperRecord(record: CanonicalPaper): PaperRecord {
  const authorDetails = (record.authors ?? [])
    .map((author) => ({
      name: author.name?.trim() ?? '',
      affiliations: (author.affiliations ?? []).filter(Boolean),
    }))
    .filter((author) => Boolean(author.name));
  const authors = authorDetails.map((author) => author.name);
  const topicEvidence = (record.classification?.matches ?? []).map((match) => ({
    topic: match.topic ?? 'Unknown topic',
    score: match.score ?? undefined,
    threshold: match.threshold ?? undefined,
    matchedTitleTerms: match.matched_title_terms ?? [],
    matchedAbstractTerms: match.matched_abstract_terms ?? [],
    matchedRequiredTerms: match.matched_all_terms ?? [],
    evidenceComplete: match.evidence_complete ?? false,
  }));
  const matchedTerms = topicEvidence
    .flatMap((match) => [
      ...match.matchedTitleTerms,
      ...match.matchedAbstractTerms,
      ...match.matchedRequiredTerms,
    ])
    .filter((term, index, terms) => term && terms.indexOf(term) === index);
  const paperUrl = safeHttpUrl(record.links?.abstract) || `https://arxiv.org/abs/${record.id}`;
  const codeUrl = safeHttpUrl(record.code?.url);
  const codeStatus = record.code?.status === 'verified'
    || record.code?.status === 'candidate'
    || record.code?.status === 'unavailable'
    ? record.code.status
    : 'missing';
  return {
    id: record.id,
    title: record.title || `arXiv ${record.id}`,
    titleAvailable: Boolean(record.title),
    abstract: record.abstract || 'Abstract unavailable while historical metadata is backfilled.',
    authors,
    published: record.published || 'Unknown',
    updated: record.updated || record.published || 'Unknown',
    topics: record.topics ?? [],
    categories: record.arxiv_categories ?? [],
    status: statusFor(record),
    codeUrl: record.code?.status === 'verified' ? codeUrl : undefined,
    paperUrl,
    pdfUrl: safeHttpUrl(record.links?.pdf) || paperUrl.replace('/abs/', '/pdf/'),
    recordStatus: record.record_status ?? 'partial',
    authorDetails,
    primaryCategory: record.primary_category ?? undefined,
    classifierVersion: record.classification?.classifier_version ?? 'unknown',
    matchedTerms,
    topicEvidence,
    sourceOrigin: record.source?.origin ?? 'unknown',
    sourceFetchedAt: record.source?.fetched_at ?? undefined,
    arxivVersion: record.source?.arxiv_version ?? undefined,
    codeStatus,
    code: {
      status: codeStatus,
      url: codeUrl,
      source: record.code?.source ?? undefined,
      confidence: record.code?.confidence ?? undefined,
      checkedAt: record.code?.checked_at ?? undefined,
      evidence: record.code?.evidence ?? [],
    },
    doi: record.doi ?? undefined,
    journalReference: record.journal_reference ?? undefined,
  };
}

function previewRecords(): PaperRecord[] {
  return previewPapers.map((paper: PaperPreview) => ({
    ...paper,
    titleAvailable: true,
    recordStatus: 'complete',
    authorDetails: paper.authors.map((name) => ({ name, affiliations: [] })),
    primaryCategory: paper.categories[0],
    classifierVersion: 'preview',
    matchedTerms: [],
    topicEvidence: paper.topics.map((topic) => ({
      topic,
      matchedTitleTerms: [],
      matchedAbstractTerms: [],
      matchedRequiredTerms: [],
      evidenceComplete: false,
    })),
    sourceOrigin: 'preview',
    codeStatus: paper.codeUrl ? 'verified' : 'missing',
    code: {
      status: paper.codeUrl ? 'verified' : 'missing',
      url: paper.codeUrl,
      evidence: paper.codeUrl ? ['Preview record code link'] : [],
    },
  }));
}

export function loadPapers(): PaperRecord[] {
  if (cachedPapers) return cachedPapers;
  if (!existsSync(canonicalPath)) {
    cachedPapers = previewRecords();
    return cachedPapers;
  }
  const document = JSON.parse(readFileSync(canonicalPath, 'utf8')) as Record<string, CanonicalPaper>;
  cachedPapers = Object.values(document)
    .map(toPaperRecord)
    .sort((left, right) => {
      const dateOrder = right.updated.localeCompare(left.updated);
      return dateOrder || right.id.localeCompare(left.id);
    });
  return cachedPapers;
}

export function loadArchiveSummary(): ArchiveSummary {
  if (!existsSync(siteCardPath)) {
    return {
      uniquePapers: archiveSummary.uniquePapers,
      topicAssignments: archiveSummary.topicAssignments,
      verifiedCode: 0,
      updatedAt: archiveSummary.updatedAt,
      topics: { ...topicCounts },
    };
  }
  const document = JSON.parse(readFileSync(siteCardPath, 'utf8')) as {
    unique_papers: number;
    topic_assignments: number;
    verified_code: number;
    updated_at: string;
    topics: Record<string, number>;
  };
  return {
    uniquePapers: document.unique_papers,
    topicAssignments: document.topic_assignments,
    verifiedCode: document.verified_code,
    updatedAt: document.updated_at,
    topics: document.topics,
  };
}