import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { archiveSummary, previewPapers, topicCounts } from './preview-papers';
import type { PaperPreview, PaperRecord, PaperStatus } from '../types/paper';

interface CanonicalAuthor {
  name?: string;
}

interface CanonicalMatch {
  matched_title_terms?: string[];
  matched_abstract_terms?: string[];
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
  record_status?: 'complete' | 'partial';
  classification?: {
    classifier_version?: string;
    matches?: CanonicalMatch[];
  };
  source?: {
    origin?: string;
  };
  code?: {
    status?: string;
    url?: string | null;
  };
  links?: {
    abstract?: string | null;
    pdf?: string | null;
  };
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

function toPaperRecord(record: CanonicalPaper): PaperRecord {
  const authors = (record.authors ?? [])
    .map((author) => author.name?.trim())
    .filter((name): name is string => Boolean(name));
  const matchedTerms = (record.classification?.matches ?? [])
    .flatMap((match) => [
      ...(match.matched_title_terms ?? []),
      ...(match.matched_abstract_terms ?? []),
    ])
    .filter((term, index, terms) => term && terms.indexOf(term) === index);
  const paperUrl = record.links?.abstract || `https://arxiv.org/abs/${record.id}`;
  return {
    id: record.id,
    title: record.title || `arXiv ${record.id}`,
    abstract: record.abstract || 'Abstract unavailable while historical metadata is backfilled.',
    authors,
    published: record.published || record.updated || 'Unknown',
    updated: record.updated || record.published || 'Unknown',
    topics: record.topics ?? [],
    categories: record.arxiv_categories ?? [],
    status: statusFor(record),
    codeUrl: record.code?.status === 'verified' && record.code.url ? record.code.url : undefined,
    paperUrl,
    pdfUrl: record.links?.pdf || paperUrl.replace('/abs/', '/pdf/'),
    recordStatus: record.record_status ?? 'partial',
    primaryCategory: record.primary_category ?? undefined,
    classifierVersion: record.classification?.classifier_version ?? 'unknown',
    matchedTerms,
    sourceOrigin: record.source?.origin ?? 'unknown',
    codeStatus: record.code?.status ?? 'missing',
  };
}

function previewRecords(): PaperRecord[] {
  return previewPapers.map((paper: PaperPreview) => ({
    ...paper,
    recordStatus: 'complete',
    primaryCategory: paper.categories[0],
    classifierVersion: 'preview',
    matchedTerms: [],
    sourceOrigin: 'preview',
    codeStatus: paper.codeUrl ? 'verified' : 'missing',
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