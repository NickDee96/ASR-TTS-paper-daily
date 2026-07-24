export type PaperStatus = 'new' | 'revised';

export interface PaperAuthorDetail {
  name: string;
  affiliations: string[];
}

export interface TopicMatchEvidence {
  topic: string;
  score?: number;
  threshold?: number;
  matchedTitleTerms: string[];
  matchedAbstractTerms: string[];
  matchedRequiredTerms: string[];
  evidenceComplete: boolean;
}

export interface PaperCodeDetail {
  status: 'missing' | 'candidate' | 'verified' | 'unavailable';
  url?: string;
  source?: string;
  confidence?: number;
  checkedAt?: string;
  evidence: string[];
}

export interface PaperPreview {
  id: string;
  title: string;
  abstract: string;
  authors: string[];
  published: string;
  updated: string;
  topics: string[];
  categories: string[];
  status: PaperStatus;
  codeUrl?: string;
  paperUrl: string;
  pdfUrl: string;
  titleAvailable?: boolean;
}

export interface PaperRecord extends PaperPreview {
  recordStatus: 'complete' | 'partial' | 'unavailable';
  authorDetails: PaperAuthorDetail[];
  primaryCategory?: string;
  classifierVersion: string;
  matchedTerms: string[];
  topicEvidence: TopicMatchEvidence[];
  sourceOrigin: string;
  sourceFetchedAt?: string;
  firstSeenAt?: string;
  arxivVersion?: string;
  codeStatus: string;
  code: PaperCodeDetail;
  doi?: string;
  journalReference?: string;
}