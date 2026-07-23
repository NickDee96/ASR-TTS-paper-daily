export type PaperStatus = 'new' | 'revised';

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
}