export interface CitationPaper {
  id: string;
  title: string;
  authors: string[];
  published: string;
  primaryCategory?: string;
  paperUrl: string;
  doi?: string;
  journalReference?: string;
}

function publicationYear(value: string): string {
  const match = /^(\d{4})/.exec(value);
  return match?.[1] ?? 'n.d.';
}

function authorList(authors: string[]): string {
  if (authors.length === 0) return 'Unknown author';
  if (authors.length === 1) return authors[0];
  if (authors.length === 2) return `${authors[0]} and ${authors[1]}`;
  return `${authors.slice(0, -1).join(', ')}, and ${authors.at(-1)}`;
}

function bibtexEscapePlain(value: string): string {
  return value
    .replace(/([{}%&_#$])/g, '\\$1')
    .replace(/\^/g, '\\textasciicircum{}')
    .replace(/~/g, '\\textasciitilde{}');
}

function bibtexEscapeTitle(value: string): string {
  const segments = value.split(/(\$[^$]*\$)/g);
  return segments.map((segment, index) => (
    index % 2 === 1 ? segment : bibtexEscapePlain(segment)
  )).join('');
}

function citationKey(paper: CitationPaper): string {
  const firstAuthor = paper.authors[0]?.split(/\s+/).at(-1) ?? 'unknown';
  const normalized = firstAuthor.normalize('NFKD').replace(/[^a-zA-Z0-9]/g, '');
  const year = publicationYear(paper.published).replace(/\D/g, '') || 'nd';
  return `${normalized || 'unknown'}${year}${paper.id.replace('.', '')}`;
}

export function plainCitation(paper: CitationPaper): string {
  return `${authorList(paper.authors)} (${publicationYear(paper.published)}). ${paper.title}. arXiv:${paper.id}. ${paper.paperUrl}`;
}

export function bibtexCitation(paper: CitationPaper): string {
  const fields = [
    `  title = {${bibtexEscapeTitle(paper.title)}}`,
    `  author = {${paper.authors.length ? paper.authors.map(bibtexEscapePlain).join(' and ') : 'Unknown author'}}`,
    `  year = {${publicationYear(paper.published)}}`,
    `  eprint = {${paper.id}}`,
    '  archivePrefix = {arXiv}',
    paper.primaryCategory && `  primaryClass = {${bibtexEscapePlain(paper.primaryCategory)}}`,
    paper.doi && `  doi = {${bibtexEscapePlain(paper.doi)}}`,
    paper.journalReference && `  journal = {${bibtexEscapePlain(paper.journalReference)}}`,
    `  url = {${bibtexEscapePlain(paper.paperUrl)}}`,
  ].filter(Boolean);
  return `@misc{${citationKey(paper)},\n${fields.join(',\n')}\n}\n`;
}

// RIS is a line-oriented format: every tag occupies its own physical line, so a
// value may never contain a raw line break. Unicode is preserved as UTF-8.
function risValue(value: string): string {
  return value.replace(/[\r\n]+/g, ' ').replace(/\s+/g, ' ').trim();
}

export function risCitation(paper: CitationPaper): string {
  const authors = paper.authors.length ? paper.authors : ['Unknown author'];
  const lines = [
    'TY  - GEN',
    `ID  - ${risValue(paper.id)}`,
    `TI  - ${risValue(paper.title)}`,
    ...authors.map((author) => `AU  - ${risValue(author)}`),
    `PY  - ${publicationYear(paper.published)}`,
  ];
  if (paper.primaryCategory) lines.push(`KW  - ${risValue(paper.primaryCategory)}`);
  if (paper.doi) lines.push(`DO  - ${risValue(paper.doi)}`);
  if (paper.journalReference) lines.push(`JF  - ${risValue(paper.journalReference)}`);
  lines.push(`UR  - ${risValue(paper.paperUrl)}`);
  lines.push(`N1  - arXiv:${risValue(paper.id)}`);
  lines.push('ER  - ');
  return `${lines.join('\r\n')}\r\n`;
}

// RFC 4180: wrap every field in quotes and double any embedded quote.
function csvField(value: string): string {
  return `"${value.replace(/"/g, '""')}"`;
}

const CSV_COLUMNS = [
  'id',
  'title',
  'authors',
  'published',
  'primary_category',
  'paper_url',
  'doi',
  'journal_reference',
] as const;

function csvRow(paper: CitationPaper): string {
  return [
    paper.id,
    paper.title,
    paper.authors.join('; '),
    paper.published,
    paper.primaryCategory ?? '',
    paper.paperUrl,
    paper.doi ?? '',
    paper.journalReference ?? '',
  ].map((value) => csvField(value.replace(/[\r\n]+/g, ' '))).join(',');
}

// Escape the inline Markdown control characters that would otherwise change how
// surrounding text is parsed; line breaks are collapsed to keep one item per row.
function markdownInline(value: string): string {
  return value
    .replace(/[\r\n]+/g, ' ')
    .replace(/([\\`*_[\]|<>])/g, '\\$1')
    .trim();
}

export function bibtexExport(papers: CitationPaper[]): string {
  return papers.map(bibtexCitation).join('\n');
}

export function risExport(papers: CitationPaper[]): string {
  return papers.map(risCitation).join('\r\n');
}

export function csvExport(papers: CitationPaper[]): string {
  return `${[CSV_COLUMNS.join(','), ...papers.map(csvRow)].join('\r\n')}\r\n`;
}

export function markdownExport(papers: CitationPaper[]): string {
  const lines = [`# Exported papers (${papers.length})`, ''];
  for (const paper of papers) {
    const authors = paper.authors.length
      ? paper.authors.map(markdownInline).join(', ')
      : 'Unknown author';
    lines.push(
      `- **${markdownInline(paper.title)}** — ${authors} `
      + `(${publicationYear(paper.published)}). `
      + `[arXiv:${markdownInline(paper.id)}](${paper.paperUrl})`,
    );
  }
  return `${lines.join('\n')}\n`;
}

export interface ExportFormat {
  label: string;
  extension: string;
  mimeType: string;
  render(papers: CitationPaper[]): string;
}

export const EXPORT_FORMATS: Record<'bibtex' | 'ris' | 'csv' | 'markdown', ExportFormat> = {
  bibtex: {
    label: 'BibTeX',
    extension: 'bib',
    mimeType: 'application/x-bibtex;charset=utf-8',
    render: bibtexExport,
  },
  ris: {
    label: 'RIS',
    extension: 'ris',
    mimeType: 'application/x-research-info-systems;charset=utf-8',
    render: risExport,
  },
  csv: {
    label: 'CSV',
    extension: 'csv',
    mimeType: 'text/csv;charset=utf-8',
    render: csvExport,
  },
  markdown: {
    label: 'Markdown',
    extension: 'md',
    mimeType: 'text/markdown;charset=utf-8',
    render: markdownExport,
  },
};