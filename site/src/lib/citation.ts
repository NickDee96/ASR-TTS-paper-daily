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