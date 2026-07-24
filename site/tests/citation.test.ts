import assert from 'node:assert/strict';
import test from 'node:test';
import {
  bibtexCitation,
  bibtexExport,
  csvExport,
  markdownExport,
  plainCitation,
  risCitation,
  risExport,
} from '../src/lib/citation.ts';
import type { CitationPaper } from '../src/lib/citation.ts';

const paper = {
  id: '2607.00001',
  title: 'Speech & Language {Models}',
  authors: ['Māia Example', 'Alex Researcher'],
  published: '2026-07-01',
  primaryCategory: 'cs.CL',
  paperUrl: 'https://arxiv.org/abs/2607.00001',
  doi: '10.1000/example_1',
  journalReference: 'Speech & Language 42',
};

// A brace-aware BibTeX reader: proves the export decodes into structured entries
// with balanced braces rather than merely matching substrings.
function parseBibtex(text: string): Array<{ type: string; key: string; fields: Record<string, string> }> {
  const entries: Array<{ type: string; key: string; fields: Record<string, string> }> = [];
  let index = 0;
  while (index < text.length) {
    if (text[index] !== '@') { index += 1; continue; }
    const header = /^@(\w+)\{([^,]+),/.exec(text.slice(index));
    if (!header) throw new Error('Invalid BibTeX entry header');
    const entry = { type: header[1], key: header[2].trim(), fields: {} as Record<string, string> };
    index += header[0].length;
    while (index < text.length && text[index] !== '@') {
      const field = /^\s*(\w+)\s*=\s*\{/.exec(text.slice(index));
      if (!field) {
        if (text[index] === '}') { index += 1; break; }
        index += 1;
        continue;
      }
      index += field[0].length;
      let depth = 1;
      let value = '';
      while (index < text.length && depth > 0) {
        const character = text[index];
        if (character === '\\') { value += character + (text[index + 1] ?? ''); index += 2; continue; }
        if (character === '{') depth += 1;
        else if (character === '}') { depth -= 1; if (depth === 0) { index += 1; break; } }
        value += character;
        index += 1;
      }
      entry.fields[field[1]] = value;
    }
    entries.push(entry);
  }
  return entries;
}

function parseRis(text: string): Array<Record<string, string[]>> {
  const records: Array<Record<string, string[]>> = [];
  let current: Record<string, string[]> | null = null;
  for (const line of text.split('\r\n')) {
    if (line.trim() === '') continue;
    const match = /^([A-Z0-9]{2})  - (.*)$/.exec(line);
    if (!match) throw new Error(`Invalid RIS line: ${JSON.stringify(line)}`);
    const [, tag, value] = match;
    if (tag === 'TY') current = {};
    if (!current) throw new Error('RIS record started without a TY tag');
    (current[tag] ??= []).push(value);
    if (tag === 'ER') { records.push(current); current = null; }
  }
  if (current) throw new Error('RIS record was not terminated with ER');
  return records;
}

function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = '';
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (quoted) {
      if (character === '"') {
        if (text[index + 1] === '"') { field += '"'; index += 1; }
        else quoted = false;
      } else field += character;
    } else if (character === '"') {
      quoted = true;
    } else if (character === ',') {
      row.push(field);
      field = '';
    } else if (character === '\n') {
      row.push(field);
      field = '';
      rows.push(row);
      row = [];
    } else if (character !== '\r') {
      field += character;
    }
  }
  if (field !== '' || row.length > 0) { row.push(field); rows.push(row); }
  return rows;
}

test('formats a readable plain citation with Unicode authors', () => {
  assert.equal(
    plainCitation(paper),
    'Māia Example and Alex Researcher (2026). Speech & Language {Models}. arXiv:2607.00001. https://arxiv.org/abs/2607.00001',
  );
});

test('formats valid escaped BibTeX fields and stable key', () => {
  const citation = bibtexCitation(paper);
  assert.match(citation, /^@misc\{Example2026260700001,/);
  assert.match(citation, /title = \{Speech \\& Language \\\{Models\\\}\}/);
  assert.match(citation, /author = \{Māia Example and Alex Researcher\}/);
  assert.match(citation, /primaryClass = \{cs\.CL\}/);
  assert.match(citation, /doi = \{10\.1000\/example\\_1\}/);
  assert.match(citation, /journal = \{Speech \\& Language 42\}/);
  assert.ok(citation.endsWith('}\n'));
});

test('BibTeX decodes to one structured entry with balanced braces', () => {
  const [entry, ...rest] = parseBibtex(bibtexCitation(paper));
  assert.equal(rest.length, 0);
  assert.equal(entry.type, 'misc');
  assert.equal(entry.key, 'Example2026260700001');
  assert.equal(entry.fields.eprint, '2607.00001');
  assert.equal(entry.fields.year, '2026');
  assert.equal(entry.fields.archivePrefix, 'arXiv');
  assert.deepEqual(entry.fields.author.split(' and '), ['Māia Example', 'Alex Researcher']);
});

test('preserves TeX math segments while escaping surrounding BibTeX text', () => {
  const citation = bibtexCitation({
    ...paper,
    title: 'Alignment $x \\rightarrow y$ with A&B',
  });
  assert.match(citation, /title = \{Alignment \$x \\rightarrow y\$ with A\\&B\}/);
  assert.doesNotMatch(citation, /textbackslash/);
  assert.doesNotThrow(() => parseBibtex(citation));
});

test('RIS decodes to a terminated record with repeated author tags', () => {
  const [record, ...rest] = parseRis(risCitation(paper));
  assert.equal(rest.length, 0);
  assert.deepEqual(record.TY, ['GEN']);
  assert.deepEqual(record.ID, ['2607.00001']);
  assert.deepEqual(record.TI, ['Speech & Language {Models}']);
  assert.deepEqual(record.AU, ['Māia Example', 'Alex Researcher']);
  assert.deepEqual(record.PY, ['2026']);
  assert.deepEqual(record.DO, ['10.1000/example_1']);
  assert.deepEqual(record.JF, ['Speech & Language 42']);
  assert.deepEqual(record.KW, ['cs.CL']);
  assert.deepEqual(record.UR, ['https://arxiv.org/abs/2607.00001']);
  assert.deepEqual(record.N1, ['arXiv:2607.00001']);
  assert.ok('ER' in record);
});

test('RIS collapses embedded line breaks so every tag stays on one line', () => {
  const risText = risCitation({ ...paper, title: 'Line one\nLine two' });
  const [record] = parseRis(risText);
  assert.deepEqual(record.TI, ['Line one Line two']);
});

test('multi-record BibTeX and RIS decode to every selected paper', () => {
  const second: CitationPaper = {
    id: '2606.00002',
    title: 'Second Paper',
    authors: ['Solo Author'],
    published: '2026-06-15',
    paperUrl: 'https://arxiv.org/abs/2606.00002',
  };
  const bibEntries = parseBibtex(bibtexExport([paper, second]));
  assert.deepEqual(bibEntries.map((entry) => entry.fields.eprint), ['2607.00001', '2606.00002']);
  const risRecords = parseRis(risExport([paper, second]));
  assert.deepEqual(risRecords.map((record) => record.ID[0]), ['2607.00001', '2606.00002']);
  assert.deepEqual(risRecords[1].AU, ['Solo Author']);
});

test('CSV escapes quotes and delimiters and round-trips through a parser', () => {
  const tricky: CitationPaper = {
    id: '2606.00002',
    title: 'Speech, "Language" & Models',
    authors: ['First Author', 'Second Author'],
    published: '2026-06-15',
    primaryCategory: 'eess.AS',
    paperUrl: 'https://arxiv.org/abs/2606.00002',
  };
  const rows = parseCsv(csvExport([paper, tricky]));
  assert.deepEqual(rows[0], [
    'id', 'title', 'authors', 'published', 'primary_category', 'paper_url', 'doi', 'journal_reference',
  ]);
  assert.equal(rows.length, 3);
  assert.equal(rows[2][1], 'Speech, "Language" & Models');
  assert.equal(rows[2][2], 'First Author; Second Author');
  assert.equal(rows[1][0], '2607.00001');
});

test('Markdown export escapes inline control characters and links every paper', () => {
  const risky: CitationPaper = {
    id: '2606.00002',
    title: 'Pipes | and *stars* in [titles]',
    authors: ['Author One'],
    published: '2026-06-15',
    paperUrl: 'https://arxiv.org/abs/2606.00002',
  };
  const markdown = markdownExport([paper, risky]);
  assert.match(markdown, /^# Exported papers \(2\)\n/);
  const links = [...markdown.matchAll(/\[arXiv:([^\]]+)\]\(([^)]+)\)/g)];
  assert.deepEqual(links.map((match) => match[1]), ['2607.00001', '2606.00002']);
  assert.equal(links[1][2], 'https://arxiv.org/abs/2606.00002');
  assert.match(markdown, /Pipes \\\| and \\\*stars\\\* in \\\[titles\\\]/);
});

test('handles missing authors and publication date honestly', () => {
  const incomplete = { ...paper, authors: [], published: 'Unknown' };
  assert.match(plainCitation(incomplete), /^Unknown author \(n\.d\.\)/);
  assert.match(bibtexCitation(incomplete), /year = \{n\.d\.\}/);
  assert.match(bibtexCitation(incomplete), /author = \{Unknown author\}/);
  assert.match(bibtexCitation(incomplete), /^@misc\{unknownnd260700001,/);
  const [record] = parseRis(risCitation(incomplete));
  assert.deepEqual(record.AU, ['Unknown author']);
  assert.deepEqual(record.PY, ['n.d.']);
});