import assert from 'node:assert/strict';
import test from 'node:test';
import { bibtexCitation, plainCitation } from '../src/lib/citation.ts';

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

test('preserves TeX math segments while escaping surrounding BibTeX text', () => {
  const citation = bibtexCitation({
    ...paper,
    title: 'Alignment $x \\rightarrow y$ with A&B',
  });
  assert.match(citation, /title = \{Alignment \$x \\rightarrow y\$ with A\\&B\}/);
  assert.doesNotMatch(citation, /textbackslash/);
});

test('handles missing authors and publication date honestly', () => {
  const incomplete = { ...paper, authors: [], published: 'Unknown' };
  assert.match(plainCitation(incomplete), /^Unknown author \(n\.d\.\)/);
  assert.match(bibtexCitation(incomplete), /year = \{n\.d\.\}/);
  assert.match(bibtexCitation(incomplete), /author = \{Unknown author\}/);
  assert.match(bibtexCitation(incomplete), /^@misc\{unknownnd260700001,/);
});