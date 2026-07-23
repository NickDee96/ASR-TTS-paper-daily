import assert from 'node:assert/strict';
import test from 'node:test';
import {
  activeFilterCount,
  criteriaKey,
  normalizeSearchState,
  parseSearchState,
  resetFilters,
  serializeSearchState,
  yearsInRange,
} from '../src/lib/search-state.ts';

test('parses and normalizes URL search state', () => {
  const state = parseSearchState(
    '?q=streaming&topic=ASR&category=cs.CL&code=verified&status=revised'
    + '&metadata=complete&from=2026&to=2024&sort=updated&page=3',
  );
  assert.deepEqual(state, {
    query: 'streaming',
    topic: 'ASR',
    category: 'cs.CL',
    code: 'verified',
    status: 'revised',
    recordStatus: 'complete',
    fromYear: '2024',
    toYear: '2026',
    sort: 'updated',
    page: 3,
  });
});

test('rejects invalid years, sort values, and pages', () => {
  const state = parseSearchState('?from=24&to=banana&sort=random&page=-8');
  assert.equal(state.fromYear, '');
  assert.equal(state.toYear, '');
  assert.equal(state.sort, 'relevance');
  assert.equal(state.page, 1);
});

test('serializes only meaningful nondefault values', () => {
  const serialized = serializeSearchState(parseSearchState(
    '?q=asr&topic=ASR&sort=newest&page=2',
  ));
  assert.equal(serialized, 'q=asr&topic=ASR&sort=newest&page=2');
  assert.equal(serializeSearchState(parseSearchState('')), 'sort=relevance');
  assert.equal(
    serializeSearchState({ ...parseSearchState(''), sort: 'relevance' }),
    'sort=relevance',
  );
});

test('counts combined filters and preserves query when clearing them', () => {
  const state = parseSearchState(
    '?q=asr&topic=ASR&code=verified&from=2024&to=2026&metadata=complete',
  );
  assert.equal(activeFilterCount(state), 4);
  assert.deepEqual(resetFilters(state), {
    query: 'asr',
    topic: '',
    category: '',
    code: '',
    status: '',
    recordStatus: '',
    fromYear: '',
    toYear: '',
    sort: 'relevance',
    page: 1,
  });
});

test('derives inclusive date ranges from indexed years', () => {
  assert.deepEqual(
    yearsInRange(['Unknown', '2026', '2024', '2025'], '2024', '2025'),
    ['2024', '2025'],
  );
  assert.deepEqual(yearsInRange(['2026', '2025'], '', ''), ['2025', '2026']);
});

test('normalizes interactive year bounds and ignores page in criteria keys', () => {
  const state = parseSearchState('?q=asr&from=2024&to=2026&sort=updated&page=3');
  const normalized = normalizeSearchState({ ...state, fromYear: '2026', toYear: '2024' });
  assert.equal(normalized.fromYear, '2024');
  assert.equal(normalized.toYear, '2026');
  assert.equal(criteriaKey(state), criteriaKey({ ...state, page: 7 }));
});