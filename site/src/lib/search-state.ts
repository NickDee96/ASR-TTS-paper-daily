export const SORT_VALUES = ['relevance', 'newest', 'updated'] as const;
export type SearchSort = typeof SORT_VALUES[number];

export interface SearchUrlState {
  query: string;
  topic: string;
  category: string;
  code: string;
  status: string;
  recordStatus: string;
  fromYear: string;
  toYear: string;
  sort: SearchSort;
  page: number;
}

export const DEFAULT_SEARCH_STATE: SearchUrlState = {
  query: '',
  topic: '',
  category: '',
  code: '',
  status: '',
  recordStatus: '',
  fromYear: '',
  toYear: '',
  sort: 'relevance',
  page: 1,
};

function clean(parameters: URLSearchParams, name: string): string {
  return (parameters.get(name) ?? '').trim();
}

function validYear(value: string): string {
  return /^\d{4}$/.test(value) ? value : '';
}

export function parseSearchState(search: string): SearchUrlState {
  const parameters = new URLSearchParams(search);
  const requestedSort = clean(parameters, 'sort');
  const parsedPage = Number.parseInt(clean(parameters, 'page'), 10);
  let fromYear = validYear(clean(parameters, 'from'));
  let toYear = validYear(clean(parameters, 'to'));
  if (fromYear && toYear && Number(fromYear) > Number(toYear)) {
    [fromYear, toYear] = [toYear, fromYear];
  }
  return {
    query: clean(parameters, 'q'),
    topic: clean(parameters, 'topic'),
    category: clean(parameters, 'category'),
    code: clean(parameters, 'code'),
    status: clean(parameters, 'status'),
    recordStatus: clean(parameters, 'metadata'),
    fromYear,
    toYear,
    sort: SORT_VALUES.includes(requestedSort as SearchSort)
      ? requestedSort as SearchSort
      : DEFAULT_SEARCH_STATE.sort,
    page: Number.isFinite(parsedPage) && parsedPage > 0 ? parsedPage : 1,
  };
}

export function serializeSearchState(state: SearchUrlState): string {
  const parameters = new URLSearchParams();
  if (state.query) parameters.set('q', state.query);
  if (state.topic) parameters.set('topic', state.topic);
  if (state.category) parameters.set('category', state.category);
  if (state.code) parameters.set('code', state.code);
  if (state.status) parameters.set('status', state.status);
  if (state.recordStatus) parameters.set('metadata', state.recordStatus);
  if (state.fromYear) parameters.set('from', state.fromYear);
  if (state.toYear) parameters.set('to', state.toYear);
  if (
    state.sort !== DEFAULT_SEARCH_STATE.sort
    || (!state.query && state.sort === 'relevance')
  ) parameters.set('sort', state.sort);
  if (state.page > 1) parameters.set('page', String(state.page));
  return parameters.toString();
}

export function activeFilterCount(state: SearchUrlState): number {
  return [
    state.topic,
    state.category,
    state.code,
    state.status,
    state.recordStatus,
    state.fromYear || state.toYear,
  ].filter(Boolean).length;
}

export function yearsInRange(
  availableYears: string[],
  fromYear: string,
  toYear: string,
): string[] {
  const normalized = availableYears
    .filter((year) => /^\d{4}$/.test(year))
    .sort();
  return normalized.filter((year) => (
    (!fromYear || year >= fromYear) && (!toYear || year <= toYear)
  ));
}

export function resetFilters(state: SearchUrlState): SearchUrlState {
  return {
    ...DEFAULT_SEARCH_STATE,
    query: state.query,
    sort: state.sort,
  };
}

export function normalizeSearchState(state: SearchUrlState): SearchUrlState {
  return parseSearchState(`?${serializeSearchState(state)}`);
}

export function criteriaKey(state: SearchUrlState): string {
  return serializeSearchState({ ...state, page: 1 });
}