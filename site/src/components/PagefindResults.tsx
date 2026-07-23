import {
  Button,
  FluentProvider,
  MessageBar,
  MessageBarBody,
  MessageBarTitle,
  Spinner,
  webLightTheme,
} from '@fluentui/react-components';
import { ArrowDown, ArrowUpRight, RotateCw } from 'lucide-react';
import { startTransition, useEffect, useRef, useState } from 'react';
import SearchFilters from './SearchFilters';
import {
  DEFAULT_SEARCH_STATE,
  criteriaKey,
  normalizeSearchState,
  parseSearchState,
  resetFilters,
  serializeSearchState,
  yearsInRange,
} from '../lib/search-state';
import type { SearchUrlState } from '../lib/search-state';
import type {
  PagefindModule,
  PagefindResponse,
  PagefindResult,
  PagefindResultData,
} from '../types/pagefind';

interface PagefindResultsProps {
  baseUrl: string;
  bundlePath: string;
  updatedAt: string;
  staleAfterHours?: number;
}

interface SearchState {
  status: 'loading' | 'ready' | 'empty' | 'no-match' | 'error';
  total: number;
  items: PagefindResultData[];
  facets: Record<string, Record<string, number>>;
  message?: string;
}

const PAGE_SIZE = 25;
const researchTheme = {
  ...webLightTheme,
  colorBrandBackground: '#176b5f',
  colorBrandBackgroundHover: '#10584f',
  colorBrandBackgroundPressed: '#0d4942',
  colorBrandForeground1: '#176b5f',
  fontFamilyBase: '"Instrument Sans", sans-serif',
  borderRadiusMedium: '4px',
  borderRadiusLarge: '6px',
};

function displayDate(value?: string): string {
  if (!value || value === 'Unknown') return 'Date unknown';
  const parsed = new Date(`${value}T00:00:00Z`);
  if (Number.isNaN(parsed.getTime())) return value;
  return new Intl.DateTimeFormat('en', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    timeZone: 'UTC',
  }).format(parsed);
}

function first(values: string[] | undefined, fallback = ''): string {
  return values?.[0] ?? fallback;
}

function stateFromLocation(): SearchUrlState {
  const parsed = parseSearchState(window.location.search);
  const parameters = new URLSearchParams(window.location.search);
  if (!parameters.has('sort') && !parsed.query) parsed.sort = 'updated';
  return parsed;
}

function ResultRow({ result }: { result: PagefindResultData }) {
  const authors = result.filters.author ?? [];
  const topics = result.filters.topic ?? [];
  const categories = result.filters.category ?? [];
  const codeStatus = first(result.filters.code, result.meta.code_status || 'missing');
  const status = result.meta.status || 'new';
  const arxivId = result.meta.arxiv_id || result.url.split('/').filter(Boolean).at(-1) || '';
  return (
    <article className="search-result-row">
      <div className="search-result-date">
        <time dateTime={result.meta.updated}>{displayDate(result.meta.updated)}</time>
        <span className={status}>{status}</span>
      </div>
      <div className="search-result-content">
        <div className="paper-kicker">
          <span>{arxivId}</span>
          {categories.length > 0 && <span>{categories.join(' / ')}</span>}
          {codeStatus === 'verified' && <span className="code-verified">verified code</span>}
        </div>
        <h2><a href={result.url}>{result.meta.title || arxivId}</a></h2>
        <p className="search-excerpt" dangerouslySetInnerHTML={{ __html: result.excerpt }} />
        <div className="paper-footer">
          <span className="authors">{authors.length > 0 ? authors.join(', ') : 'Authors unavailable'}</span>
          <span className="topic-list">{topics.map((topic) => <span key={topic}>{topic}</span>)}</span>
        </div>
      </div>
      <a className="row-action" href={result.url} aria-label={`Open ${result.meta.title || arxivId}`}>
        <ArrowUpRight aria-hidden="true" size={18} />
      </a>
    </article>
  );
}

export default function PagefindResults({
  baseUrl,
  bundlePath,
  updatedAt,
  staleAfterHours = 36,
}: PagefindResultsProps) {
  const resultReferences = useRef<PagefindResult[]>([]);
  const searchGeneration = useRef(0);
  const pageGeneration = useRef(0);
  const activeCriteria = useRef('');
  const pagefindReference = useRef<Promise<PagefindModule> | null>(null);
  const globalFacetsReference = useRef<Record<string, Record<string, number>>>({});
  const [attempt, setAttempt] = useState(0);
  const [loadingMore, setLoadingMore] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [urlState, setUrlState] = useState<SearchUrlState>(() => (
    typeof window === 'undefined' ? DEFAULT_SEARCH_STATE : stateFromLocation()
  ));
  const urlStateReference = useRef<SearchUrlState>(urlState);
  const [state, setState] = useState<SearchState>({
    status: 'loading',
    total: 0,
    items: [],
    facets: {},
  });
  const updatedTime = new Date(updatedAt).getTime();
  const stale = Number.isFinite(updatedTime)
    && Date.now() - updatedTime > staleAfterHours * 60 * 60 * 1000;

  urlStateReference.current = urlState;

  function updateUrl(requested: SearchUrlState, mode: 'push' | 'replace' = 'push') {
    const next = normalizeSearchState(requested);
    const queryString = serializeSearchState(next);
    const nextUrl = `${window.location.pathname}${queryString ? `?${queryString}` : ''}`;
    window.history[mode === 'push' ? 'pushState' : 'replaceState'](null, '', nextUrl);
    setUrlState(next);
  }

  async function restorePage(page: number) {
    const generation = searchGeneration.current;
    const requestedPageGeneration = pageGeneration.current + 1;
    pageGeneration.current = requestedPageGeneration;
    const limit = Math.min(resultReferences.current.length, PAGE_SIZE * page);
    try {
      const items = await Promise.all(
        resultReferences.current.slice(0, limit).map((result) => result.data()),
      );
      if (
        generation !== searchGeneration.current
        || requestedPageGeneration !== pageGeneration.current
      ) return;
      setState((current) => ({ ...current, items }));
    } catch {
      setAttempt((value) => value + 1);
    }
  }

  useEffect(() => {
    function onPopState() {
      const current = urlStateReference.current;
      const next = stateFromLocation();
      setUrlState(next);
      if (criteriaKey(current) === criteriaKey(next) && current.page !== next.page) {
        void restorePage(next.page);
      }
    }
    function onSearchSubmit(event: SubmitEvent) {
      const form = event.target;
      if (!(form instanceof HTMLFormElement) || !form.matches('.search-toolbar')) return;
      event.preventDefault();
      const formData = new FormData(form);
      const nextQuery = String(formData.get('q') ?? '').trim();
      const sortWasExplicit = new URLSearchParams(window.location.search).has('sort');
      updateUrl({
        ...urlState,
        query: nextQuery,
        sort: nextQuery && !urlState.query && !sortWasExplicit
          ? 'relevance'
          : urlState.sort,
        page: 1,
      });
    }
    function onTopicClick(event: MouseEvent) {
      const target = event.target;
      const link = target instanceof Element ? target.closest<HTMLAnchorElement>('[data-topic-link]') : null;
      if (!link) return;
      if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      const selectedTopic = link.dataset.topic ?? '';
      updateUrl({
        ...urlState,
        topic: selectedTopic === urlState.topic ? '' : selectedTopic,
        page: 1,
      });
    }
    function onTopicClear(event: MouseEvent) {
      const target = event.target;
      const link = target instanceof Element ? target.closest<HTMLAnchorElement>('.topic-clear') : null;
      if (!link) return;
      if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      updateUrl({ ...urlState, topic: '', page: 1 });
    }
    window.addEventListener('popstate', onPopState);
    document.addEventListener('submit', onSearchSubmit);
    document.addEventListener('click', onTopicClick);
    document.addEventListener('click', onTopicClear);
    return () => {
      window.removeEventListener('popstate', onPopState);
      document.removeEventListener('submit', onSearchSubmit);
      document.removeEventListener('click', onTopicClick);
      document.removeEventListener('click', onTopicClear);
    };
  }, [urlState]);

  useEffect(() => {
    const queryInput = document.querySelector<HTMLInputElement>('input[name="q"]');
    if (queryInput) queryInput.value = urlState.query;
    const topicInput = document.querySelector<HTMLInputElement>('input[name="topic"]');
    if (topicInput) topicInput.value = urlState.topic;
    for (const link of document.querySelectorAll<HTMLAnchorElement>('[data-topic-link]')) {
      const selected = link.dataset.topic === urlState.topic;
      const next = {
        ...urlState,
        topic: selected ? '' : link.dataset.topic ?? '',
        page: 1,
      };
      link.href = `${window.location.pathname}?${serializeSearchState(next)}`;
      link.classList.toggle('active', selected);
      if (selected) link.setAttribute('aria-current', 'true');
      else link.removeAttribute('aria-current');
    }
    const clearLink = document.querySelector<HTMLAnchorElement>('.topic-clear');
    if (clearLink) {
      clearLink.hidden = !urlState.topic;
      clearLink.href = `${window.location.pathname}?${serializeSearchState({ ...urlState, topic: '', page: 1 })}`;
    }
  }, [urlState]);

  useEffect(() => {
    let cancelled = false;
    const generation = searchGeneration.current + 1;
    searchGeneration.current = generation;
    pageGeneration.current += 1;
    activeCriteria.current = criteriaKey(urlState);
    async function runSearch() {
      setState((current) => ({ ...current, status: 'loading', total: 0, items: [] }));
      try {
        if (!pagefindReference.current) {
          pagefindReference.current = (async () => {
            const moduleUrl = `${bundlePath}pagefind.js`;
            const pagefind = await import(/* @vite-ignore */ moduleUrl) as PagefindModule;
            await pagefind.options({ basePath: bundlePath, baseUrl, excerptLength: 38 });
            await pagefind.init();
            globalFacetsReference.current = await pagefind.filters();
            return pagefind;
          })();
        }
        const pagefind = await pagefindReference.current;
        const options: Parameters<PagefindModule['search']>[1] = {};
        const filters: Record<string, unknown> = {};
        if (urlState.topic) filters.topic = urlState.topic;
        if (urlState.category) filters.category = urlState.category;
        if (urlState.code) filters.code = urlState.code;
        if (urlState.status) filters.status = urlState.status;
        if (urlState.recordStatus) filters.record_status = urlState.recordStatus;
        if (urlState.fromYear || urlState.toYear) {
          const years = yearsInRange(
            Object.keys(globalFacetsReference.current.year ?? {}),
            urlState.fromYear,
            urlState.toYear,
          );
          if (years.length === 1) filters.year = years[0];
          else if (years.length > 1) filters.any = years.map((year) => ({ year }));
          else filters.year = '__no_matching_year__';
        }
        if (Object.keys(filters).length > 0) options.filters = filters;
        if (urlState.sort === 'newest') options.sort = { published: 'desc' };
        else if (urlState.sort === 'updated') options.sort = { updated: 'desc' };
        const response: PagefindResponse = await pagefind.search(urlState.query || null, options);
        const initialLimit = Math.min(response.results.length, PAGE_SIZE * urlState.page);
        const initialItems = await Promise.all(
          response.results.slice(0, initialLimit).map((result) => result.data()),
        );
        if (cancelled || generation !== searchGeneration.current) return;
        resultReferences.current = response.results;
        const status = response.results.length === 0
          ? (urlState.query || Object.keys(filters).length ? 'no-match' : 'empty')
          : 'ready';
        setState({
          status,
          total: response.results.length,
          items: initialItems,
          facets: Object.keys(response.totalFilters).length
            ? response.totalFilters
            : globalFacetsReference.current,
        });
      } catch (error) {
        if (cancelled || generation !== searchGeneration.current) return;
        pagefindReference.current = null;
        setState({
          status: 'error',
          total: 0,
          items: [],
          facets: globalFacetsReference.current,
          message: error instanceof Error ? error.message : 'Search could not be loaded.',
        });
      }
    }
    void runSearch();
    return () => { cancelled = true; };
  }, [
    attempt,
    baseUrl,
    bundlePath,
    urlState.category,
    urlState.code,
    urlState.fromYear,
    urlState.query,
    urlState.recordStatus,
    urlState.sort,
    urlState.status,
    urlState.toYear,
    urlState.topic,
  ]);

  async function loadMore() {
    const generation = searchGeneration.current;
    const requestedPageGeneration = pageGeneration.current + 1;
    pageGeneration.current = requestedPageGeneration;
    const criteria = activeCriteria.current;
    const start = state.items.length;
    const nextReferences = resultReferences.current.slice(start, start + PAGE_SIZE);
    if (nextReferences.length === 0) return;
    setLoadingMore(true);
    try {
      const nextItems = await Promise.all(nextReferences.map((result) => result.data()));
      if (
        generation !== searchGeneration.current
        || requestedPageGeneration !== pageGeneration.current
        || criteria !== criteriaKey(urlStateReference.current)
      ) {
        return;
      }
      startTransition(() => {
        setState((current) => ({ ...current, items: [...current.items, ...nextItems] }));
      });
      const next = { ...urlStateReference.current, page: urlStateReference.current.page + 1 };
      const queryString = serializeSearchState(next);
      window.history.pushState(null, '', `${window.location.pathname}?${queryString}`);
      setUrlState(next);
    } finally {
      setLoadingMore(false);
    }
  }

  const context = [urlState.query && `"${urlState.query}"`, urlState.topic].filter(Boolean).join(' in ');
  const updateFilters = (patch: Partial<SearchUrlState>) => {
    updateUrl(normalizeSearchState({ ...urlState, ...patch, page: 1 }));
  };
  return (
    <FluentProvider theme={researchTheme} className="results-provider">
      {stale && (
        <MessageBar intent="warning" className="search-message">
          <MessageBarBody>
            <MessageBarTitle>Archive update delayed</MessageBarTitle>
            Results remain available from the last successful build on {displayDate(updatedAt.slice(0, 10))}.
          </MessageBarBody>
        </MessageBar>
      )}
      <SearchFilters
        state={urlState}
        facets={state.facets}
        drawerOpen={drawerOpen}
        onDrawerOpenChange={setDrawerOpen}
        onChange={updateFilters}
        onClear={() => updateUrl(resetFilters(urlState))}
      />
      <div className="results-heading">
        <div>
          <h2 id="results-heading">{context ? `Results for ${context}` : 'Newest indexed papers'}</h2>
          <p aria-live="polite">
            {state.status === 'loading' ? 'Loading search index' : `${state.total.toLocaleString()} ${state.total === 1 ? 'paper' : 'papers'}`}
          </p>
        </div>
      </div>

      {state.status === 'loading' && (
        <div className="search-state" role="status">
          <Spinner label="Loading the paper index" size="medium" />
          <p>Only the search chunks needed for this query will be downloaded.</p>
        </div>
      )}
      {state.status === 'error' && (
        <div className="search-state" role="alert">
          <h2>Search is temporarily unavailable</h2>
          <p>{state.message}</p>
          <Button icon={<RotateCw aria-hidden="true" size={17} />} onClick={() => setAttempt((value) => value + 1)}>
            Retry search
          </Button>
        </div>
      )}
      {state.status === 'empty' && (
        <div className="search-state">
          <h2>The index is empty</h2>
          <p>The next successful archive build will repopulate paper search.</p>
        </div>
      )}
      {state.status === 'no-match' && (
        <div className="search-state">
          <h2>No matching papers</h2>
          <p>Try a broader phrase, an exact arXiv ID, or a different topic.</p>
        </div>
      )}
      {state.status === 'ready' && (
        <>
          <div className="search-result-list">
            {state.items.map((result) => <ResultRow key={result.url} result={result} />)}
          </div>
          {state.items.length < state.total && (
            <div className="load-more">
              <Button
                appearance="secondary"
                icon={<ArrowDown aria-hidden="true" size={17} />}
                disabled={loadingMore}
                onClick={() => void loadMore()}
              >
                {loadingMore ? 'Loading papers' : `Load 25 more (${state.total - state.items.length} remaining)`}
              </Button>
            </div>
          )}
        </>
      )}
    </FluentProvider>
  );
}