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
  query: string;
  topic: string;
  total: number;
  items: PagefindResultData[];
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
  const [attempt, setAttempt] = useState(0);
  const [loadingMore, setLoadingMore] = useState(false);
  const [state, setState] = useState<SearchState>({
    status: 'loading',
    query: '',
    topic: '',
    total: 0,
    items: [],
  });
  const updatedTime = new Date(updatedAt).getTime();
  const stale = Number.isFinite(updatedTime)
    && Date.now() - updatedTime > staleAfterHours * 60 * 60 * 1000;

  useEffect(() => {
    let cancelled = false;
    async function runSearch() {
      const parameters = new URLSearchParams(window.location.search);
      const query = (parameters.get('q') ?? '').trim();
      const topic = (parameters.get('topic') ?? '').trim();
      setState({ status: 'loading', query, topic, total: 0, items: [] });
      try {
        const moduleUrl = `${bundlePath}pagefind.js`;
        const pagefind = await import(/* @vite-ignore */ moduleUrl) as PagefindModule;
        await pagefind.options({ basePath: bundlePath, baseUrl, excerptLength: 38 });
        await pagefind.init();
        await pagefind.filters();
        const options: Parameters<PagefindModule['search']>[1] = {};
        if (topic) options.filters = { topic };
        if (!query) options.sort = { date: 'desc' };
        const response: PagefindResponse = await pagefind.search(query || null, options);
        resultReferences.current = response.results;
        const initialItems = await Promise.all(
          response.results.slice(0, PAGE_SIZE).map((result) => result.data()),
        );
        if (cancelled) return;
        const status = response.results.length === 0
          ? (query || topic ? 'no-match' : 'empty')
          : 'ready';
        setState({
          status,
          query,
          topic,
          total: response.results.length,
          items: initialItems,
        });
      } catch (error) {
        if (cancelled) return;
        setState({
          status: 'error',
          query,
          topic,
          total: 0,
          items: [],
          message: error instanceof Error ? error.message : 'Search could not be loaded.',
        });
      }
    }
    void runSearch();
    return () => { cancelled = true; };
  }, [attempt, baseUrl, bundlePath]);

  async function loadMore() {
    const start = state.items.length;
    const nextReferences = resultReferences.current.slice(start, start + PAGE_SIZE);
    if (nextReferences.length === 0) return;
    setLoadingMore(true);
    try {
      const nextItems = await Promise.all(nextReferences.map((result) => result.data()));
      startTransition(() => {
        setState((current) => ({ ...current, items: [...current.items, ...nextItems] }));
      });
    } finally {
      setLoadingMore(false);
    }
  }

  const context = [state.query && `"${state.query}"`, state.topic].filter(Boolean).join(' in ');
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