export interface PagefindResultData {
  url: string;
  raw_url?: string;
  content: string;
  word_count: number;
  meta: Record<string, string>;
  filters: Record<string, string[]>;
  excerpt: string;
  plain_excerpt: string;
}

export interface PagefindResult {
  id: string;
  score: number;
  data(): Promise<PagefindResultData>;
}

export interface PagefindResponse {
  results: PagefindResult[];
  unfilteredResultCount: number;
  filters: Record<string, Record<string, number>>;
  totalFilters: Record<string, Record<string, number>>;
}

export interface PagefindModule {
  options(options: {
    basePath: string;
    baseUrl: string;
    excerptLength?: number;
  }): Promise<void>;
  init(): Promise<void>;
  filters(): Promise<Record<string, Record<string, number>>>;
  search(
    query: string | null,
    options?: {
      filters?: Record<string, unknown>;
      sort?: Record<string, 'asc' | 'desc'>;
    },
  ): Promise<PagefindResponse>;
}