export const READER_STATE_KEY = 'asr-tts-reader-state:v1';
export const LEGACY_BOOKMARK_KEY = 'asr-tts-bookmarks:v1';
export const READER_SESSION_KEY = 'asr-tts-reader-session:v1';
export const READER_STATE_EVENT = 'asr-tts-reader-state';
export const MAX_BOOKMARKS = 5_000;

export interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem?(key: string): void;
}

export interface BookmarkSnapshot {
  id: string;
  title: string;
  abstract: string;
  authors: string[];
  published: string;
  updated: string;
  topics: string[];
  categories: string[];
  codeStatus: string;
  url: string;
  savedAt: string;
}

export interface ReaderState {
  version: 1;
  bookmarks: Record<string, BookmarkSnapshot>;
  lastVisitAt: string | null;
}

export interface ToggleBookmarkResult {
  state: ReaderState;
  bookmarked: boolean;
}

export class ReaderStateError extends Error {}

function emptyState(): ReaderState {
  return { version: 1, bookmarks: {}, lastVisitAt: null };
}

function cleanText(value: unknown, maxLength: number): string {
  return typeof value === 'string'
    ? value.replace(/\s+/g, ' ').trim().slice(0, maxLength)
    : '';
}

function cleanList(value: unknown, maxItems: number, maxLength: number): string[] {
  if (!Array.isArray(value)) return [];
  return [...new Set(value.map((item) => cleanText(item, maxLength)).filter(Boolean))]
    .slice(0, maxItems);
}

function cleanDate(value: unknown): string {
  const text = cleanText(value, 32);
  return /^\d{4}-\d{2}-\d{2}(?:T[^\s]+)?$/.test(text) ? text : 'Unknown';
}

function cleanTimestamp(value: unknown): string | null {
  const text = cleanText(value, 40);
  return text && Number.isFinite(Date.parse(text)) ? new Date(text).toISOString() : null;
}

function cleanUrl(value: unknown): string {
  const text = cleanText(value, 500);
  return text.startsWith('/') || /^https?:\/\//i.test(text) ? text : '';
}

export function sanitizeBookmark(value: unknown): BookmarkSnapshot | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const source = value as Record<string, unknown>;
  const id = cleanText(source.id, 80);
  if (!id) return null;
  return {
    id,
    title: cleanText(source.title, 500) || `arXiv ${id}`,
    abstract: cleanText(source.abstract, 4_000),
    authors: cleanList(source.authors, 100, 200),
    published: cleanDate(source.published),
    updated: cleanDate(source.updated),
    topics: cleanList(source.topics, 50, 100),
    categories: cleanList(source.categories, 50, 100),
    codeStatus: cleanText(source.codeStatus, 32) || 'missing',
    url: cleanUrl(source.url),
    savedAt: cleanTimestamp(source.savedAt) ?? new Date(0).toISOString(),
  };
}

function sanitizeState(value: unknown): ReaderState {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return emptyState();
  const source = value as Record<string, unknown>;
  const rawBookmarks = source.bookmarks;
  const bookmarks: Record<string, BookmarkSnapshot> = {};
  if (rawBookmarks && typeof rawBookmarks === 'object' && !Array.isArray(rawBookmarks)) {
    for (const raw of Object.values(rawBookmarks).slice(0, MAX_BOOKMARKS)) {
      const bookmark = sanitizeBookmark(raw);
      if (bookmark) bookmarks[bookmark.id] = bookmark;
    }
  }
  return {
    version: 1,
    bookmarks,
    lastVisitAt: cleanTimestamp(source.lastVisitAt),
  };
}

function migrateLegacyBookmarks(storage: StorageLike): ReaderState | null {
  try {
    const raw = storage.getItem(LEGACY_BOOKMARK_KEY);
    if (!raw) return null;
    const ids = JSON.parse(raw);
    if (!Array.isArray(ids)) return null;
    const bookmarks: Record<string, BookmarkSnapshot> = {};
    for (const item of ids.slice(0, MAX_BOOKMARKS)) {
      const id = cleanText(item, 80);
      if (!id) continue;
      bookmarks[id] = {
        id,
        title: `arXiv ${id}`,
        abstract: 'Open the paper page to refresh this migrated bookmark.',
        authors: [],
        published: 'Unknown',
        updated: 'Unknown',
        topics: [],
        categories: [],
        codeStatus: 'missing',
        url: '',
        savedAt: new Date(0).toISOString(),
      };
    }
    return { version: 1, bookmarks, lastVisitAt: null };
  } catch {
    return null;
  }
}

export function readReaderState(storage: StorageLike): ReaderState {
  try {
    const raw = storage.getItem(READER_STATE_KEY);
    if (raw) return sanitizeState(JSON.parse(raw));
    const migrated = migrateLegacyBookmarks(storage);
    if (migrated) {
      writeReaderState(storage, migrated);
      storage.removeItem?.(LEGACY_BOOKMARK_KEY);
      return migrated;
    }
  } catch {
    return emptyState();
  }
  return emptyState();
}

export function writeReaderState(storage: StorageLike, value: ReaderState): ReaderState {
  const state = sanitizeState(value);
  storage.setItem(READER_STATE_KEY, JSON.stringify(state));
  return state;
}

export function toggleBookmark(
  storage: StorageLike,
  value: BookmarkSnapshot,
): ToggleBookmarkResult {
  const bookmark = sanitizeBookmark(value);
  if (!bookmark) throw new ReaderStateError('Paper bookmark data is invalid.');
  const state = readReaderState(storage);
  if (state.bookmarks[bookmark.id]) {
    delete state.bookmarks[bookmark.id];
    return { state: writeReaderState(storage, state), bookmarked: false };
  }
  if (Object.keys(state.bookmarks).length >= MAX_BOOKMARKS) {
    throw new ReaderStateError(
      'Bookmark limit reached. Export or remove bookmarks before adding more.',
    );
  }
  state.bookmarks[bookmark.id] = bookmark;
  return { state: writeReaderState(storage, state), bookmarked: true };
}

export function refreshBookmarkIfPresent(
  storage: StorageLike,
  value: BookmarkSnapshot,
): ReaderState {
  const bookmark = sanitizeBookmark(value);
  const state = readReaderState(storage);
  if (!bookmark || !state.bookmarks[bookmark.id]) return state;
  state.bookmarks[bookmark.id] = {
    ...bookmark,
    savedAt: state.bookmarks[bookmark.id].savedAt,
  };
  return writeReaderState(storage, state);
}

export function exportReaderState(state: ReaderState): string {
  return `${JSON.stringify(sanitizeState(state), null, 2)}\n`;
}

export function importReaderState(
  storage: StorageLike,
  text: string,
): ReaderState {
  let imported: ReaderState;
  try {
    const parsed = JSON.parse(text);
    if (!parsed || typeof parsed !== 'object' || parsed.version !== 1) {
      throw new ReaderStateError('Bookmark file has an unsupported format.');
    }
    imported = sanitizeState(parsed);
  } catch (error) {
    if (error instanceof ReaderStateError) throw error;
    throw new ReaderStateError('Bookmark file is not valid JSON.');
  }
  const current = readReaderState(storage);
  const merged = { ...current.bookmarks, ...imported.bookmarks };
  if (Object.keys(merged).length > MAX_BOOKMARKS) {
    throw new ReaderStateError(`Bookmark file exceeds the ${MAX_BOOKMARKS} paper limit.`);
  }
  return writeReaderState(storage, {
    version: 1,
    bookmarks: merged,
    lastVisitAt: current.lastVisitAt,
  });
}

export function beginReaderVisit(
  storage: StorageLike,
  sessionStorage: StorageLike,
  now: Date = new Date(),
): string | null {
  const sessionBaseline = sessionStorage.getItem(READER_SESSION_KEY);
  if (sessionBaseline !== null) return cleanTimestamp(sessionBaseline);
  const state = readReaderState(storage);
  const previous = state.lastVisitAt;
  const current = now.toISOString();
  state.lastVisitAt = current;
  writeReaderState(storage, state);
  sessionStorage.setItem(READER_SESSION_KEY, previous ?? '');
  return previous;
}

export function isNewSinceVisit(updated: string, baseline: string | null): boolean {
  if (!baseline || !/^\d{4}-\d{2}-\d{2}/.test(updated)) return false;
  return updated.slice(0, 10) > baseline.slice(0, 10);
}