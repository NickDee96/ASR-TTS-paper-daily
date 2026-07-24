export const READER_META_KEY = 'asr-tts-reader-meta:v2';
export const READER_BOOKMARK_PREFIX = 'asr-tts-reader-bookmark:v2:';
export const READER_SEEN_PREFIX = 'asr-tts-reader-seen:v2:';
export const READER_SEEN_INDEX_KEY = 'asr-tts-reader-seen-index:v2';
export const READER_RECOVERY_PREFIX = 'asr-tts-reader-recovery:';
export const V1_READER_STATE_KEY = 'asr-tts-reader-state:v1';
export const LEGACY_BOOKMARK_KEY = 'asr-tts-bookmarks:v1';
export const READER_SESSION_KEY = 'asr-tts-reader-session:v2';
export const READER_STATE_EVENT = 'asr-tts-reader-state';
export const MAX_BOOKMARKS = 5_000;
export const MAX_SEEN_PAPERS = 2_000;
export const MAX_IMPORT_BYTES = 5_000_000;

const ARXIV_ID_PATTERN = /^(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?\/\d{7})$/;

export interface StorageLike {
  readonly length: number;
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
  key(index: number): string | null;
}

export interface BookmarkSnapshot {
  id: string;
  title: string;
  abstract: string;
  authors: string[];
  published: string;
  updated: string;
  firstSeenAt: string | null;
  topics: string[];
  categories: string[];
  codeStatus: string;
  url: string;
  savedAt: string;
}

export interface ReaderState {
  version: 2;
  bookmarks: Record<string, BookmarkSnapshot>;
  seen: Record<string, string>;
  lastVisitAt: string | null;
}

interface ReaderMeta {
  version: 2;
  lastVisitAt: string | null;
}

export interface ToggleBookmarkResult {
  state: ReaderState;
  bookmarked: boolean;
}

export class ReaderStateError extends Error {}

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
  return /^\d{4}-\d{2}-\d{2}$/.test(text) ? text : 'Unknown';
}

function cleanTimestamp(value: unknown): string | null {
  const text = cleanText(value, 40);
  return text && Number.isFinite(Date.parse(text)) ? new Date(text).toISOString() : null;
}

function cleanId(value: unknown): string {
  const id = cleanText(value, 80);
  return ARXIV_ID_PATTERN.test(id) ? id : '';
}

export function paperPath(id: string): string {
  return `/ASR-TTS-paper-daily/papers/${encodeURIComponent(id)}/`;
}

function bookmarkKey(id: string): string {
  return `${READER_BOOKMARK_PREFIX}${encodeURIComponent(id)}`;
}

function seenKey(id: string): string {
  return `${READER_SEEN_PREFIX}${encodeURIComponent(id)}`;
}

function storageKeys(storage: StorageLike, prefix: string): string[] {
  const keys: string[] = [];
  for (let index = 0; index < storage.length; index += 1) {
    const key = storage.key(index);
    if (key?.startsWith(prefix)) keys.push(key);
  }
  return keys.sort();
}

function quarantine(storage: StorageLike, key: string, raw: string): void {
  try {
    const suffix = `${Date.now()}:${encodeURIComponent(key)}`;
    storage.setItem(`${READER_RECOVERY_PREFIX}${suffix}`, raw.slice(0, 5_000_000));
    storage.removeItem(key);
  } catch {
    // Keep the original untouched when recovery storage is unavailable.
  }
}

function readSeenIndex(storage: StorageLike): string[] {
  const raw = storage.getItem(READER_SEEN_INDEX_KEY);
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) throw new Error('invalid seen index');
    return [...new Set(parsed.map(cleanId).filter(Boolean))].slice(-MAX_SEEN_PAPERS);
  } catch {
    quarantine(storage, READER_SEEN_INDEX_KEY, raw);
    return [];
  }
}

export function sanitizeBookmark(value: unknown): BookmarkSnapshot | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const source = value as Record<string, unknown>;
  const id = cleanId(source.id);
  if (!id) return null;
  return {
    id,
    title: cleanText(source.title, 500) || `arXiv ${id}`,
    abstract: cleanText(source.abstract, 4_000),
    authors: cleanList(source.authors, 100, 200),
    published: cleanDate(source.published),
    updated: cleanDate(source.updated),
    firstSeenAt: cleanTimestamp(source.firstSeenAt),
    topics: cleanList(source.topics, 50, 100),
    categories: cleanList(source.categories, 50, 100),
    codeStatus: cleanText(source.codeStatus, 32) || 'missing',
    url: paperPath(id),
    savedAt: cleanTimestamp(source.savedAt) ?? new Date(0).toISOString(),
  };
}

function readMeta(storage: StorageLike): ReaderMeta {
  const raw = storage.getItem(READER_META_KEY);
  if (!raw) return { version: 2, lastVisitAt: null };
  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    if (parsed.version !== 2) {
      quarantine(storage, READER_META_KEY, raw);
      return { version: 2, lastVisitAt: null };
    }
    return { version: 2, lastVisitAt: cleanTimestamp(parsed.lastVisitAt) };
  } catch {
    quarantine(storage, READER_META_KEY, raw);
    return { version: 2, lastVisitAt: null };
  }
}

function writeMeta(storage: StorageLike, meta: ReaderMeta): ReaderMeta {
  const sanitized = { version: 2 as const, lastVisitAt: cleanTimestamp(meta.lastVisitAt) };
  storage.setItem(READER_META_KEY, JSON.stringify(sanitized));
  return sanitized;
}

function migrateV1(storage: StorageLike): void {
  const rawState = storage.getItem(V1_READER_STATE_KEY);
  if (rawState) {
    try {
      const parsed = JSON.parse(rawState) as Record<string, unknown>;
      if (parsed.version !== 1) {
        quarantine(storage, V1_READER_STATE_KEY, rawState);
      } else {
        const rawBookmarks = parsed.bookmarks;
        if (rawBookmarks && typeof rawBookmarks === 'object' && !Array.isArray(rawBookmarks)) {
          for (const raw of Object.values(rawBookmarks).slice(0, MAX_BOOKMARKS)) {
            const bookmark = sanitizeBookmark(raw);
            if (bookmark) storage.setItem(bookmarkKey(bookmark.id), JSON.stringify(bookmark));
          }
        }
        writeMeta(storage, { version: 2, lastVisitAt: cleanTimestamp(parsed.lastVisitAt) });
        storage.removeItem(V1_READER_STATE_KEY);
      }
    } catch {
      quarantine(storage, V1_READER_STATE_KEY, rawState);
    }
  }

  const rawLegacy = storage.getItem(LEGACY_BOOKMARK_KEY);
  if (!rawLegacy) return;
  try {
    const ids = JSON.parse(rawLegacy);
    if (!Array.isArray(ids)) throw new Error('invalid legacy bookmark list');
    for (const rawId of ids.slice(0, MAX_BOOKMARKS)) {
      const id = cleanId(rawId);
      if (!id) continue;
      const bookmark = sanitizeBookmark({
        id,
        title: `arXiv ${id}`,
        abstract: 'Open the paper page to refresh this migrated bookmark.',
        authors: [],
        published: 'Unknown',
        updated: 'Unknown',
        firstSeenAt: null,
        topics: [],
        categories: [],
        codeStatus: 'missing',
        savedAt: new Date(0).toISOString(),
      });
      if (bookmark) storage.setItem(bookmarkKey(id), JSON.stringify(bookmark));
    }
    storage.removeItem(LEGACY_BOOKMARK_KEY);
  } catch {
    quarantine(storage, LEGACY_BOOKMARK_KEY, rawLegacy);
  }
}

export function readReaderState(storage: StorageLike): ReaderState {
  migrateV1(storage);
  const bookmarks: Record<string, BookmarkSnapshot> = {};
  for (const key of storageKeys(storage, READER_BOOKMARK_PREFIX).slice(0, MAX_BOOKMARKS)) {
    const raw = storage.getItem(key);
    if (!raw) continue;
    try {
      const bookmark = sanitizeBookmark(JSON.parse(raw));
      if (bookmark) bookmarks[bookmark.id] = bookmark;
      else quarantine(storage, key, raw);
    } catch {
      quarantine(storage, key, raw);
    }
  }
  const seen: Record<string, string> = {};
  for (const id of readSeenIndex(storage)) {
    const timestamp = cleanTimestamp(storage.getItem(seenKey(id)));
    if (timestamp) seen[id] = timestamp;
  }
  const meta = readMeta(storage);
  return { version: 2, bookmarks, seen, lastVisitAt: meta.lastVisitAt };
}

export function toggleBookmark(
  storage: StorageLike,
  value: BookmarkSnapshot,
): ToggleBookmarkResult {
  const bookmark = sanitizeBookmark(value);
  if (!bookmark) throw new ReaderStateError('Paper bookmark data is invalid.');
  const key = bookmarkKey(bookmark.id);
  if (storage.getItem(key)) {
    storage.removeItem(key);
    return { state: readReaderState(storage), bookmarked: false };
  }
  const count = storageKeys(storage, READER_BOOKMARK_PREFIX).length;
  if (count >= MAX_BOOKMARKS) {
    throw new ReaderStateError(
      'Bookmark limit reached. Export or remove bookmarks before adding more.',
    );
  }
  storage.setItem(key, JSON.stringify(bookmark));
  return { state: readReaderState(storage), bookmarked: true };
}

export function refreshBookmarkIfPresent(
  storage: StorageLike,
  value: BookmarkSnapshot,
): ReaderState {
  const bookmark = sanitizeBookmark(value);
  if (!bookmark) return readReaderState(storage);
  const key = bookmarkKey(bookmark.id);
  const raw = storage.getItem(key);
  if (!raw) return readReaderState(storage);
  const existing = sanitizeBookmark(JSON.parse(raw));
  storage.setItem(key, JSON.stringify({
    ...bookmark,
    savedAt: existing?.savedAt ?? bookmark.savedAt,
  }));
  return readReaderState(storage);
}

export function exportReaderState(state: ReaderState): string {
  return `${JSON.stringify({ version: 2, bookmarks: state.bookmarks }, null, 2)}\n`;
}

export function importReaderState(storage: StorageLike, text: string): ReaderState {
  if (new Blob([text]).size > MAX_IMPORT_BYTES) {
    throw new ReaderStateError('Bookmark file is larger than 5 MB.');
  }
  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(text) as Record<string, unknown>;
  } catch {
    throw new ReaderStateError('Bookmark file is not valid JSON.');
  }
  if (parsed.version !== 1 && parsed.version !== 2) {
    throw new ReaderStateError('Bookmark file has an unsupported format.');
  }
  const rawBookmarks = parsed.bookmarks;
  if (!rawBookmarks || typeof rawBookmarks !== 'object' || Array.isArray(rawBookmarks)) {
    throw new ReaderStateError('Bookmark file has an unsupported format.');
  }
  const imported = Object.values(rawBookmarks)
    .map(sanitizeBookmark)
    .filter((bookmark): bookmark is BookmarkSnapshot => Boolean(bookmark));
  const currentCount = storageKeys(storage, READER_BOOKMARK_PREFIX).length;
  const newCount = imported.filter((bookmark) => !storage.getItem(bookmarkKey(bookmark.id))).length;
  if (currentCount + newCount > MAX_BOOKMARKS) {
    throw new ReaderStateError(`Bookmark file exceeds the ${MAX_BOOKMARKS} paper limit.`);
  }
  for (const bookmark of imported) {
    storage.setItem(bookmarkKey(bookmark.id), JSON.stringify(bookmark));
  }
  return readReaderState(storage);
}

export function beginReaderVisit(
  storage: StorageLike,
  sessionStorage: StorageLike,
): string | null {
  const sessionBaseline = sessionStorage.getItem(READER_SESSION_KEY);
  if (sessionBaseline !== null) return cleanTimestamp(sessionBaseline);
  const previous = readMeta(storage).lastVisitAt;
  sessionStorage.setItem(READER_SESSION_KEY, previous ?? '');
  return previous;
}

export function completeReaderVisit(
  storage: StorageLike,
  now: Date = new Date(),
): void {
  writeMeta(storage, { version: 2, lastVisitAt: now.toISOString() });
}

export function markPaperSeen(
  storage: StorageLike,
  id: string,
  now: Date = new Date(),
): void {
  const clean = cleanId(id);
  if (!clean) return;
  const key = seenKey(clean);
  const existed = storage.getItem(key) !== null;
  storage.setItem(key, now.toISOString());
  if (existed) return;
  const index = readSeenIndex(storage).filter((entry) => entry !== clean);
  index.push(clean);
  const removed = index.splice(0, Math.max(0, index.length - MAX_SEEN_PAPERS));
  for (const removedId of removed) storage.removeItem(seenKey(removedId));
  storage.setItem(READER_SEEN_INDEX_KEY, JSON.stringify(index));
}

export function hasSeenPaper(storage: StorageLike, id: string): boolean {
  const clean = cleanId(id);
  return Boolean(clean && storage.getItem(seenKey(clean)));
}

export function isNewSinceVisit(
  firstSeenAt: string | null,
  baseline: string | null,
): boolean {
  if (!firstSeenAt || !baseline) return false;
  const firstSeen = Date.parse(firstSeenAt);
  const previousVisit = Date.parse(baseline);
  return Number.isFinite(firstSeen) && Number.isFinite(previousVisit)
    && firstSeen > previousVisit;
}