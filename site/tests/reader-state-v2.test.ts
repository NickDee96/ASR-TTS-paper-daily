import assert from 'node:assert/strict';
import test from 'node:test';
import {
  beginReaderVisit,
  completeReaderVisit,
  exportReaderState,
  importReaderState,
  isNewSinceVisit,
  LEGACY_BOOKMARK_KEY,
  markPaperSeen,
  MAX_SEEN_PAPERS,
  paperPath,
  readReaderState,
  READER_BOOKMARK_PREFIX,
  READER_META_KEY,
  READER_RECOVERY_PREFIX,
  READER_SEEN_PREFIX,
  READER_SESSION_KEY,
  toggleBookmark,
  V1_READER_STATE_KEY,
} from '../src/lib/reader-state-v2.ts';
import type { BookmarkSnapshot, StorageLike } from '../src/lib/reader-state-v2.ts';

class MemoryStorage implements StorageLike {
  values = new Map<string, string>();
  get length() { return this.values.size; }
  getItem(key: string) { return this.values.get(key) ?? null; }
  setItem(key: string, value: string) { this.values.set(key, value); }
  removeItem(key: string) { this.values.delete(key); }
  key(index: number) { return [...this.values.keys()][index] ?? null; }
}

const bookmark: BookmarkSnapshot = {
  id: '2607.00001',
  title: 'Speech Research',
  abstract: 'A useful paper.',
  authors: ['Māia Example'],
  published: '2026-07-01',
  updated: '2026-07-03',
  firstSeenAt: '2026-07-03T12:00:00Z',
  topics: ['ASR'],
  categories: ['cs.CL'],
  codeStatus: 'verified',
  url: 'https://malicious.example/paper',
  savedAt: '2026-07-24T10:00:00Z',
};

test('stores bookmarks independently and reconstructs internal paper URLs', () => {
  const storage = new MemoryStorage();
  toggleBookmark(storage, bookmark);
  toggleBookmark(storage, { ...bookmark, id: '2607.00002', title: 'Second' });
  const state = readReaderState(storage);
  assert.deepEqual(Object.keys(state.bookmarks), ['2607.00001', '2607.00002']);
  assert.equal(state.bookmarks[bookmark.id].url, paperPath(bookmark.id));
  assert.equal(
    [...storage.values.keys()].filter((key) => key.startsWith(READER_BOOKMARK_PREFIX)).length,
    2,
  );
});

test('migrates v1 snapshots and the legacy ID array to per-key storage', () => {
  const storage = new MemoryStorage();
  storage.setItem(V1_READER_STATE_KEY, JSON.stringify({
    version: 1,
    bookmarks: { [bookmark.id]: bookmark },
    lastVisitAt: '2026-07-20T09:00:00Z',
  }));
  storage.setItem(LEGACY_BOOKMARK_KEY, JSON.stringify(['2607.00002']));
  const state = readReaderState(storage);
  assert.deepEqual(Object.keys(state.bookmarks), ['2607.00001', '2607.00002']);
  assert.equal(state.lastVisitAt, '2026-07-20T09:00:00.000Z');
  assert.equal(storage.getItem(V1_READER_STATE_KEY), null);
  assert.equal(storage.getItem(LEGACY_BOOKMARK_KEY), null);
});

test('quarantines malformed and future-version metadata before recovery', () => {
  const storage = new MemoryStorage();
  storage.setItem(READER_META_KEY, JSON.stringify({ version: 99, private: 'keep me' }));
  const state = readReaderState(storage);
  assert.equal(state.lastVisitAt, null);
  assert.equal(storage.getItem(READER_META_KEY), null);
  const recovery = [...storage.values.entries()].find(([key]) => key.startsWith(READER_RECOVERY_PREFIX));
  assert.match(recovery?.[1] ?? '', /keep me/);
});

test('exports bookmarks without private visit or seen history and merges imports', () => {
  const storage = new MemoryStorage();
  toggleBookmark(storage, bookmark);
  completeReaderVisit(storage, new Date('2026-07-24T10:00:00Z'));
  markPaperSeen(storage, bookmark.id, new Date('2026-07-24T11:00:00Z'));
  const exported = exportReaderState(readReaderState(storage));
  assert.doesNotMatch(exported, /lastVisitAt|"seen"/);
  assert.match(exported, /"version": 2/);
  const target = new MemoryStorage();
  const imported = importReaderState(target, exported);
  assert.equal(imported.bookmarks[bookmark.id].title, bookmark.title);
  assert.equal(imported.lastVisitAt, null);
  assert.throws(() => importReaderState(target, 'x'.repeat(5_000_001)), /larger than 5 MB/);
});

test('keeps visit metadata isolated and stable within a session', () => {
  const storage = new MemoryStorage();
  const session = new MemoryStorage();
  completeReaderVisit(storage, new Date('2026-07-20T09:00:00Z'));
  assert.equal(beginReaderVisit(storage, session), '2026-07-20T09:00:00.000Z');
  completeReaderVisit(storage, new Date('2026-07-24T10:00:00Z'));
  assert.equal(beginReaderVisit(storage, session), '2026-07-20T09:00:00.000Z');
  assert.equal(session.getItem(READER_SESSION_KEY), '2026-07-20T09:00:00.000Z');
  assert.equal(readReaderState(storage).lastVisitAt, '2026-07-24T10:00:00.000Z');
});

test('tracks bounded per-paper seen state and compares first-seen timestamps', () => {
  const storage = new MemoryStorage();
  for (let index = 0; index < MAX_SEEN_PAPERS + 2; index += 1) {
    markPaperSeen(
      storage,
      `2607.${String(index).padStart(5, '0')}`,
      new Date(1_700_000_000_000 + index * 1_000),
    );
  }
  const state = readReaderState(storage);
  assert.equal(Object.keys(state.seen).length, MAX_SEEN_PAPERS);
  assert.equal(
    [...storage.values.keys()].filter((key) => key.startsWith(READER_SEEN_PREFIX)).length,
    MAX_SEEN_PAPERS,
  );
  assert.equal(
    isNewSinceVisit('2026-07-21T00:00:00Z', '2026-07-20T23:59:59Z'),
    true,
  );
  assert.equal(isNewSinceVisit(null, '2026-07-20T00:00:00Z'), false);
});