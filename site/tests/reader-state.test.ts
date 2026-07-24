import assert from 'node:assert/strict';
import test from 'node:test';
import {
  beginReaderVisit,
  exportReaderState,
  importReaderState,
  isNewSinceVisit,
  LEGACY_BOOKMARK_KEY,
  MAX_BOOKMARKS,
  readReaderState,
  READER_STATE_KEY,
  ReaderStateError,
  refreshBookmarkIfPresent,
  toggleBookmark,
} from '../src/lib/reader-state.ts';
import type { BookmarkSnapshot, StorageLike } from '../src/lib/reader-state.ts';

class MemoryStorage implements StorageLike {
  values = new Map<string, string>();
  getItem(key: string) { return this.values.get(key) ?? null; }
  setItem(key: string, value: string) { this.values.set(key, value); }
  removeItem(key: string) { this.values.delete(key); }
}

const bookmark: BookmarkSnapshot = {
  id: '2607.00001',
  title: 'Speech Research',
  abstract: 'A useful paper.',
  authors: ['Māia Example'],
  published: '2026-07-01',
  updated: '2026-07-03',
  topics: ['ASR'],
  categories: ['cs.CL'],
  codeStatus: 'verified',
  url: '/ASR-TTS-paper-daily/papers/2607.00001/',
  savedAt: '2026-07-24T10:00:00Z',
};

test('toggles a complete bookmark snapshot without losing Unicode', () => {
  const storage = new MemoryStorage();
  const added = toggleBookmark(storage, bookmark);
  assert.equal(added.bookmarked, true);
  assert.equal(added.state.bookmarks[bookmark.id].authors[0], 'Māia Example');
  const removed = toggleBookmark(storage, bookmark);
  assert.equal(removed.bookmarked, false);
  assert.deepEqual(removed.state.bookmarks, {});
});

test('recovers from corrupt state and sanitizes executable URLs', () => {
  const storage = new MemoryStorage();
  storage.setItem(READER_STATE_KEY, '{broken');
  assert.deepEqual(readReaderState(storage).bookmarks, {});
  const added = toggleBookmark(storage, { ...bookmark, url: 'javascript:alert(1)' });
  assert.equal(added.state.bookmarks[bookmark.id].url, '');
});

test('migrates the legacy ID-only key once', () => {
  const storage = new MemoryStorage();
  storage.setItem(LEGACY_BOOKMARK_KEY, JSON.stringify(['2607.00001', 12, '']));
  const state = readReaderState(storage);
  assert.deepEqual(Object.keys(state.bookmarks), ['2607.00001']);
  assert.equal(state.bookmarks['2607.00001'].title, 'arXiv 2607.00001');
  assert.equal(storage.getItem(LEGACY_BOOKMARK_KEY), null);
});

test('refreshes migrated snapshots only when already bookmarked', () => {
  const storage = new MemoryStorage();
  storage.setItem(LEGACY_BOOKMARK_KEY, JSON.stringify([bookmark.id]));
  readReaderState(storage);
  const refreshed = refreshBookmarkIfPresent(storage, bookmark);
  assert.equal(refreshed.bookmarks[bookmark.id].title, bookmark.title);
  assert.equal(refreshed.bookmarks[bookmark.id].savedAt, new Date(0).toISOString());
});

test('exports and merges validated bookmark files', () => {
  const source = new MemoryStorage();
  const target = new MemoryStorage();
  const sourceState = toggleBookmark(source, bookmark).state;
  toggleBookmark(target, { ...bookmark, id: '2607.00002', title: 'Existing' });
  const imported = importReaderState(target, exportReaderState(sourceState));
  assert.deepEqual(Object.keys(imported.bookmarks).sort(), ['2607.00001', '2607.00002']);
  assert.throws(
    () => importReaderState(target, '{nope'),
    (error) => error instanceof ReaderStateError && /valid JSON/.test(error.message),
  );
});

test('enforces the bookmark bound before writing', () => {
  const storage = new MemoryStorage();
  const bookmarks = Object.fromEntries(Array.from({ length: MAX_BOOKMARKS }, (_, index) => [
    String(index),
    { ...bookmark, id: String(index) },
  ]));
  storage.setItem(READER_STATE_KEY, JSON.stringify({
    version: 1,
    bookmarks,
    lastVisitAt: null,
  }));
  assert.throws(
    () => toggleBookmark(storage, { ...bookmark, id: 'over-limit' }),
    /Bookmark limit reached/,
  );
});

test('keeps one visit baseline per session and compares daily updates', () => {
  const storage = new MemoryStorage();
  const session = new MemoryStorage();
  storage.setItem(READER_STATE_KEY, JSON.stringify({
    version: 1,
    bookmarks: {},
    lastVisitAt: '2026-07-20T09:00:00Z',
  }));
  assert.equal(
    beginReaderVisit(storage, session, new Date('2026-07-24T10:00:00Z')),
    '2026-07-20T09:00:00.000Z',
  );
  assert.equal(
    beginReaderVisit(storage, session, new Date('2026-07-25T10:00:00Z')),
    '2026-07-20T09:00:00.000Z',
  );
  assert.equal(isNewSinceVisit('2026-07-21', '2026-07-20T09:00:00Z'), true);
  assert.equal(isNewSinceVisit('2026-07-20', '2026-07-20T09:00:00Z'), false);
});