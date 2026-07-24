import { useEffect, useState } from 'react';
import {
  readReaderState,
  READER_STATE_EVENT,
  ReaderStateError,
  refreshBookmarkIfPresent,
  toggleBookmark,
} from '../lib/reader-state';
import type { BookmarkSnapshot } from '../lib/reader-state';

function notifyReaderState() {
  window.dispatchEvent(new CustomEvent(READER_STATE_EVENT));
}

export function useReaderBookmark(snapshot: BookmarkSnapshot) {
  const [bookmarked, setBookmarked] = useState(false);

  useEffect(() => {
    function sync() {
      setBookmarked(Boolean(readReaderState(localStorage).bookmarks[snapshot.id]));
    }
    try {
      if (readReaderState(localStorage).bookmarks[snapshot.id]) {
        refreshBookmarkIfPresent(localStorage, snapshot);
        notifyReaderState();
      }
    } catch {
      // A blocked storage provider leaves the button usable for an honest retry.
    }
    sync();
    window.addEventListener(READER_STATE_EVENT, sync);
    window.addEventListener('storage', sync);
    return () => {
      window.removeEventListener(READER_STATE_EVENT, sync);
      window.removeEventListener('storage', sync);
    };
  }, [snapshot.id, snapshot.title, snapshot.updated, snapshot.url]);

  function toggle(): string {
    try {
      const result = toggleBookmark(localStorage, {
        ...snapshot,
        savedAt: new Date().toISOString(),
      });
      setBookmarked(result.bookmarked);
      notifyReaderState();
      return result.bookmarked
        ? 'Paper bookmarked on this device.'
        : 'Bookmark removed.';
    } catch (error) {
      if (error instanceof ReaderStateError) return error.message;
      return 'Bookmark could not be saved in this browser.';
    }
  }

  return { bookmarked, toggle };
}