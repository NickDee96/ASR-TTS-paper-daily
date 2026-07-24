import { useEffect, useState } from 'react';
import {
  readReaderState,
  READER_STATE_EVENT,
  ReaderStateError,
  refreshBookmarkIfPresent,
  toggleBookmark,
} from '../lib/reader-state-v2';
import type { BookmarkSnapshot } from '../lib/reader-state-v2';

function notifyReaderState() {
  window.dispatchEvent(new CustomEvent(READER_STATE_EVENT));
}

export interface BookmarkToggleOutcome {
  bookmarked: boolean | null;
  message: string;
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

  function toggle(): BookmarkToggleOutcome {
    try {
      const result = toggleBookmark(localStorage, {
        ...snapshot,
        savedAt: new Date().toISOString(),
      });
      setBookmarked(result.bookmarked);
      notifyReaderState();
      return {
        bookmarked: result.bookmarked,
        message: result.bookmarked
          ? 'Paper bookmarked on this device.'
          : 'Bookmark removed.',
      };
    } catch (error) {
      return {
        bookmarked: null,
        message: error instanceof ReaderStateError
          ? error.message
          : 'Bookmark could not be saved in this browser.',
      };
    }
  }

  return { bookmarked, toggle };
}