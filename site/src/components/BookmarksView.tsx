import {
  Button,
  FluentProvider,
  MessageBar,
  MessageBarBody,
  MessageBarTitle,
  webLightTheme,
} from '@fluentui/react-components';
import { Download, FileUp } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import BookmarkToggle from './BookmarkToggle';
import {
  exportReaderState,
  importReaderState,
  MAX_IMPORT_BYTES,
  readReaderState,
  READER_STATE_EVENT,
} from '../lib/reader-state-v2';
import type { BookmarkSnapshot, ReaderState } from '../lib/reader-state-v2';

interface BookmarksViewProps {
  baseUrl: string;
}

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

function sortedBookmarks(state: ReaderState): BookmarkSnapshot[] {
  return Object.values(state.bookmarks).sort((left, right) => (
    right.savedAt.localeCompare(left.savedAt) || left.title.localeCompare(right.title)
  ));
}

export default function BookmarksView({ baseUrl }: BookmarksViewProps) {
  const inputReference = useRef<HTMLInputElement>(null);
  const importButtonReference = useRef<HTMLButtonElement>(null);
  const bookmarkLinks = useRef(new Map<string, HTMLAnchorElement>());
  const pendingFocus = useRef<string | null>(null);
  const [state, setState] = useState<ReaderState>({
    version: 2,
    bookmarks: {},
    seen: {},
    lastVisitAt: null,
  });
  const [status, setStatus] = useState('');

  useEffect(() => {
    function sync() {
      setState(readReaderState(localStorage));
    }
    sync();
    window.addEventListener(READER_STATE_EVENT, sync);
    window.addEventListener('storage', sync);
    return () => {
      window.removeEventListener(READER_STATE_EVENT, sync);
      window.removeEventListener('storage', sync);
    };
  }, []);

  const bookmarks = sortedBookmarks(state);

  useEffect(() => {
    const target = pendingFocus.current;
    if (!target) return;
    pendingFocus.current = null;
    if (target === 'commands') importButtonReference.current?.focus();
    else bookmarkLinks.current.get(target)?.focus();
  }, [bookmarks.length]);

  function exportBookmarks() {
    const current = readReaderState(localStorage);
    const blob = new Blob([exportReaderState(current)], { type: 'application/json;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `asr-tts-bookmarks-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.append(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
    const count = Object.keys(current.bookmarks).length;
    setStatus(`Exported ${count} ${count === 1 ? 'bookmark' : 'bookmarks'}.`);
  }

  async function importBookmarks(file: File) {
    try {
      if (file.size > MAX_IMPORT_BYTES) {
        throw new Error('Bookmark file is larger than 5 MB.');
      }
      const imported = importReaderState(localStorage, await file.text());
      setState(imported);
      window.dispatchEvent(new CustomEvent(READER_STATE_EVENT));
      setStatus(`Imported bookmark file. Your library now contains ${Object.keys(imported.bookmarks).length} papers.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Bookmark file could not be imported.');
    } finally {
      if (inputReference.current) inputReference.current.value = '';
    }
  }

  function handleRemoval(bookmark: BookmarkSnapshot) {
    const index = bookmarks.findIndex((entry) => entry.id === bookmark.id);
    pendingFocus.current = bookmarks[index + 1]?.id ?? bookmarks[index - 1]?.id ?? 'commands';
    setStatus(`Removed ${bookmark.title} from bookmarks.`);
  }

  return (
    <FluentProvider theme={researchTheme} className="bookmarks-provider">
      <div className="bookmarks-command-bar">
        <Button
          ref={importButtonReference}
          icon={<FileUp aria-hidden="true" size={17} />}
          onClick={() => inputReference.current?.click()}
        >
          Import JSON
        </Button>
        <input
          ref={inputReference}
          className="visually-hidden"
          type="file"
          accept="application/json,.json"
          aria-label="Choose bookmark JSON file"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) void importBookmarks(file);
          }}
        />
        <Button
          icon={<Download aria-hidden="true" size={17} />}
          disabled={bookmarks.length === 0}
          onClick={exportBookmarks}
        >
          Export JSON
        </Button>
      </div>
      <p className="action-status" role="status" aria-live="polite">{status}</p>

      <MessageBar intent="info" className="bookmarks-note">
        <MessageBarBody>
          <MessageBarTitle>Stored only on this device</MessageBarTitle>
          Bookmark snapshots, import files, and visit history are never placed in URLs or sent to the repository.
        </MessageBarBody>
      </MessageBar>

      {bookmarks.length === 0 ? (
        <div className="bookmarks-empty">
          <h2>No bookmarked papers yet</h2>
          <p>Bookmark a paper from search results or its detail page. This view keeps a local snapshot for offline reading.</p>
          <a href={`${baseUrl}search/`}>Search the archive</a>
        </div>
      ) : (
        <div className="bookmark-list" aria-label="Bookmarked papers">
          {bookmarks.map((bookmark) => {
            const url = bookmark.url || `${baseUrl}papers/${encodeURIComponent(bookmark.id)}/`;
            return (
              <article className="bookmark-row" key={bookmark.id}>
                <div className="bookmark-row-main">
                  <div className="paper-kicker">
                    <span>{bookmark.id}</span>
                    {bookmark.categories.length > 0 && <span>{bookmark.categories.join(' / ')}</span>}
                    {bookmark.codeStatus === 'verified' && <span className="code-verified">verified code</span>}
                  </div>
                  <h2><a
                    ref={(node) => {
                      if (node) bookmarkLinks.current.set(bookmark.id, node);
                      else bookmarkLinks.current.delete(bookmark.id);
                    }}
                    href={url}
                  >{bookmark.title}</a></h2>
                  <p>{bookmark.abstract || 'Abstract unavailable in this saved snapshot.'}</p>
                  <div className="paper-footer">
                    <span className="authors">{bookmark.authors.length > 0 ? bookmark.authors.join(', ') : 'Authors unavailable'}</span>
                    <span className="topic-list">{bookmark.topics.map((topic) => <span key={topic}>{topic}</span>)}</span>
                  </div>
                </div>
                <div className="bookmark-row-actions">
                  <time dateTime={bookmark.savedAt}>Saved {bookmark.savedAt.slice(0, 10)}</time>
                  <BookmarkToggle
                    snapshot={bookmark}
                    compact
                    onToggle={(outcome) => {
                      if (outcome.bookmarked === false) handleRemoval(bookmark);
                    }}
                  />
                </div>
              </article>
            );
          })}
        </div>
      )}
    </FluentProvider>
  );
}