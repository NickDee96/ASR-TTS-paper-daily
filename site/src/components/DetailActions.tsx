import {
  Button,
  FluentProvider,
  Tooltip,
  webLightTheme,
} from '@fluentui/react-components';
import {
  Bookmark,
  BookmarkCheck,
  Copy,
  Download,
  Share2,
} from 'lucide-react';
import { useEffect, useState } from 'react';
import { bibtexCitation, plainCitation } from '../lib/citation';
import type { CitationPaper } from '../lib/citation';

interface DetailActionsProps {
  paper: CitationPaper;
}

const BOOKMARK_KEY = 'asr-tts-bookmarks:v1';
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

function readBookmarks(): string[] {
  try {
    const value = JSON.parse(localStorage.getItem(BOOKMARK_KEY) ?? '[]');
    return Array.isArray(value)
      ? value.filter((id): id is string => typeof id === 'string').slice(0, 5_000)
      : [];
  } catch {
    return [];
  }
}

async function copyText(value: string): Promise<void> {
  if (!navigator.clipboard?.writeText) {
    throw new Error('Clipboard access is unavailable in this browser.');
  }
  await navigator.clipboard.writeText(value);
}

export default function DetailActions({ paper }: DetailActionsProps) {
  const [bookmarked, setBookmarked] = useState(false);
  const [status, setStatus] = useState('');

  useEffect(() => {
    setBookmarked(readBookmarks().includes(paper.id));
  }, [paper.id]);

  async function copyCitation() {
    try {
      await copyText(plainCitation(paper));
      setStatus('Citation copied to clipboard.');
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Citation could not be copied.');
    }
  }

  function downloadBibtex() {
    const blob = new Blob([bibtexCitation(paper)], { type: 'application/x-bibtex;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${paper.id}.bib`;
    document.body.append(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
    setStatus('BibTeX download started.');
  }

  function toggleBookmark() {
    try {
      const bookmarks = new Set(readBookmarks());
      if (bookmarks.has(paper.id)) bookmarks.delete(paper.id);
      else {
        if (bookmarks.size >= 5_000) {
          setStatus('Bookmark limit reached. Export or remove bookmarks before adding more.');
          return;
        }
        bookmarks.add(paper.id);
      }
      localStorage.setItem(BOOKMARK_KEY, JSON.stringify([...bookmarks]));
      const selected = bookmarks.has(paper.id);
      setBookmarked(selected);
      setStatus(selected ? 'Paper bookmarked on this device.' : 'Bookmark removed.');
    } catch {
      setStatus('Bookmark could not be saved in this browser.');
    }
  }

  async function sharePaper() {
    if (navigator.share) {
      try {
        await navigator.share({ title: paper.title, text: plainCitation(paper), url: window.location.href });
        setStatus('Paper shared.');
        return;
      } catch (error) {
        if (error instanceof DOMException && error.name === 'AbortError') return;
      }
    }
    try {
      await copyText(window.location.href);
      setStatus('Paper link copied to clipboard.');
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Paper link could not be copied.');
    }
  }

  return (
    <FluentProvider theme={researchTheme} className="detail-tools-provider">
      <div className="detail-tools" aria-label="Paper actions">
        <Tooltip content="Copy a plain-text citation" relationship="label">
          <Button icon={<Copy aria-hidden="true" size={17} />} onClick={() => void copyCitation()}>
            Copy citation
          </Button>
        </Tooltip>
        <Button icon={<Download aria-hidden="true" size={17} />} onClick={downloadBibtex}>
          BibTeX
        </Button>
        <Button
          appearance={bookmarked ? 'primary' : 'secondary'}
          icon={bookmarked
            ? <BookmarkCheck aria-hidden="true" size={17} />
            : <Bookmark aria-hidden="true" size={17} />}
          aria-pressed={bookmarked}
          onClick={toggleBookmark}
        >
          {bookmarked ? 'Bookmarked' : 'Bookmark'}
        </Button>
        <Button icon={<Share2 aria-hidden="true" size={17} />} onClick={() => void sharePaper()}>
          Share
        </Button>
      </div>
      <p className="action-status" role="status" aria-live="polite">{status}</p>
    </FluentProvider>
  );
}