import {
  Button,
  FluentProvider,
  Tooltip,
  webLightTheme,
} from '@fluentui/react-components';
import {
  Copy,
  Download,
  Share2,
} from 'lucide-react';
import { useState } from 'react';
import BookmarkToggle from './BookmarkToggle';
import { bibtexCitation, plainCitation, risCitation } from '../lib/citation';
import type { CitationPaper } from '../lib/citation';
import type { BookmarkSnapshot } from '../lib/reader-state-v2';

interface DetailActionsProps {
  paper: CitationPaper;
  bookmark: BookmarkSnapshot;
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

async function copyText(value: string): Promise<void> {
  if (!navigator.clipboard?.writeText) {
    throw new Error('Clipboard access is unavailable in this browser.');
  }
  await navigator.clipboard.writeText(value);
}

function downloadText(filename: string, mimeType: string, content: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

export default function DetailActions({ paper, bookmark }: DetailActionsProps) {
  const [status, setStatus] = useState('');

  async function copyCitation() {
    try {
      await copyText(plainCitation(paper));
      setStatus('Citation copied to clipboard.');
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Citation could not be copied.');
    }
  }

  function downloadBibtex() {
    downloadText(`${paper.id}.bib`, 'application/x-bibtex;charset=utf-8', bibtexCitation(paper));
    setStatus('BibTeX download started.');
  }

  function downloadRis() {
    downloadText(`${paper.id}.ris`, 'application/x-research-info-systems;charset=utf-8', risCitation(paper));
    setStatus('RIS download started.');
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
        <Button icon={<Download aria-hidden="true" size={17} />} onClick={downloadRis}>
          RIS
        </Button>
        <BookmarkToggle snapshot={bookmark} />
        <Button icon={<Share2 aria-hidden="true" size={17} />} onClick={() => void sharePaper()}>
          Share
        </Button>
      </div>
      <p className="action-status" role="status" aria-live="polite">{status}</p>
    </FluentProvider>
  );
}