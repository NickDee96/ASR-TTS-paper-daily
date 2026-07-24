import { useEffect } from 'react';
import { markPaperSeen, READER_STATE_EVENT } from '../lib/reader-state-v2';

interface PaperSeenTrackerProps {
  paperId: string;
}

export default function PaperSeenTracker({ paperId }: PaperSeenTrackerProps) {
  useEffect(() => {
    try {
      markPaperSeen(localStorage, paperId);
      window.dispatchEvent(new CustomEvent(READER_STATE_EVENT));
    } catch {
      // Reading remains available when private storage is blocked.
    }
  }, [paperId]);
  return null;
}