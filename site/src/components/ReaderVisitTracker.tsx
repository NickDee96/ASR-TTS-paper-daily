import { useEffect } from 'react';
import { beginReaderVisit, completeReaderVisit } from '../lib/reader-state-v2';

interface ReaderVisitTrackerProps {
  manifestUrl: string;
}

export default function ReaderVisitTracker({ manifestUrl }: ReaderVisitTrackerProps) {
  useEffect(() => {
    let cancelled = false;
    try {
      beginReaderVisit(localStorage, sessionStorage);
    } catch {
      // Browsing remains functional when private storage is blocked.
    }
    fetch(manifestUrl, { cache: 'no-store' })
      .then((response) => {
        if (!cancelled && response.ok) completeReaderVisit(localStorage);
      })
      .catch(() => {
        // Offline and stale-data visits must not advance the successful-visit timestamp.
      });
    return () => { cancelled = true; };
  }, [manifestUrl]);
  return null;
}