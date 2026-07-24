import { useEffect } from 'react';
import { beginReaderVisit, completeReaderVisit } from '../lib/reader-state-v2';

interface ReaderVisitTrackerProps {
  manifestUrl: string;
}

export default function ReaderVisitTracker({ manifestUrl }: ReaderVisitTrackerProps) {
  useEffect(() => {
    let cancelled = false;
    let completed = false;
    try {
      beginReaderVisit(localStorage, sessionStorage);
    } catch {
      // Browsing remains functional when private storage is blocked.
    }

    async function validateCurrentData() {
      if (cancelled || completed || !navigator.onLine) return;
      try {
        const response = await fetch(manifestUrl, { cache: 'no-store' });
        const manifest = response.ok ? await response.json() : null;
        if (
          !cancelled
          && manifest
          && typeof manifest === 'object'
          && typeof manifest.unique_papers === 'number'
        ) {
          completeReaderVisit(localStorage);
          completed = true;
        }
      } catch {
        // Offline and stale-data visits must not advance the successful-visit timestamp.
      }
    }

    void validateCurrentData();
    window.addEventListener('online', validateCurrentData);
    return () => {
      cancelled = true;
      window.removeEventListener('online', validateCurrentData);
    };
  }, [manifestUrl]);
  return null;
}
