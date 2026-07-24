import { useEffect } from 'react';
import { beginReaderVisit } from '../lib/reader-state';

export default function ReaderVisitTracker() {
  useEffect(() => {
    try {
      beginReaderVisit(localStorage, sessionStorage);
    } catch {
      // Browsing remains functional when private storage is blocked.
    }
  }, []);
  return null;
}