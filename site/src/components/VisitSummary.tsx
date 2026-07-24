import { useEffect, useState } from 'react';
import { beginReaderVisit, isNewSinceVisit } from '../lib/reader-state-v2';

interface VisitSummaryProps {
  updates: string[];
}

export default function VisitSummary({ updates }: VisitSummaryProps) {
  const [message, setMessage] = useState('');
  useEffect(() => {
    try {
      const baseline = beginReaderVisit(localStorage, sessionStorage);
      if (!baseline) {
        setMessage('Future visits will highlight newly added papers.');
        return;
      }
      const count = updates.filter((updated) => isNewSinceVisit(updated, baseline)).length;
      setMessage(count > 0
        ? `${count} ${count === 1 ? 'paper is' : 'papers are'} new since your last visit.`
        : 'No papers in this feed are newer than your last visit.');
    } catch {
      setMessage('Last-visit tracking is unavailable in this browser.');
    }
  }, [updates]);
  return <p className="visit-summary" role="status">{message}</p>;
}