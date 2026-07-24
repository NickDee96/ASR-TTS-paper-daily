import { Button, Tooltip } from '@fluentui/react-components';
import { Bookmark, BookmarkCheck } from 'lucide-react';
import { useState } from 'react';
import { useReaderBookmark } from '../hooks/useReaderBookmark';
import type { BookmarkToggleOutcome } from '../hooks/useReaderBookmark';
import type { BookmarkSnapshot } from '../lib/reader-state-v2';

interface BookmarkToggleProps {
  snapshot: BookmarkSnapshot;
  compact?: boolean;
  onToggle?: (outcome: BookmarkToggleOutcome) => void;
}

export default function BookmarkToggle({ snapshot, compact = false, onToggle }: BookmarkToggleProps) {
  const { bookmarked, toggle } = useReaderBookmark(snapshot);
  const [status, setStatus] = useState('');
  const label = bookmarked ? 'Remove bookmark' : 'Bookmark paper';
  return (
    <span className="bookmark-toggle">
      <Tooltip content={label} relationship="label">
        <Button
          appearance={bookmarked ? 'primary' : 'subtle'}
          size={compact ? 'small' : 'medium'}
          icon={bookmarked
            ? <BookmarkCheck aria-hidden="true" size={17} />
            : <Bookmark aria-hidden="true" size={17} />}
          aria-label={label}
          aria-pressed={bookmarked}
          onClick={() => {
            const outcome = toggle();
            setStatus(outcome.message);
            onToggle?.(outcome);
          }}
        >
          {compact ? null : (bookmarked ? 'Bookmarked' : 'Bookmark')}
        </Button>
      </Tooltip>
      <span className="visually-hidden" role="status" aria-live="polite">{status}</span>
    </span>
  );
}