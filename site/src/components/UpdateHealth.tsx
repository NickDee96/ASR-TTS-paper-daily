import { useEffect, useState } from 'react';

interface UpdateHealthProps {
  updatedAt: string;
  staleAfterHours: number;
}

const relativeFormatter = new Intl.RelativeTimeFormat('en', { numeric: 'auto' });
const dateFormatter = new Intl.DateTimeFormat('en', {
  dateStyle: 'medium',
  timeStyle: 'short',
  timeZone: 'UTC',
});

function relativeLabel(hoursAgo: number): string {
  if (hoursAgo < 1) return relativeFormatter.format(-Math.round(hoursAgo * 60), 'minute');
  if (hoursAgo < 48) return relativeFormatter.format(-Math.round(hoursAgo), 'hour');
  return relativeFormatter.format(-Math.round(hoursAgo / 24), 'day');
}

export default function UpdateHealth({ updatedAt, staleAfterHours }: UpdateHealthProps) {
  // Staleness depends on the reader's current clock, so it is computed after hydration.
  const [now, setNow] = useState<number | null>(null);
  useEffect(() => {
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 60_000);
    return () => window.clearInterval(timer);
  }, []);

  const updatedMs = new Date(updatedAt).getTime();
  const known = Number.isFinite(updatedMs) && updatedMs > 0;
  const formatted = known ? `${dateFormatter.format(updatedMs)} UTC` : 'unknown';

  if (now === null) {
    return (
      <div className="update-health" role="status" aria-live="polite">
        <p className="update-health-line"><span className="update-dot" aria-hidden="true" />Last successful update: {formatted}</p>
      </div>
    );
  }

  const hoursAgo = known ? (now - updatedMs) / 3_600_000 : Number.NaN;
  const stale = Number.isFinite(hoursAgo) && hoursAgo > staleAfterHours;

  return (
    <div className={`update-health ${stale ? 'stale' : 'fresh'}`} role="status" aria-live="polite">
      <p className="update-health-line">
        <span className="update-dot" aria-hidden="true" />
        {known
          ? `Last successful update ${relativeLabel(hoursAgo)} (${formatted})`
          : 'Last successful update time is unavailable.'}
      </p>
      {stale && (
        <p className="update-health-warning" role="alert">
          Data may be stale: no successful update in over {staleAfterHours} hours. The archive
          below reflects the last complete build.
        </p>
      )}
    </div>
  );
}
