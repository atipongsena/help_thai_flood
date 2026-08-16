import { useMemo } from 'react';
import { useGetTimelineQuery } from '../features/cases/casesApi';

const PRIORITY_COLORS: Record<string, string> = {
  P1: '#ef4444',
  P2: '#f97316',
  P3: '#3b82f6',
};

export const CaseTimelineChart = () => {
  const { data = [], isFetching } = useGetTimelineQuery(10);

  const points = useMemo(() => data.slice(-10), [data]);
  const maxTotal = useMemo(() => Math.max(...points.map((p) => p.total), 1), [points]);

  return (
    <section className="timeline-card">
      <header>
        <div>
          <h3>กราฟจำนวนเคส (10 วัน)</h3>
          <p className="muted">แบ่งตาม Priority</p>
        </div>
        {isFetching && <span className="muted">กำลังโหลด...</span>}
      </header>
      <div className="timeline-bars">
        {points.map((point) => (
          <div key={point.day} className="timeline-column">
            <div className="timeline-bar" aria-label={`${point.day} total ${point.total}`}>
              {['P1', 'P2', 'P3'].map((level) => {
                const value = point.priority[level] ?? 0;
                const height = (value / maxTotal) * 120;
                if (!value) return null;
                return (
                  <span
                    key={level}
                    style={{ height: `${height}px`, backgroundColor: PRIORITY_COLORS[level] }}
                    title={`${level}: ${value}`}
                  />
                );
              })}
            </div>
            <small>{point.day.slice(5)}</small>
            <strong>{point.total}</strong>
          </div>
        ))}
      </div>
    </section>
  );
};

