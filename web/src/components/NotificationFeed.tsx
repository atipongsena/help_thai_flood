import { useGetNotificationsQuery } from '../features/cases/casesApi';

const priorityColorMap: Record<string, string> = {
  P1: '#ef4444',
  P2: '#f97316',
  P3: '#3b82f6',
};

export const NotificationFeed = () => {
  const { data = [], isFetching } = useGetNotificationsQuery({ limit: 25 }, { pollingInterval: 30_000 });

  return (
    <section className="notification-card">
      <header>
        <h3>อัพเดตล่าสุด</h3>
        {isFetching && <span className="muted">รีเฟรช...</span>}
      </header>
      {data.length === 0 ? (
        <p className="muted">ยังไม่มีอัพเดต</p>
      ) : (
        <ul>
          {data.map((item) => (
            <li key={`${item.case_id}-${item.entry.at}`}>
              <div className="badge" style={{ backgroundColor: priorityColorMap[item.priority_label] ?? '#475569' }}>
                {item.priority_label}
              </div>
              <div>
                <strong>#{item.case_id.slice(-6)}</strong> · {item.status}
                <p>{item.entry.message || item.entry.action}</p>
                <small>{new Date(item.entry.at ?? '').toLocaleString()}</small>
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
};

