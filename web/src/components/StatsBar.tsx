import { useGetMetricsQuery } from '../features/cases/casesApi';
import { Activity, CheckCircle, Clock, AlertTriangle } from 'lucide-react';

export const StatsBar = () => {
  const { data, isFetching } = useGetMetricsQuery();

  if (isFetching || !data) {
    return (
      <section className="stats-container">
        <div className="stat-card glass-panel">
          <p>กำลังโหลดข้อมูล...</p>
        </div>
      </section>
    );
  }

  const status = data.status || {};
  const priority = data.priority || {};

  return (
    <section className="stats-container">
      <div className="stat-card glass-panel">
        <div className="stat-icon" style={{ background: '#3b82f6' }}>
          <Activity />
        </div>
        <div>
          <p className="text-sm text-muted" style={{ margin: 0 }}>เคสทั้งหมด</p>
          <strong className="text-lg">{data.total}</strong>
        </div>
      </div>

      <div className="stat-card glass-panel">
        <div className="stat-icon" style={{ background: '#ef4444' }}>
          <AlertTriangle />
        </div>
        <div>
          <p className="text-sm text-muted" style={{ margin: 0 }}>วิกฤต (P1)</p>
          <strong className="text-lg">{priority.P1 ?? 0}</strong>
        </div>
      </div>

      <div className="stat-card glass-panel">
        <div className="stat-icon" style={{ background: '#f59e0b' }}>
          <Clock />
        </div>
        <div>
          <p className="text-sm text-muted" style={{ margin: 0 }}>รอรับเรื่อง</p>
          <strong className="text-lg">{status.pending ?? 0}</strong>
        </div>
      </div>

      <div className="stat-card glass-panel">
        <div className="stat-icon" style={{ background: '#22c55e' }}>
          <CheckCircle />
        </div>
        <div>
          <p className="text-sm text-muted" style={{ margin: 0 }}>ช่วยเหลือแล้ว</p>
          <strong className="text-lg">{status.resolved ?? 0}</strong>
        </div>
      </div>
    </section>
  );
};

