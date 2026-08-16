import { useMemo, useState } from 'react';
import { useGetMetricsQuery } from '../features/cases/casesApi';
import { AlertTriangle, Users, Heart, Activity } from 'lucide-react';

const RESOURCE_TAG_MAPPING: Record<string, string> = {
  'rescue_boat': 'เรือกู้ภัย',
  'food_drop': 'อาหาร/น้ำดื่ม',
  'medical_evac': 'เคลื่อนย้ายผู้ป่วย',
  'power_supply': 'ไฟฟ้าสำรอง',
  'sandbag': 'กระสอบทราย',
  'volunteer': 'อาสาสมัคร',
  'other': 'อื่นๆ'
};

const StatsPage = () => {
  const { data: stats, isLoading, error } = useGetMetricsQuery();

  const chartData = useMemo(() => {
    if (!stats?.tags) return [];
    
    return Object.entries(stats.tags)
      .map(([tag, count]) => {
        const label = RESOURCE_TAG_MAPPING[tag] || tag;
        return { label, count: Number(count) };
      })
      .sort((a, b) => b.count - a.count)
      .slice(0, 5);
  }, [stats]);

  if (isLoading) {
    return <div className="stats-loading">กำลังโหลดข้อมูล...</div>;
  }

  if (error) {
    return (
      <div className="stats-loading" style={{ color: 'red' }}>
        <p>เกิดข้อผิดพลาดในการโหลดข้อมูล</p>
        <pre style={{ fontSize: '0.8rem' }}>{JSON.stringify(error, null, 2)}</pre>
      </div>
    );
  }

  if (!stats) return null;

  return (
    <div className="stats-page">
      <div className="stats-header">
        <h2 className="stats-title">ศูนย์ติดตามภัยพิบัติ</h2>
        <div className="stats-total-badge">
          เคสทั้งหมด: {stats.total.toLocaleString()}
        </div>
      </div>

      {/* Summary Cards */}
      <div className="stats-grid">
        <div className="stats-card card-critical">
          <div className="card-header">
            <span className="card-label">วิกฤต</span>
            <AlertTriangle size={20} />
          </div>
          <div className="card-value">{(stats.priority?.P1 || 0).toLocaleString()}</div>
          <div className="card-subtext">ต้องการความช่วยเหลือทันที</div>
        </div>

        <div className="stats-card card-affected">
          <div className="card-header">
            <span className="card-label">ผู้ได้รับผลกระทบ</span>
            <Users size={20} />
          </div>
          <div className="card-value">{(stats.affected || 0).toLocaleString()}</div>
          <div className="card-subtext">คน (รวมเด็ก/ผู้สูงอายุ)</div>
        </div>

        <div className="stats-card card-medical">
          <div className="card-header">
            <span className="card-label">ค่าขอทางการแพทย์</span>
            <Heart size={20} />
          </div>
          <div className="card-value">{(stats.medical || 0).toLocaleString()}</div>
          <div className="card-subtext">เคสที่ต้องการแพทย์</div>
        </div>

        <div className="stats-card card-active">
          <div className="card-header">
            <span className="card-label">เคสที่เปิดอยู่</span>
            <Activity size={20} />
          </div>
          <div className="card-value">{(stats.status?.pending || 0).toLocaleString()}</div>
          <div className="card-subtext">รอการตรวจสอบ/ช่วยเหลือ</div>
        </div>
      </div>

      {/* Charts Section */}
      <div className="charts-grid">
        {/* Bar Chart */}
        <div className="stats-card chart-card">
          <h3 className="chart-title">ประเภทคำขอ</h3>
          <div className="bar-chart-container">
            {chartData.map(({ label, count }) => (
              <div key={label} className="bar-row">
                <div className="bar-label-row">
                  <span>{label}</span>
                  <span className="bar-count">{count.toLocaleString()}</span>
                </div>
                <div className="bar-track">
                  <div 
                    className="bar-fill"
                    style={{ width: `${(count / stats.total) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Donut Chart */}
        <div className="stats-card chart-card donut-card">
          <h3 className="chart-title">การกระจายความเร่งด่วน</h3>
          <DonutChart stats={stats} />
        </div>
      </div>
    </div>
  );
};

const DonutChart = ({ stats }: { stats: any }) => {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

  const segments = useMemo(() => {
    const total = stats.total || 1;
    const p1 = stats.priority?.P1 || 0;
    const p2 = stats.priority?.P2 || 0;
    const p3 = stats.priority?.P3 || 0;
    const resolved = stats.status?.resolved || 0;

    return [
      { label: 'วิกฤต', color: '#ef4444', value: p1, percent: (p1 / total) * 100 },
      { label: 'เร่งด่วน', color: '#f97316', value: p2, percent: (p2 / total) * 100 },
      { label: 'เฝ้าระวัง', color: '#eab308', value: p3, percent: (p3 / total) * 100 },
      { label: 'สำเร็จ', color: '#22c55e', value: resolved, percent: (resolved / total) * 100 },
    ];
  }, [stats]);

  let offset = 0;

  const handleMouseMove = (e: React.MouseEvent) => {
    setMousePos({ x: e.clientX, y: e.clientY });
  };

  return (
    <div style={{ position: 'relative' }} onMouseMove={handleMouseMove}>
      <div className="donut-container">
        <svg viewBox="0 0 36 36" className="donut-svg">
          <path
            className="donut-bg"
            d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
          />
          {segments.map((seg, i) => {
            const dashArray = `${seg.percent}, 100`;
            const currentOffset = offset;
            offset += seg.percent;

            return (
              <path
                key={i}
                className="donut-segment"
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                stroke={seg.color}
                strokeDasharray={dashArray}
                strokeDashoffset={-currentOffset}
                onMouseEnter={() => setHoveredIndex(i)}
                onMouseLeave={() => setHoveredIndex(null)}
                style={{
                  opacity: hoveredIndex !== null && hoveredIndex !== i ? 0.3 : 1,
                  cursor: 'pointer',
                  transition: 'all 0.3s ease',
                  strokeWidth: hoveredIndex === i ? 4 : 2.8 // Thicker on hover
                }}
              />
            );
          })}
        </svg>
        <div className="donut-center">
          <span className="donut-total" style={{ color: hoveredIndex !== null ? segments[hoveredIndex].color : 'inherit' }}>
            {hoveredIndex !== null ? segments[hoveredIndex].value.toLocaleString() : stats.total.toLocaleString()}
          </span>
          <span className="donut-label">
            {hoveredIndex !== null ? segments[hoveredIndex].label : 'เคสทั้งหมด'}
          </span>
        </div>
      </div>
      
      {/* Tooltip */}
      {hoveredIndex !== null && (
        <div 
          style={{
            position: 'fixed',
            left: mousePos.x + 15,
            top: mousePos.y - 15,
            background: 'rgba(0, 0, 0, 0.8)',
            color: 'white',
            padding: '8px 12px',
            borderRadius: '6px',
            fontSize: '14px',
            pointerEvents: 'none',
            zIndex: 9999, // Increased z-index
            border: `1px solid ${segments[hoveredIndex].color}`,
            boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
            backdropFilter: 'blur(4px)',
            whiteSpace: 'nowrap'
          }}
        >
          <div style={{ fontWeight: 'bold', marginBottom: '2px' }}>{segments[hoveredIndex].label}</div>
          <div>{segments[hoveredIndex].value.toLocaleString()} เคส</div>
          <div style={{ fontSize: '12px', opacity: 0.8 }}>({segments[hoveredIndex].percent.toFixed(1)}%)</div>
        </div>
      )}

      <div className="donut-legend">
        {segments.map((seg, i) => (
          <div 
            key={i} 
            className="legend-item" 
            onMouseEnter={() => setHoveredIndex(i)}
            onMouseLeave={() => setHoveredIndex(null)}
            style={{ opacity: hoveredIndex !== null && hoveredIndex !== i ? 0.3 : 1, cursor: 'pointer', transition: 'opacity 0.3s' }}
          >
            <span className="dot" style={{ backgroundColor: seg.color }}></span>
            {seg.label}
          </div>
        ))}
      </div>
    </div>
  );
};


export default StatsPage;
