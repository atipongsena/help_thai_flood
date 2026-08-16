import React, { Suspense, useState } from 'react';
import { MapView } from "./components/MapView";
const CaseCreateForm = React.lazy(() => import("./components/CaseCreateForm").then(module => ({ default: module.CaseCreateForm })));
const CaseDetailDrawer = React.lazy(() => import("./components/CaseDetailDrawer").then(module => ({ default: module.CaseDetailDrawer })));
const StatsPage = React.lazy(() => import("./components/StatsPage"));
import { StatsBar } from "./components/StatsBar";
import { CaseFilters } from "./components/CaseFilters";
import { useAppDispatch } from "./store/hooks";
import { setCaseFormOpen } from "./features/cases/casesSlice";
import { LifeBuoy, Megaphone, BarChart2, Map as MapIcon, Menu, X } from "lucide-react";

function App() {
  const dispatch = useAppDispatch();
  const [view, setView] = useState<'map' | 'stats'>('map');
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="flex-center" style={{ gap: '12px' }}>
          <button 
            className="sidebar-toggle"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            aria-label="Toggle Menu"
          >
            {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
          <div style={{ background: 'linear-gradient(135deg, #2563eb, #1d4ed8)', padding: '8px', borderRadius: '12px', color: 'white' }}>
            <LifeBuoy size={24} />
          </div>
          <div>
            <h1>ไทยช่วยกัน</h1>
            <p className="text-sm text-muted" style={{ margin: 0 }}>ระบบประสานงานช่วยเหลือผู้ประสบภัยน้ำท่วม</p>
          </div>
        </div>
        <div className="header-actions">
          <button 
            className={`btn-secondary ${view === 'map' ? 'active' : ''}`}
            onClick={() => setView('map')}
            title="แผนที่"
          >
            <MapIcon size={18} />
            <span className="hidden sm:inline">แผนที่</span>
          </button>
          <button 
            className={`btn-secondary ${view === 'stats' ? 'active' : ''}`}
            onClick={() => setView('stats')}
            title="สถิติ"
          >
            <BarChart2 size={18} />
            <span className="hidden sm:inline">สถิติ</span>
          </button>
          <button 
            className="btn-primary"
            onClick={() => dispatch(setCaseFormOpen(true))}
          >
            <Megaphone size={18} />
            แจ้งขอความช่วยเหลือ
          </button>
        </div>
      </header>
      {view === 'map' && <StatsBar />}
      <section className="public-note">
        <p>
          {view === 'map' 
            ? "แผนที่ด้านล่างแสดงตำแหน่งคำขอความช่วยเหลือทั้งหมดจาก SOS ล่าสุด (P1 = ด่วนที่สุด, P2 = ด่วนมาก, P3 = เฝ้าระวัง) เพื่อให้ประชาชนติดตามสถานการณ์ได้แบบเรียลไทม์"
            : "ข้อมูลสถิตินี้มาจากจำนวนที่ได้รับแจ้งในระบบ ไม่ได้แสดงถึงตัวเลขผู้ประสบภัยจริงทั้งหมดในสถานการณ์ปัจจุบัน โปรดใช้วิจารณญาณในการอ้างอิง"
          }
        </p>
      </section>
      <section className="app-body">
        {view === 'map' ? (
          <>
            <div className="map-panel">
              <MapView />
            </div>
            <aside className={`app-sidebar ${sidebarOpen ? 'open' : ''}`}>
              <div className="sidebar-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 600 }}>ตัวกรอง & รายการ</h3>
                <button 
                  className="close-btn md:hidden"
                  onClick={() => setSidebarOpen(false)}
                  style={{ display: 'none' }}
                >
                  <X size={20} />
                </button>
              </div>
              <CaseFilters />
            </aside>
            {/* Overlay for mobile sidebar */}
            {sidebarOpen && (
              <div 
                className="sidebar-overlay"
                style={{
                  position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)', zIndex: 1200
                }}
                onClick={() => setSidebarOpen(false)}
              />
            )}
          </>
        ) : (
          <Suspense fallback={<div className="p-8 text-center">กำลังโหลดหน้าสถิติ...</div>}>
            <StatsPage />
          </Suspense>
        )}
      </section>
      <Suspense fallback={null}>
        <CaseDetailDrawer />
        <CaseCreateForm />
      </Suspense>
    </div>
  );
}

export default App;
