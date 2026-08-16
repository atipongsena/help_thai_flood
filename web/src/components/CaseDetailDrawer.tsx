import { useMemo, useState, useEffect } from 'react';
import { useGetCaseDetailQuery, useUpdateCaseMutation } from '../features/cases/casesApi';
import { selectSelectedCase, setSelectedCase } from '../features/cases/casesSlice';
import { useAppDispatch, useAppSelector } from '../store/hooks';

const statusLabels: Record<string, string> = {
  pending: 'รอการช่วยเหลือ',
  assigned: 'กำลังดำเนินการ',
  resolved: 'ปิดเคสแล้ว',
};

const priorityLabels: Record<string, string> = {
  P1: 'ด่วนที่สุด (ชีวิตเสี่ยงสูง)',
  P2: 'ด่วนมาก (ต้องติดตาม)',
  P3: 'เฝ้าระวัง/ติดตาม',
};

const resourceTagLabels: Record<string, string> = {
  medical_evac: 'ส่งแพทย์ / เคลื่อนย้ายผู้ป่วย',
  food_drop: 'อาหารและน้ำดื่ม',
  rescue_boat: 'เรือกู้ภัย / ขนย้าย',
  body_recovery: 'ค้นหาผู้สูญหาย',
  power_supply: 'พาวเวอร์แบงก์ / ไฟฟ้าชั่วคราว',
};

const riskLabels: Record<string, string> = {
  has_children: 'มีเด็กเล็ก',
  has_elderly: 'มีผู้สูงอายุ',
  has_disabled: 'มีผู้ป่วยติดเตียง',
  trapped: 'ติดอยู่ชั้นบน/ออกไม่ได้',
  power_outage: 'ไฟฟ้าดับ',
  no_food: 'ขาดอาหาร',
  no_water: 'ขาดน้ำสะอาด',
  needs_transport: 'ต้องการพาหนะขนย้าย',
};

export const CaseDetailDrawer = () => {
  const selectedId = useAppSelector(selectSelectedCase);
  const dispatch = useAppDispatch();
  const { data, isFetching } = useGetCaseDetailQuery(selectedId ?? '', { skip: !selectedId });
  const [updateCase, { isLoading: isUpdating }] = useUpdateCaseMutation();

  const [isEditing, setIsEditing] = useState(false);
  const [editForm, setEditForm] = useState({
    status: '',
    priority: '',
    text: '',
  });

  // Initialize form when data loads or editing starts
  useEffect(() => {
    if (data) {
      setEditForm({
        status: data.status,
        priority: data.priority_label ?? 'P3',
        text: data.text,
      });
    }
  }, [data, isEditing]);

  const riskEntries = useMemo(() => {
    if (!data?.risk_flags) return [];
    return Object.entries(data.risk_flags)
      .filter(([, value]) => Boolean(value))
      .map(([key]) => riskLabels[key] ?? key.replace(/_/g, ' '));
  }, [data]);

  const resourceEntries = useMemo(() => {
    if (!data?.resource_tags?.length) return [];
    return data.resource_tags.map((tag) => resourceTagLabels[tag] ?? tag.replace(/_/g, ' '));
  }, [data]);

  if (!selectedId) return null;

  const close = () => {
    setIsEditing(false);
    dispatch(setSelectedCase(undefined));
  };

  const handleSave = async () => {
    if (!selectedId) return;
    try {
      await updateCase({
        id: selectedId,
        patch: {
          status: editForm.status as 'pending' | 'assigned' | 'resolved',
          priority_label: editForm.priority as 'P1' | 'P2' | 'P3',
          text: editForm.text,
        },
      }).unwrap();
      setIsEditing(false);
    } catch (error) {
      console.error('Failed to update case:', error);
      alert('เกิดข้อผิดพลาดในการบันทึกข้อมูล');
    }
  };

  const metadataTypeName =
    data && typeof data.metadata?.['type_name'] === 'string' ? (data.metadata['type_name'] as string) : undefined;

  return (
    <div className="case-detail-overlay" onClick={(e) => {
      if (e.target === e.currentTarget) close();
    }}>
      <div className="case-detail-modal">
        <header className="case-detail-header">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <span className="text-sm text-muted font-mono">#{data?._id.slice(-6)}</span>
              {!isEditing && (
                <div className={`status-badge status-${data?.status}`}>
                  {statusLabels[data?.status ?? ''] ?? data?.status}
                </div>
              )}
            </div>
            {!isEditing ? (
              <h2 className="text-xl font-bold m-0">
                {priorityLabels[data?.priority_label ?? ''] ?? data?.priority_label}
              </h2>
            ) : (
              <h2 className="text-xl font-bold m-0">แก้ไขข้อมูลเคส</h2>
            )}
          </div>
          <div className="flex items-center gap-2">
            {!isEditing && (
              <button 
                onClick={() => setIsEditing(true)}
                className="btn-secondary text-sm px-3 py-1"
              >
                แก้ไข
              </button>
            )}
            <button onClick={close} className="close-btn" aria-label="close detail">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18"></line>
                <line x1="6" y1="6" x2="18" y2="18"></line>
              </svg>
            </button>
          </div>
        </header>

        <div className="case-detail-content">
          {isFetching && <p className="p-4 text-muted text-center">กำลังโหลดข้อมูล...</p>}

          {data && !isEditing && (
            <>
              <div className="case-detail-section">
                <strong>รายละเอียด</strong>
                <p className="case-detail-text">{data.text}</p>
              </div>

              <div className="case-detail-grid">
                <div className="case-detail-section">
                  <strong>จุดเกิดเหตุ</strong>
                  <p>{data.address || metadataTypeName || 'ไม่ระบุที่อยู่'}</p>
                  <div className="flex flex-col gap-1 mt-2">
                    {data.location?.coordinates && (
                      <small className="text-muted font-mono block">
                        Lat: {data.location.coordinates[1].toFixed(6)}, Lng: {data.location.coordinates[0].toFixed(6)}
                      </small>
                    )}
                    {data.running_number && <small className="text-muted block">Ref: {data.running_number}</small>}
                    <small className="text-muted block">แจ้งเมื่อ: {new Date(data.createdAt).toLocaleString('th-TH')}</small>
                  </div>
                </div>

                <div className="case-detail-section">
                  <strong>ข้อมูลติดต่อ</strong>
                  {data.contact?.phone ? (
                    <p>
                      {data.contact?.name && <span>{data.contact.name} <br/></span>}
                      <span className="text-lg font-mono">{data.contact.phone}</span>
                    </p>
                  ) : (
                    <p className="text-muted">- ไม่มีข้อมูลติดต่อ -</p>
                  )}
                  {data.assigned_team && (
                    <div className="mt-2 p-2 bg-blue-50 rounded border border-blue-100">
                      <small className="text-blue-800 font-semibold">ทีมที่รับผิดชอบ</small>
                      <p className="text-blue-900 m-0">{data.assigned_team}</p>
                    </div>
                  )}
                </div>
              </div>

              <div className="case-detail-grid">
                {riskEntries.length > 0 && (
                  <div className="case-detail-section">
                    <strong>ความเสี่ยง</strong>
                    <div className="chips">
                      {riskEntries.map((label) => (
                        <span key={label} style={{ background: '#fee2e2', color: '#991b1b' }}>{label}</span>
                      ))}
                    </div>
                  </div>
                )}

                {resourceEntries.length > 0 && (
                  <div className="case-detail-section">
                    <strong>สิ่งที่ต้องการ</strong>
                    <div className="chips">
                      {resourceEntries.map((label) => (
                        <span key={label} style={{ background: '#e0f2fe', color: '#075985' }}>{label}</span>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {data.people && Object.values(data.people).some((count) => Number(count) > 0) && (
                <div className="case-detail-section">
                  <strong>ผู้ประสบภัย</strong>
                  <div className="flex gap-4 p-3 bg-slate-50 rounded-lg overflow-x-auto">
                    <div className="text-center min-w-[60px]">
                      <div className="text-xl font-bold">{data.people.adults ?? 0}</div>
                      <div className="text-xs text-muted">ผู้ใหญ่</div>
                    </div>
                    <div className="text-center min-w-[60px]">
                      <div className="text-xl font-bold">{data.people.children ?? 0}</div>
                      <div className="text-xs text-muted">เด็ก</div>
                    </div>
                    <div className="text-center min-w-[60px]">
                      <div className="text-xl font-bold">{data.people.infants ?? 0}</div>
                      <div className="text-xs text-muted">ทารก</div>
                    </div>
                    <div className="text-center min-w-[60px]">
                      <div className="text-xl font-bold">{data.people.elderly ?? 0}</div>
                      <div className="text-xs text-muted">ผู้สูงอายุ</div>
                    </div>
                  </div>
                </div>
              )}

              {data.notes && (
                <div className="case-detail-section">
                  <strong>หมายเหตุเพิ่มเติม</strong>
                  <p className="text-muted bg-gray-50 p-3 rounded">{data.notes}</p>
                </div>
              )}

              {data.history && data.history.length > 0 && (
                <div className="case-detail-section">
                  <strong>ประวัติการดำเนินการ</strong>
                  <ul className="history">
                    {data.history
                      .slice()
                      .reverse()
                      .map((entry, idx) => (
                        <li key={`${entry.at}-${idx}`}>
                          <span>{new Date(entry.at ?? '').toLocaleString('th-TH')}</span>
                          <p>{entry.message ?? entry.action}</p>
                          {entry.by && <small className="text-muted">โดย: {entry.by}</small>}
                        </li>
                      ))}
                  </ul>
                </div>
              )}

              {data.status !== 'resolved' && (
                <div className="pt-4 border-t mt-4">
                  <button
                    onClick={async () => {
                      if (confirm('ยืนยันว่าได้รับความช่วยเหลือแล้ว?')) {
                        try {
                          await updateCase({
                            id: data._id,
                            patch: { status: 'resolved' }
                          }).unwrap();
                        } catch (err) {
                          console.error(err);
                          alert('เกิดข้อผิดพลาด');
                        }
                      }
                    }}
                    className="btn-primary w-full justify-center py-3 text-lg shadow-md hover:shadow-lg transform hover:-translate-y-0.5 transition-all"
                    style={{ background: 'linear-gradient(135deg, #16a34a 0%, #15803d 100%)' }}
                  >
                    ✓ แจ้งได้รับความช่วยเหลือแล้ว
                  </button>
                </div>
              )}
            </>
          )}


          {data && isEditing && (
            <div className="drawer__form">
              <div className="form-group">
                <label>รายละเอียด</label>
                <textarea
                  value={editForm.text}
                  onChange={(e) => setEditForm({ ...editForm, text: e.target.value })}
                  className="form-textarea"
                  rows={8}
                />
              </div>

              <div className="form-actions pt-4 border-t">
                <button 
                  onClick={() => setIsEditing(false)}
                  className="btn-secondary"
                  disabled={isUpdating}
                >
                  ยกเลิก
                </button>
                <button 
                  onClick={handleSave}
                  className="btn-primary justify-center"
                  disabled={isUpdating}
                >
                  {isUpdating ? 'กำลังบันทึก...' : 'บันทึกการเปลี่ยนแปลง'}
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

