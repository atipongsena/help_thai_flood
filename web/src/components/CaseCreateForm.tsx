import { FormEvent, useState, useEffect } from 'react';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import { selectCreateFormOpen, setCaseFormOpen, selectNewCaseLocation, setNewCaseLocation } from '../features/cases/casesSlice';
import { useCreateCaseMutation } from '../features/cases/casesApi';
import { X, MapPin, Phone, User, FileText, Navigation } from 'lucide-react';

const initialState = {
  text: '',
  address: '',
  lat: '',
  lng: '',
  contactName: '',
  contactPhone: '',
};

export const CaseCreateForm = () => {
  const isOpen = useAppSelector(selectCreateFormOpen);
  const newLocation = useAppSelector(selectNewCaseLocation);
  const dispatch = useAppDispatch();
  const [form, setForm] = useState(initialState);
  const [createCase, { isLoading }] = useCreateCaseMutation();

  useEffect(() => {
    if (isOpen && newLocation) {
      setForm((prev) => ({
        ...prev,
        lat: newLocation.lat.toFixed(6),
        lng: newLocation.lng.toFixed(6),
      }));

    }
  }, [isOpen, newLocation]);

  const close = () => {
    dispatch(setCaseFormOpen(false));
    dispatch(setNewCaseLocation(undefined));
    setForm(initialState);
  };

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!form.text.trim()) return;
    await createCase({
      text: form.text,
      address: form.address,
      contact: { name: form.contactName, phone: form.contactPhone },
      location:
        form.lat && form.lng ? { type: 'Point', coordinates: [Number(form.lng), Number(form.lat)] } : undefined,
    }).unwrap();
    close();
  };

  if (!isOpen) return null;

  return (
    <div className="modal-backdrop">
      <form className="modal" onSubmit={handleSubmit}>
        <header className="modal__header">
          <div className="flex-center" style={{ gap: '10px' }}>
            <div style={{ background: '#eff6ff', padding: '8px', borderRadius: '50%', color: '#2563eb' }}>
              <FileText size={20} />
            </div>
            <h3>แจ้งเหตุขอความช่วยเหลือ</h3>
          </div>
          <button type="button" onClick={close} className="close-btn" aria-label="close form">
            <X size={20} />
          </button>
        </header>

        <label>
          <span className="flex-center" style={{ justifyContent: 'flex-start', gap: '6px' }}>
            <FileText size={16} className="text-muted" /> รายละเอียดเหตุการณ์
          </span>
          <textarea
            value={form.text}
            onChange={(e) => setForm({ ...form, text: e.target.value })}
            required
            minLength={20}
            rows={4}
            autoFocus
            placeholder="เช่น น้ำท่วมสูง 2 เมตร ต้องการเรือกู้ภัยด่วน..."
          />
        </label>

        <label>
          <span className="flex-center" style={{ justifyContent: 'flex-start', gap: '6px' }}>
            <MapPin size={16} className="text-muted" /> ที่อยู่ / จุดสังเกต
          </span>
          <input 
            value={form.address} 
            onChange={(e) => setForm({ ...form, address: e.target.value })} 
            placeholder="บ้านเลขที่, ซอย, ถนน, ตำบล..."
          />
        </label>

        <div style={{ marginBottom: '0.5rem' }}>
          <button
            type="button"
            className="btn-secondary"
            style={{ width: '100%', justifyContent: 'center', display: 'flex', alignItems: 'center', gap: '8px' }}
            onClick={() => {
              if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(
                  (position) => {
                    setForm({
                      ...form,
                      lat: position.coords.latitude.toFixed(6),
                      lng: position.coords.longitude.toFixed(6),
                    });
                  },
                  (error) => alert('ไม่สามารถดึงตำแหน่งได้: ' + error.message)
                );
              } else {
                alert('เบราว์เซอร์นี้ไม่รองรับการระบุตำแหน่ง');
              }
            }}
          >
            <Navigation size={16} /> ดึงตำแหน่งปัจจุบัน
          </button>
        </div>

        <div className="modal__grid">
          <label>
            <span className="text-sm text-muted">ละติจูด</span>
            <input
              type="number"
              step="0.0001"
              value={form.lat}
              onChange={(e) => setForm({ ...form, lat: e.target.value })}
              placeholder="0.0000"
            />
          </label>
          <label>
            <span className="text-sm text-muted">ลองจิจูด</span>
            <input
              type="number"
              step="0.0001"
              value={form.lng}
              onChange={(e) => setForm({ ...form, lng: e.target.value })}
              placeholder="0.0000"
            />
          </label>
        </div>

        <div className="modal__grid">
          <label>
            <span className="flex-center" style={{ justifyContent: 'flex-start', gap: '6px' }}>
              <User size={16} className="text-muted" /> ผู้ประสานงาน
            </span>
            <input value={form.contactName} onChange={(e) => setForm({ ...form, contactName: e.target.value })} placeholder="ชื่อ-นามสกุล" />
          </label>
          <label>
            <span className="flex-center" style={{ justifyContent: 'flex-start', gap: '6px' }}>
              <Phone size={16} className="text-muted" /> เบอร์โทร
            </span>
            <input value={form.contactPhone} onChange={(e) => setForm({ ...form, contactPhone: e.target.value })} placeholder="08x-xxx-xxxx" />
          </label>
        </div>

        <button type="submit" disabled={isLoading} className="btn-primary" style={{ justifyContent: 'center', marginTop: '10px' }}>
          {isLoading ? 'กำลังส่งข้อมูล...' : 'ส่งคำขอความช่วยเหลือ'}
        </button>
      </form>
    </div>
  );
};

