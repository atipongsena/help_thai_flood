import { changeFilters, selectFilters } from '../features/cases/casesSlice';
import { useAppDispatch, useAppSelector } from '../store/hooks';

const PRIORITY_OPTIONS = [
  { value: 'P1', label: 'P1 - เร่งด่วนสูง' },
  { value: 'P2', label: 'P2 - เร่งด่วน' },
  { value: 'P3', label: 'P3 - เฝ้าระวัง' },
];

const STATUS_OPTIONS = [
  { value: 'pending', label: 'รอการช่วยเหลือ' },
  { value: 'assigned', label: 'กำลังช่วย' },
  { value: 'resolved', label: 'ปิดเคส' },
];

export function CaseFilters() {
  const filters = useAppSelector(selectFilters);
  const dispatch = useAppDispatch();

  const toggle = (field: 'priority' | 'status', value: string) => {
    const list = new Set(filters[field]);
    if (list.has(value)) list.delete(value);
    else list.add(value);
    dispatch(changeFilters({ [field]: Array.from(list) }));
  };

  return (
    <div className="filter-card">
      <h3>ตัวกรอง</h3>
      <div className="filter-group">
        <p>Priority</p>
        {PRIORITY_OPTIONS.map((opt) => (
          <label key={opt.value}>
            <input
              type="checkbox"
              checked={filters.priority.includes(opt.value)}
              onChange={() => toggle('priority', opt.value)}
            />
            {opt.label}
          </label>
        ))}
      </div>
      <div className="filter-group">
        <p>Status</p>
        {STATUS_OPTIONS.map((opt) => (
          <label key={opt.value}>
            <input
              type="checkbox"
              checked={filters.status.includes(opt.value)}
              onChange={() => toggle('status', opt.value)}
            />
            {opt.label}
          </label>
        ))}
      </div>
    </div>
  );
}

