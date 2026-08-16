import { useMemo } from 'react';
import { useGetBoardCasesQuery } from '../features/cases/casesApi';
import { setSelectedCase } from '../features/cases/casesSlice';
import { useAppDispatch } from '../store/hooks';

const columns = [
  { key: 'pending', label: 'รอจัดการ' },
  { key: 'assigned', label: 'กำลังดำเนินการ' },
  { key: 'resolved', label: 'ปิดเคส' },
] as const;

export const AssignmentBoard = () => {
  const dispatch = useAppDispatch();
  const { data: cases = [], isFetching } = useGetBoardCasesQuery(120);

  const grouped = useMemo(() => {
    return columns.reduce<Record<string, typeof cases>>((acc, col) => {
      acc[col.key] = cases.filter((item) => item.status === col.key);
      return acc;
    }, {} as Record<string, typeof cases>);
  }, [cases]);

  return (
    <section className="board-card">
      <header>
        <h3>สถานะทีมปฏิบัติการ</h3>
        {isFetching && <span className="muted">กำลังโหลด...</span>}
      </header>
      <div className='board-columns'>
        {columns.map((col) => (
          <div key={col.key} className='board-column'>
            <h4>{col.label}</h4>
            <small>{grouped[col.key]?.length ?? 0} เคส</small>
            <ul>
              {(grouped[col.key] ?? []).slice(0, 6).map((item) => (
                <li key={item._id} onClick={() => dispatch(setSelectedCase(item._id))}>
                  <div className='board-pill'>{item.priority_label}</div>
                  <div>
                    <strong>#{item._id.slice(-6)}</strong>
                    <p>{item.assigned_team || 'ยังไม่มอบหมาย'}</p>
                  </div>
                </li>
              ))}
              {(grouped[col.key] ?? []).length === 0 && <li className='muted empty'>ไม่มีเคส</li>}
            </ul>
          </div>
        ))}
      </div>
    </section>
  );
};

