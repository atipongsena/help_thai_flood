import { useGetCasesQuery } from '../features/cases/casesApi';
import { selectFilters, selectSelectedCase, setCaseFormOpen, setSelectedCase } from '../features/cases/casesSlice';
import { useAppDispatch, useAppSelector } from '../store/hooks';

export function CaseSidebar() {
  const filters = useAppSelector(selectFilters);
  const selectedId = useAppSelector(selectSelectedCase);
  const dispatch = useAppDispatch();
  const { data: cases = [], isFetching } = useGetCasesQuery(filters);

  return (
    <div className="sidebar-card">
      <header>
        <h3>รายการ ({cases.length})</h3>
        {isFetching && <span className="muted">กำลังโหลด...</span>}
        <button className="link" onClick={() => dispatch(setCaseFormOpen(true))}>
          + แจ้งเหตุ
        </button>
      </header>
      <ul className="case-list">
        {cases.map((item) => (
          <li
            key={item._id}
            className={item._id === selectedId ? 'active' : ''}
            onClick={() => dispatch(setSelectedCase(item._id))}
          >
            <div>
              <strong>{item.priority_label}</strong> · <small>{item.status}</small>
            </div>
            <p>{item.text.slice(0, 120)}...</p>
            <div className="tag-row">
              {item.resource_tags?.slice(0, 3).map((tag) => (
                <span key={tag} className="tag">
                  {tag}
                </span>
              ))}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

