import { useEffect, useRef, type ReactNode } from 'react';
import { CaretDownIcon, FunnelSimpleIcon, MagnifyingGlassIcon } from '@phosphor-icons/react';
import { useSettings } from '../../features/settings/SettingsProvider';

export function ListToolbar({ label, placeholder, query, onQueryChange, children, activeFilters = 0,
  count, total, onReset, disabled = false }: {
  label: string; placeholder: string; query: string; onQueryChange: (value: string) => void;
  children: ReactNode; activeFilters?: number; count: number; total: number;
  onReset: () => void; disabled?: boolean;
}) {
  const { tr } = useSettings();
  const menu = useRef<HTMLDetailsElement>(null);
  useEffect(() => {
    function dismiss(event: PointerEvent) {
      if (menu.current?.open && !menu.current.contains(event.target as Node)) menu.current.open = false;
    }
    function escape(event: KeyboardEvent) {
      if (event.key === 'Escape' && menu.current?.open) {
        menu.current.open = false;
        menu.current.querySelector('summary')?.focus();
      }
    }
    document.addEventListener('pointerdown', dismiss);
    document.addEventListener('keydown', escape);
    return () => { document.removeEventListener('pointerdown', dismiss); document.removeEventListener('keydown', escape); };
  }, []);
  return <div className="list-toolbar">
    <label className="list-search"><MagnifyingGlassIcon size={16} aria-hidden="true" />
      <input type="search" aria-label={label} placeholder={placeholder} value={query}
        maxLength={200} onChange={event => onQueryChange(event.target.value)} />
    </label>
    <details className="list-filters" ref={menu}>
      <summary><FunnelSimpleIcon size={15} aria-hidden="true" />{tr('筛选', 'Filter')}
        {activeFilters > 0 && <b>{activeFilters}</b>}<CaretDownIcon size={12} aria-hidden="true" /></summary>
      <div className="list-filter-fields">{children}
        <button type="button" onClick={onReset} disabled={disabled}>{tr('重置', 'Reset')}</button>
      </div>
    </details>
    <small className="list-match-count" role="status">{count} / {total}</small>
  </div>;
}
