/** Arrange the business-panel grid and persist each panel’s collapsed state per page. */
import { Children, isValidElement, ReactNode, useState } from 'react';

export const PANEL_COLLAPSED_KEY = 'mfq.studio.panel-collapsed.v2';
/** Read locally stored panel-collapse state, returning an empty configuration if the data is invalid. */
export function loadCollapsedPanels(): Record<string, boolean> {
  try {
    const value = JSON.parse(localStorage.getItem(PANEL_COLLAPSED_KEY) || '{}');
    return value && typeof value === 'object' ? value as Record<string, boolean> : {};
  } catch {
    return {};
  }
}

export interface PanelDeckProps {
  page: string;
  children: ReactNode;
  labels: { collapse: string; expand: string };
}
/** Convert React child keys into stable panel-persistence keys. */
export function panelKey(panel: React.ReactElement, index: number): string {
  const value = String(panel.key ?? `panel-${index}`);
  return value.startsWith('.$') ? value.slice(2) : value.startsWith('.') ? value.slice(1) : value;
}
/** Organize business panels by page and persist each panel’s collapsed state. */
export function PanelDeck({ page, children, labels }: PanelDeckProps) {
  const panels = Children.toArray(children).filter(isValidElement);
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>(loadCollapsedPanels);
  const panelClasses = `panel-deck page-${page}${panels.length === 1 ? ' single' : ''}`;

  return (
    <div className={panelClasses}>
      {panels.map((panel, index) => {
        const id = panelKey(panel, index);
        const storageKey = `${page}:${id}`;
        const isCollapsed = Boolean(collapsed[storageKey]);
        return (
          <div className={`panel-shell${panels.length === 1 ? ' wide' : ''}${isCollapsed ? ' collapsed' : ''}`} data-panel-id={id} key={id}>
            <button
              aria-expanded={!isCollapsed}
              aria-label={isCollapsed ? labels.expand : labels.collapse}
              className="panel-collapse"
              onClick={() => setCollapsed((current) => {
                const updated = { ...current, [storageKey]: !isCollapsed };
                localStorage.setItem(PANEL_COLLAPSED_KEY, JSON.stringify(updated));
                return updated;
              })}
              type="button"
            >
              <span aria-hidden="true">⌄</span>
            </button>
            {panel}
          </div>
        );
      })}
    </div>
  );
}
