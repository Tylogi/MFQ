/** 组织业务面板网格，并按页面持久化每个面板的折叠状态。 */
import { Children, isValidElement, ReactNode, useState } from 'react';

export const PANEL_COLLAPSED_KEY = 'mfq.studio.panel-collapsed.v2';

/** 读取本地面板折叠状态，数据异常时返回空配置。 */
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

/** 将 React 子元素键转换为稳定的面板持久化键。 */
export function panelKey(panel: React.ReactElement, index: number): string {
  const value = String(panel.key ?? `panel-${index}`);
  return value.startsWith('.$') ? value.slice(2) : value.startsWith('.') ? value.slice(1) : value;
}

/** 按页面组织业务面板，并持久化每个面板的折叠状态。 */
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
