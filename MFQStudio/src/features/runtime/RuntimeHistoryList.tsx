/** Render accessible scrolling history with incremental loading, retry, and end-of-history feedback. */
import type { ReactNode } from 'react';
import type { useRuntimeHistory } from './useRuntimeHistory';

/** Keep each history panel scrollable and expose a button fallback to automatic bottom loading. */
export function RuntimeHistoryList<T extends { sequence: number }>({
  history,
  label,
  empty,
  renderEntry,
  tr,
}: {
  history: ReturnType<typeof useRuntimeHistory<T>>;
  label: string;
  empty: string;
  renderEntry: (entry: T) => ReactNode;
  tr: (zh: string, en: string) => string;
}) {
  const { items, scrollerRef, onScroll, loading, error, liveError, hasMore, loadMore, keyOf } = history;
  return (
    <div
      className="runtime-history"
      ref={scrollerRef}
      onScroll={onScroll}
      role="region"
      aria-label={label}
      tabIndex={0}
    >
      {liveError && <div role="status">{tr('实时连接已断开，正在重连…', 'Live connection lost. Reconnecting…')}</div>}
      {items.map((entry) => (
        <div key={keyOf(entry)} data-history-id={keyOf(entry)}>
          {renderEntry(entry)}
        </div>
      ))}
      {!items.length && !loading && !error && <div className="inline-empty">{empty}</div>}
      <div className="runtime-history-footer">
        {error ? (
          <>
            <span role="alert">{tr('历史记录加载失败', 'History could not be loaded')}</span>
            <button type="button" onClick={() => void loadMore()}>
              {tr('重试', 'Retry')}
            </button>
          </>
        ) : loading ? (
          <span role="status">{tr('加载中…', 'Loading…')}</span>
        ) : hasMore ? (
          <button type="button" onClick={() => void loadMore()}>
            {tr('加载更早记录', 'Load older records')}
          </button>
        ) : items.length > 0 ? (
          <span>{tr('没有更早记录了', 'No older records')}</span>
        ) : null}
      </div>
    </div>
  );
}
