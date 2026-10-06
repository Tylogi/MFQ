/** Render accessible scrolling history with incremental loading, retry, and end-of-history feedback. */
import type { TFunction } from 'i18next';
import type { ReactNode } from 'react';
import type { useRuntimeHistory } from './useRuntimeHistory';

/** Keep each history panel scrollable and expose a button fallback to automatic bottom loading. */
export function RuntimeHistoryList<T extends { sequence: number }>({
  history,
  label,
  empty,
  renderEntry,
  t,
}: {
  history: ReturnType<typeof useRuntimeHistory<T>>;
  label: string;
  empty: string;
  renderEntry: (entry: T) => ReactNode;
  t: TFunction;
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
      {liveError && <div role="status">{t('runtime:runtimeHistoryList.liveConnectionLostReconnecting')}</div>}
      {items.map((entry) => (
        <div key={keyOf(entry)} data-history-id={keyOf(entry)}>
          {renderEntry(entry)}
        </div>
      ))}
      {!items.length && !loading && !error && <div className="inline-empty">{empty}</div>}
      <div className="runtime-history-footer">
        {error ? (
          <>
            <span role="alert">{t('runtime:runtimeHistoryList.historyCouldNotBeLoaded')}</span>
            <button type="button" onClick={() => void loadMore()}>
              {t('common:retry')}
            </button>
          </>
        ) : loading ? (
          <span role="status">{t('runtime:runtimeHistoryList.loading')}</span>
        ) : hasMore ? (
          <button type="button" onClick={() => void loadMore()}>
            {t('runtime:runtimeHistoryList.loadOlderRecords')}
          </button>
        ) : items.length > 0 ? (
          <span>{t('runtime:runtimeHistoryList.noOlderRecords')}</span>
        ) : null}
      </div>
    </div>
  );
}
