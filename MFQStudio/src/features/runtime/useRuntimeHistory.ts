/** Page runtime history in both directions while preserving live updates and the reading position. */
import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { errorMessage } from '../../app/formatters';
import type { RuntimeHistoryQuery } from '../../shared/api/resources/runtime';

const PAGE_SIZE = 50;
type Sequenced = { sequence: number };
type PageLoader<T> = (
  limit: number,
  query: RuntimeHistoryQuery,
  signal: AbortSignal,
) => Promise<T[]>;

/** Use a stable sequence identity for append-only runtime events. */
function sequenceKey(entry: Sequenced): string {
  return String(entry.sequence);
}

/** Load the newest page, append older pages on demand, and stream new rows without losing loaded history. */
export function useRuntimeHistory<T extends Sequenced>(
  ready: boolean,
  connectionRevision: number,
  fetchPage: PageLoader<T>,
  subscribe: (after: number, onEvent: (entry: T) => void, signal: AbortSignal) => Promise<void>,
  keyOf: (entry: T) => string = sequenceKey,
) {
  const [items, setItems] = useState<T[]>([]);
  const [loading, setLoading] = useState(false);
  const [hasMore, setHasMore] = useState(false);
  const [liveError, setLiveError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const scrollerRef = useRef<HTMLDivElement>(null);
  const loadRef = useRef<() => Promise<void>>(async () => {});
  const anchor = useRef<{ id: string; offset: number } | null>(null);

  useLayoutEffect(() => {
    const node = scrollerRef.current;
    const saved = anchor.current;
    anchor.current = null;
    if (!node || !saved) return;
    const row = Array.from(node.querySelectorAll<HTMLElement>('[data-history-id]')).find(
      (candidate) => candidate.dataset.historyId === saved.id,
    );
    if (row) node.scrollTop += row.getBoundingClientRect().top - saved.offset;
  }, [items]);

  useEffect(() => {
    setItems([]);
    setLiveError(null);
    setHasMore(false);
    setError(null);
    setLoading(false);
    anchor.current = null;
    if (scrollerRef.current) scrollerRef.current.scrollTop = 0;
    if (!ready) {
      loadRef.current = async () => {};
      return;
    }
    const controller = new AbortController();
    let initialized = false;
    let oldest: number | undefined;
    let newest = 0;
    let more = false;
    let loadingOlder = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    /** Merge bounded API pages, preserving the first visible row when new entries are prepended. */
    function merge(page: T[], live: boolean) {
      if (!page.length) return;
      if (live) {
        const node = scrollerRef.current;
        if (node && node.scrollTop > 24 && !anchor.current) {
          const top = node.getBoundingClientRect().top;
          const row = Array.from(node.querySelectorAll<HTMLElement>('[data-history-id]')).find(
            (candidate) => candidate.getBoundingClientRect().bottom > top,
          );
          if (row?.dataset.historyId)
            anchor.current = {
              id: row.dataset.historyId,
              offset: row.getBoundingClientRect().top,
            };
        }
      }
      setItems((current) =>
        [...new Map([...current, ...page].map((entry) => [keyOf(entry), entry])).values()].sort(
          (a, b) => b.sequence - a.sequence,
        ),
      );
    }

    /** Fetch one older page; concurrent bottom-scroll events share the same in-flight request. */
    async function loadOlder() {
      if (controller.signal.aborted || loadingOlder || (initialized && !more)) return;
      loadingOlder = true;
      setLoading(true);
      setError(null);
      try {
        const page = await fetchPage(
          PAGE_SIZE,
          { before: oldest, order: 'desc' },
          controller.signal,
        );
        if (controller.signal.aborted) return;
        merge(page, false);
        if (page.length) {
          oldest = Math.min(...page.map((entry) => entry.sequence));
          newest = Math.max(newest, ...page.map((entry) => entry.sequence));
        }
        initialized = true;
        more = page.length === PAGE_SIZE;
        setHasMore(more);
      } catch (cause) {
        if (!controller.signal.aborted) setError(errorMessage(cause));
      } finally {
        loadingOlder = false;
        if (!controller.signal.aborted) setLoading(false);
      }
    }

    let retryDelay = 1000;
    /** Resume from the last applied event after EOF or failure; never poll history for live updates. */
    async function connect() {
      try {
        if (!initialized) await loadOlder();
        if (!initialized || controller.signal.aborted) return;
        await subscribe(newest, (entry) => {
          if (controller.signal.aborted || entry.sequence <= newest) return;
          merge([entry], true);
          newest = entry.sequence;
          if (oldest === undefined) oldest = entry.sequence;
          retryDelay = 1000;
          setLiveError(null);
        }, controller.signal);
        if (!controller.signal.aborted) setLiveError('Stream disconnected');
      } catch (cause) {
        if (!controller.signal.aborted) setLiveError(errorMessage(cause));
      } finally {
        if (!controller.signal.aborted) {
          timer = setTimeout(() => void connect(), retryDelay);
          retryDelay = Math.min(retryDelay * 2, 15000);
        }
      }
    }
    loadRef.current = loadOlder;
    void connect();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [ready, connectionRevision, fetchPage, subscribe, keyOf]);

  /** Request another history page through the current connection's loader. */
  function loadMore() {
    return loadRef.current();
  }
  /** Prefetch when keyboard, touch, or mouse scrolling approaches the history footer. */
  function onScroll() {
    const node = scrollerRef.current;
    if (node && node.scrollHeight - node.scrollTop - node.clientHeight < 100 && !error)
      void loadMore();
  }
  return { items, loading, hasMore, error, liveError, scrollerRef, loadMore, onScroll, keyOf };
}
