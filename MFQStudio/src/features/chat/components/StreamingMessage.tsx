/** Subscribe to streaming snapshots separately so token updates do not rerender the shell or saved messages. */
import type { TFunction } from 'i18next';
import { lazy, Suspense, useSyncExternalStore } from 'react';
import type { GenerationController } from '../state/generationController';

const Markdown = lazy(() =>
  import('../markdown/Markdown').then((module) => ({ default: module.Markdown })),
);

interface StreamingMessageProps {
  controller: GenerationController;
  sessionId: string | null;
  t: TFunction;
}
/** Display current-session deltas, preserve the answer on failure, and provide a read-only history recovery entry point. */
export function StreamingMessage({ controller, sessionId, t }: StreamingMessageProps) {
  const snapshot = useSyncExternalStore(controller.subscribe, controller.getSnapshot);
  if (snapshot.sessionId !== sessionId) return null;
  const { live, phase, recoveryNeeded, error } = snapshot;
  const streaming = phase === 'submitting' || phase === 'streaming' || phase === 'stopping';
  const hasContent = Boolean(live?.reasoning || live?.text || live?.tools.length);
  return (
    <>
      {live && (
        <article className="message message-assistant live-message">
          <div className="message-body">
            {live.reasoning && (
              <details className="reasoning" open>
                <summary>
                  {streaming ? t('chat:streamingMessage.thinking') : t('chat:streamingMessage.reasoning')}
                </summary>
                <Suspense fallback={<pre>{live.reasoning}</pre>}>
                  <Markdown text={live.reasoning} live={streaming} normalizeEscapedLineBreaks />
                </Suspense>
              </details>
            )}
            {live.text && (
              <Suspense fallback={<pre>{live.text}</pre>}>
                <Markdown text={live.text} live={streaming} normalizeEscapedLineBreaks />
              </Suspense>
            )}
            {live.tools.map((tool, index) => (
              <pre className="tool-call" key={index}>
                {tool}
              </pre>
            ))}
            {!hasContent && streaming && (
              <span className="thinking" aria-label={t('chat:streamingMessage.waitingForResponse')}>
                <i />
                <i />
                <i />
              </span>
            )}
            {phase === 'syncing' && (
              <p role="status">{t('chat:streamingMessage.synchronizingResponse')}</p>
            )}
            {hasContent && phase === 'cancelled' && (
              <p role="status">
                {t('chat:streamingMessage.stoppedPartialResponseRetained')}
              </p>
            )}
            {hasContent && phase === 'failed' && (
              <p role="status">
                {t('chat:streamingMessage.incompleteResponseReceivedContentRetained')}
              </p>
            )}
          </div>
        </article>
      )}
      {error && (
        <div className="error-banner" role="alert">
          <span>{error}</span>
        </div>
      )}
      {recoveryNeeded && (
        <button
          disabled={phase === 'syncing'}
          onClick={() => void controller.retrySynchronization()}
          type="button"
        >
          {t('chat:streamingMessage.synchronizeResponse')}
        </button>
      )}
    </>
  );
}
