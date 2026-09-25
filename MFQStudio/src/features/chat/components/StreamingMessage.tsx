/** 单独订阅流式快照，避免 token 更新触发应用外壳和历史消息重渲染。 */
import { lazy, Suspense, useSyncExternalStore } from 'react';
import type { GenerationController } from '../state/generationController';

const Markdown = lazy(() =>
  import('../markdown/Markdown').then((module) => ({ default: module.Markdown })),
);

interface StreamingMessageProps {
  controller: GenerationController;
  sessionId: string | null;
  tr: (zh: string, en: string) => string;
}

/** 显示当前会话增量、失败时保留的回答以及只读历史恢复入口。 */
export function StreamingMessage({ controller, sessionId, tr }: StreamingMessageProps) {
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
                  {streaming ? tr('正在思考', 'Thinking') : tr('思考过程', 'Reasoning')}
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
              <span className="thinking" aria-label={tr('等待回答', 'Waiting for response')}>
                <i />
                <i />
                <i />
              </span>
            )}
            {phase === 'syncing' && (
              <p role="status">{tr('正在同步回答…', 'Synchronizing response…')}</p>
            )}
            {hasContent && phase === 'cancelled' && (
              <p role="status">
                {tr('已停止，保留部分回答', 'Stopped; partial response retained')}
              </p>
            )}
            {hasContent && phase === 'failed' && (
              <p role="status">
                {tr(
                  '回答未完成，已保留收到的内容',
                  'Incomplete response; received content retained',
                )}
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
          {tr('重新同步回答', 'Synchronize response')}
        </button>
      )}
    </>
  );
}
