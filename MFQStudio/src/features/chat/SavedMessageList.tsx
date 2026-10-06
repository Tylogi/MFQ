/** Saved-message list displaying attachments, tool calls, inference metrics, and editing actions. */
import { i18n } from '../../i18n';
import type { TFunction } from 'i18next';
import { memo, useLayoutEffect, useMemo, useRef, type Dispatch, type SetStateAction } from 'react';
import type { McpToolResource, Message, ResponseResource } from '../../shared/api/types';
import { Icon } from '../../app/display';
import { formatNumber } from '../../app/formatters';
import { displayPrefillMetric, preferPositiveMetric } from '../runtime/metrics';
import { DocumentPartView, MediaPartView } from './MessageMedia';
import { isMediaPart, textParts } from './messageParts';
import { renderMarkdown } from './MessageMarkdown';
import { Tooltip } from '../../shared/ui/Tooltip';

export interface EditDraft { messageId: string; text: string; }
export interface MessageActions {
/** Save the edited message and continue generation from it. */
  saveEdit: (message: Message) => Promise<void>;
/** Copy the message text to the clipboard. */
  copyMessage: (message: Message) => Promise<void>;
/** Rewind from the selected assistant message and regenerate the response. */
  regenerate: (message: Message) => Promise<void>;
/** Execute an MCP tool call confirmed by the user. */
  executeToolCalls: (message: Message) => Promise<void>;
}
interface SavedMessageListProps {
  messages: Message[];
  responses: Record<string, ResponseResource>;
  mcpTools: McpToolResource[];
  busy: boolean;
  t: TFunction;
  editDraft: EditDraft | null;
  setEditDraft: Dispatch<SetStateAction<EditDraft | null>>;
  actions: MessageActions;
}
type MessageRowsProps = Omit<SavedMessageListProps, 'actions'> & MessageActions;
/** Rerender saved message content only when history, editing state, or tool capabilities change. */
const MessageRows = memo(function MessageRows({ messages, responses, mcpTools, busy, t, editDraft, setEditDraft, saveEdit, copyMessage, regenerate, executeToolCalls }: MessageRowsProps) {
  return <>{messages.map((message) => {
                  const parts = textParts(message);
                  const editing = editDraft?.messageId === message.id;
                  const response = responses[message.id];
                  const responsePrefill = displayPrefillMetric(response?.performance);
                  const responseTtftMs = preferPositiveMetric(
                    response?.performance?.ttft_ms,
                    response?.performance?.complete_prefill_ms,
                  );
                  return <article className={`message message-${message.role}`} key={message.id}>
                    <div className="message-body">
                      <time className="message-time" dateTime={message.created_at}>{new Date(message.created_at).toLocaleString(i18n.resolvedLanguage)}</time>
                      <div className="message-content">
                        {editing ? <div className="message-editor"><textarea aria-label={t('chat:savedMessageList.message')} onChange={(event) => setEditDraft((current) => current && ({ ...current, text: event.target.value }))} value={editDraft.text} /><div><button onClick={() => setEditDraft(null)} type="button">{t('common:cancel')}</button><button className="primary" onClick={() => void saveEdit(message)} type="button">{t('common:save')}</button></div></div> : <>{parts.reasoning && <details className="reasoning"><summary>{t('chat:savedMessageList.reasoning')}</summary>{renderMarkdown(parts.reasoning, false, message.role === "assistant")}</details>}{message.parts.filter(isMediaPart).map((part, index) => <MediaPartView key={`${part.media.id}-${index}`} part={part} />)}{message.parts.filter((part) => part.type === "document").map((part, index) => <DocumentPartView key={`${part.media.id}-${index}`} part={part} />)}{parts.text && renderMarkdown(parts.text, false, message.role === "assistant")}{message.parts.filter((part) => part.type === "tool_call" || part.type === "tool_result").map((part, index) => <div className="tool-call" key={index}><pre>{part.type === "tool_call" ? `${part.name}(${JSON.stringify(part.arguments, null, 2)})` : JSON.stringify(part.result, null, 2)}</pre></div>)}{message.parts.some((part) => part.type === "tool_call" && mcpTools.some((tool) => tool.qualified_name === part.name)) && <div className="tool-confirm"><button disabled={busy} onClick={() => void executeToolCalls(message)} type="button">{t('chat:savedMessageList.confirmAndRunAllTools')}</button></div>}</>}
                      </div>
                      {response?.performance && <details className="response-metrics"><summary><span>{formatNumber(response.performance.decode_tps, 1)} tok/s</span><span>{formatNumber(responsePrefill.tokensPerSecond, 1)} pp</span><span>{formatNumber(responseTtftMs, 1)} ms TTFT</span></summary><div><span>{response.performance.prefill_tokens} prompt tokens</span><span>{response.usage?.completion_tokens ?? 0} output tokens</span>{response.performance.processor_ms > 0 && <span>{t('chat:savedMessageList.mediaPreparation')} {formatNumber(response.performance.processor_ms, 1)} ms</span>}{response.performance.multimodal_ms > 0 && <span>{t('chat:savedMessageList.multimodalEncoding')} {formatNumber(response.performance.multimodal_ms, 1)} ms</span>}{response.performance.model_prefill_ms > response.performance.multimodal_ms && <span>LLM {formatNumber(response.performance.prefill_ms, 1)} ms</span>}<span>{response.finish_reason || "stop"}</span><span>T {formatNumber(response.performance.sampling.temperature, 2)}</span><span>top-p {formatNumber(response.performance.sampling.top_p, 2)}</span><span>repeat {formatNumber(response.performance.sampling.repetition_penalty, 2)}</span>{response.performance.sampling.reasoning_effort && <span>{response.performance.sampling.reasoning_effort}</span>}</div></details>}
                      {!editing && <div className="message-actions"><Tooltip content={t('chat:savedMessageList.copy')}><button aria-label={t('chat:savedMessageList.copy')} onClick={() => void copyMessage(message)} title={t('chat:savedMessageList.copy')} type="button"><Icon name="copy" size={14} /></button></Tooltip>{message.role === "user" && <button aria-label={t('chat:savedMessageList.edit')} onClick={() => setEditDraft({ messageId: message.id, text: parts.text })} title={t('chat:savedMessageList.edit')} type="button"><Icon name="edit" size={14} /></button>}{message.role === "assistant" && <button aria-label={t('chat:savedMessageList.regenerate')} onClick={() => void regenerate(message)} title={t('chat:savedMessageList.regenerate')} type="button"><Icon name="refresh" size={14} /></button>}</div>}
                    </div>
                  </article>;
                })}</>;
});
/** Bind current business callbacks to stable event handlers so draft and streaming updates do not invalidate saved-message memoization. */
export function SavedMessageList({ actions, ...props }: SavedMessageListProps) {
  const latest = useRef(actions);
  useLayoutEffect(() => { latest.current = actions; }, [actions]);
  const handlers = useMemo<MessageActions>(() => ({
    saveEdit: (message) => latest.current.saveEdit(message),
    copyMessage: (message) => latest.current.copyMessage(message),
    regenerate: (message) => latest.current.regenerate(message),
    executeToolCalls: (message) => latest.current.executeToolCalls(message),
  }), []);
  return <MessageRows {...props} {...handlers} />;
}
