/** 历史消息列表：展示附件、工具调用、推理指标及编辑操作。 */
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
  /** 保存已编辑消息，并从该消息继续生成。 */
  saveEdit: (message: Message) => Promise<void>;
  /** 复制消息正文到剪贴板。 */
  copyMessage: (message: Message) => Promise<void>;
  /** 从选定助手消息回退并重新生成回答。 */
  regenerate: (message: Message) => Promise<void>;
  /** 执行用户确认的 MCP 工具调用。 */
  executeToolCalls: (message: Message) => Promise<void>;
}
interface SavedMessageListProps {
  messages: Message[];
  responses: Record<string, ResponseResource>;
  mcpTools: McpToolResource[];
  busy: boolean;
  tr: (zh: string, en: string) => string;
  editDraft: EditDraft | null;
  setEditDraft: Dispatch<SetStateAction<EditDraft | null>>;
  actions: MessageActions;
}
type MessageRowsProps = Omit<SavedMessageListProps, 'actions'> & MessageActions;

/** 仅在历史数据、编辑状态或工具能力改变时重新渲染历史正文。 */
const MessageRows = memo(function MessageRows({ messages, responses, mcpTools, busy, tr, editDraft, setEditDraft, saveEdit, copyMessage, regenerate, executeToolCalls }: MessageRowsProps) {
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
                      <time className="message-time" dateTime={message.created_at}>{new Date(message.created_at).toLocaleString()}</time>
                      <div className="message-content">
                        {editing ? <div className="message-editor"><textarea aria-label={tr("消息", "Message")} onChange={(event) => setEditDraft((current) => current && ({ ...current, text: event.target.value }))} value={editDraft.text} /><div><button onClick={() => setEditDraft(null)} type="button">{tr("取消", "Cancel")}</button><button className="primary" onClick={() => void saveEdit(message)} type="button">{tr("保存", "Save")}</button></div></div> : <>{parts.reasoning && <details className="reasoning"><summary>{tr("思考过程", "Reasoning")}</summary>{renderMarkdown(parts.reasoning, false, message.role === "assistant")}</details>}{message.parts.filter(isMediaPart).map((part, index) => <MediaPartView key={`${part.media.id}-${index}`} part={part} />)}{message.parts.filter((part) => part.type === "document").map((part, index) => <DocumentPartView key={`${part.media.id}-${index}`} part={part} />)}{parts.text && renderMarkdown(parts.text, false, message.role === "assistant")}{message.parts.filter((part) => part.type === "tool_call" || part.type === "tool_result").map((part, index) => <div className="tool-call" key={index}><pre>{part.type === "tool_call" ? `${part.name}(${JSON.stringify(part.arguments, null, 2)})` : JSON.stringify(part.result, null, 2)}</pre></div>)}{message.parts.some((part) => part.type === "tool_call" && mcpTools.some((tool) => tool.qualified_name === part.name)) && <div className="tool-confirm"><button disabled={busy} onClick={() => void executeToolCalls(message)} type="button">{tr("确认并执行所有工具", "Confirm and run all tools")}</button></div>}</>}
                      </div>
                      {response?.performance && <details className="response-metrics"><summary><span>{formatNumber(response.performance.decode_tps, 1)} tok/s</span><span>{formatNumber(responsePrefill.tokensPerSecond, 1)} pp</span><span>{formatNumber(responseTtftMs, 1)} ms TTFT</span></summary><div><span>{response.performance.prefill_tokens} prompt tokens</span><span>{response.usage?.completion_tokens ?? 0} output tokens</span>{response.performance.processor_ms > 0 && <span>{tr("媒体准备", "Media preparation")} {formatNumber(response.performance.processor_ms, 1)} ms</span>}{response.performance.multimodal_ms > 0 && <span>{tr("多模态编码", "Multimodal encoding")} {formatNumber(response.performance.multimodal_ms, 1)} ms</span>}{response.performance.model_prefill_ms > response.performance.multimodal_ms && <span>LLM {formatNumber(response.performance.prefill_ms, 1)} ms</span>}<span>{response.finish_reason || "stop"}</span><span>T {formatNumber(response.performance.sampling.temperature, 2)}</span><span>top-p {formatNumber(response.performance.sampling.top_p, 2)}</span><span>repeat {formatNumber(response.performance.sampling.repetition_penalty, 2)}</span>{response.performance.sampling.reasoning_effort && <span>{response.performance.sampling.reasoning_effort}</span>}</div></details>}
                      {!editing && <div className="message-actions"><Tooltip content={tr("复制", "Copy")}><button aria-label={tr("复制", "Copy")} onClick={() => void copyMessage(message)} title={tr("复制", "Copy")} type="button"><Icon name="copy" size={14} /></button></Tooltip>{message.role === "user" && <button aria-label={tr("编辑", "Edit")} onClick={() => setEditDraft({ messageId: message.id, text: parts.text })} title={tr("编辑", "Edit")} type="button"><Icon name="edit" size={14} /></button>}{message.role === "assistant" && <button aria-label={tr("重新生成", "Regenerate")} onClick={() => void regenerate(message)} title={tr("重新生成", "Regenerate")} type="button"><Icon name="refresh" size={14} /></button>}</div>}
                    </div>
                  </article>;
                })}</>;
});

/** 将最新业务回调绑定为稳定事件入口，避免输入草稿及流式更新使历史正文失去缓存。 */
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
