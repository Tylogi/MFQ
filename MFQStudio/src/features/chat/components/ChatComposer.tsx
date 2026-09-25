/** 管理输入草稿、输入法提交、附件入口与生成按钮，隔离高频键入状态。 */
import { useRef, type FormEvent, type ReactNode } from 'react';
import { Icon } from '../../../app/display';
import { formatNumber } from '../../../app/formatters';
import { VideoWithFirstFrame } from '../MessageMedia';
import { useChatAttachmentActions, useChatAttachmentList } from '../ChatAttachmentsProvider';
import { isGenerationBusy, type GenerationPhase } from '../state/generationController';
import { useDraftStore } from '../state/draftStore';

interface ChatComposerProps {
  sessionId: string;
  ready: boolean;
  busy: boolean;
  recoveryNeeded: boolean;
  phase: GenerationPhase;
  placeholder: string;
  attachmentAccept: string;
  toolbar: ReactNode;
  tr: (zh: string, en: string) => string;
  /** 执行发送；准备成功后调用 accepted 清理该会话的草稿。 */
  onSend: (text: string, accepted: () => void) => Promise<void>;
  /** 取消当前生成，包含后端取消与历史同步。 */
  onStop: () => Promise<void>;
  /** 将准备输入或平台调用异常交给页面错误区域展示，保留尚未接受的草稿。 */
  onError: (error: unknown) => void;
}

/** 渲染聊天输入区域；草稿变更只更新当前输入组件，生成状态由外部控制器提供。 */
export function ChatComposer({
  sessionId,
  ready,
  busy,
  recoveryNeeded,
  phase,
  placeholder,
  attachmentAccept,
  toolbar,
  tr,
  onSend,
  onStop,
  onError,
}: ChatComposerProps) {
  const attachments = useChatAttachmentList();
  const { selectAttachments, removeAttachment } = useChatAttachmentActions();
  const draft = useDraftStore((state) => state.drafts[sessionId] ?? '');
  const setDraft = useDraftStore((state) => state.setDraft);
  const fileInput = useRef<HTMLInputElement | null>(null);
  const submitting = useRef(false);
  const generating = isGenerationBusy(phase);
  const stopping = phase === 'stopping' || phase === 'syncing';

  /** 防止连续提交；只有业务接受当前输入后才清除对应会话草稿。 */
  async function submit(event: FormEvent) {
    event.preventDefault();
    if (
      submitting.current ||
      !ready ||
      busy ||
      recoveryNeeded ||
      (!draft.trim() && !attachments.length)
    )
      return;
    submitting.current = true;
    try {
      await onSend(draft.trim(), () => setDraft(sessionId, ''));
    } catch (error) {
      onError(error);
    } finally {
      submitting.current = false;
    }
  }

  return (
    <form className="composer" onSubmit={(event) => void submit(event)}>
      {attachments.length > 0 && (
        <div className="attachment-tray">
          {attachments.map((attachment) => (
            <div className="attachment-chip" key={attachment.id}>
              {attachment.kind === 'image' ? (
                <img alt="" src={attachment.previewUrl} />
              ) : attachment.kind === 'video' ? (
                <VideoWithFirstFrame muted src={attachment.previewUrl} />
              ) : (
                <span>{attachment.kind === 'document' ? 'TXT' : '♫'}</span>
              )}
              <div>
                <strong>{attachment.file.name}</strong>
                <small>
                  {attachment.kind} · {formatNumber(attachment.file.size)} B
                </small>
              </div>
              <button
                aria-label={tr('移除附件', 'Remove attachment')}
                disabled={busy}
                onClick={() => removeAttachment(attachment.id)}
                type="button"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}
      <textarea
        aria-label={tr('消息', 'Message')}
        disabled={!ready || busy}
        maxLength={32768}
        onChange={(event) => setDraft(sessionId, event.target.value)}
        onKeyDown={(event) => {
          if (
            event.key === 'Enter' &&
            !event.shiftKey &&
            !event.nativeEvent.isComposing &&
            event.nativeEvent.keyCode !== 229
          ) {
            event.preventDefault();
            event.currentTarget.form?.requestSubmit();
          }
        }}
        placeholder={placeholder}
        rows={1}
        value={draft}
      />
      <div className="composer-toolbar">
        <input
          accept={attachmentAccept}
          hidden
          multiple
          onChange={(event) => {
            selectAttachments(event.target.files);
            event.target.value = '';
          }}
          ref={fileInput}
          type="file"
        />
        <button
          aria-label={tr('添加附件', 'Add attachment')}
          disabled={!ready || busy}
          onClick={() => fileInput.current?.click()}
          title={tr('添加文档或媒体', 'Add document or media')}
          type="button"
        >
          <Icon name="paperclip" />
        </button>
        {toolbar}
        {generating ? (
          <button
            aria-label={
              phase === 'syncing'
                ? tr('正在同步回答', 'Synchronizing response')
                : stopping
                  ? tr('正在停止生成', 'Stopping generation')
                  : tr('停止生成', 'Stop generation')
            }
            className="send-button stop"
            disabled={stopping}
            onClick={() => void onStop()}
            type="button"
          >
            <Icon name="stop" size={14} />
          </button>
        ) : (
          <button
            aria-label={tr('发送', 'Send')}
            className="send-button"
            disabled={busy || recoveryNeeded || !ready || (!draft.trim() && !attachments.length)}
            type="submit"
          >
            <Icon name="send" size={15} />
          </button>
        )}
      </div>
    </form>
  );
}
