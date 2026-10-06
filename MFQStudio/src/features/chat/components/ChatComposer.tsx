/** Manage the input draft, IME submission, attachment entry point, and generate button, isolating high-frequency typing state. */
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
/** Send the input and call accepted to clear the session draft after preparation succeeds. */
  onSend: (text: string, accepted: () => void) => Promise<void>;
/** Cancel the current generation, including backend cancellation and history synchronization. */
  onStop: () => Promise<void>;
/** Forward input-preparation or platform-call errors to the page error area, preserving drafts not yet accepted. */
  onError: (error: unknown) => void;
}
/** Render the chat input area; draft changes update only this component while generation state comes from the external controller. */
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
/** Prevent duplicate submissions; clear the session draft only after the business action accepts the input. */
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
