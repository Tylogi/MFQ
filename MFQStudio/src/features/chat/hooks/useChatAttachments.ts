/** 管理待发送附件和预览资源，并把浏览器文件转换为服务端消息片段。 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { mediaApi } from '../../../shared/api/resources/media';
import type { ContentPart } from '../../../shared/api/types';
import {
  isTextDocument,
  MAX_DOCUMENT_BYTES,
  documentMimeType,
  mediaMetadata,
  type PendingAttachment,
} from '../attachments';

/** 按会话隔离附件选择；切换或卸载时释放预览 URL。 */
export function useChatAttachments(
  sessionId: string | null,
  connectionRevision: number,
  onError: (message: string) => void,
) {
  const [attachments, setAttachments] = useState<PendingAttachment[]>([]);
  const latest = useRef(attachments);
  latest.current = attachments;
  const clearAttachments = useCallback(() => {
    latest.current.forEach((item) => {
      if (item.previewUrl) URL.revokeObjectURL(item.previewUrl);
    });
    latest.current = [];
    setAttachments([]);
  }, []);
  useEffect(() => {
    clearAttachments();
    return clearAttachments;
  }, [sessionId, connectionRevision, clearAttachments]);

  /** 校验文档体积与附件类型，每个会话最多保留八个待发送文件。 */
  const selectAttachments = useCallback(
    (files: FileList | null) => {
      if (!files) return;
      const next: PendingAttachment[] = [];
      for (const file of Array.from(files).slice(0, Math.max(0, 8 - latest.current.length))) {
        const kind = file.type.startsWith('image/')
          ? 'image'
          : file.type.startsWith('video/')
            ? 'video'
            : file.type.startsWith('audio/')
              ? 'audio'
              : isTextDocument(file)
                ? 'document'
                : null;
        if (!kind) {
          onError(`Unsupported attachment: ${file.name}`);
          continue;
        }
        if (kind === 'document' && file.size > MAX_DOCUMENT_BYTES) {
          onError(`Document exceeds 64 MiB: ${file.name}`);
          continue;
        }
        next.push({
          id: crypto.randomUUID(),
          file,
          kind,
          previewUrl: kind === 'document' ? '' : URL.createObjectURL(file),
        });
      }
      setAttachments((current) => [...current, ...next]);
    },
    [onError],
  );

  /** 移除一个附件并释放对应的对象 URL。 */
  const removeAttachment = useCallback((id: string) => {
    const removed = latest.current.find((item) => item.id === id);
    if (removed?.previewUrl) URL.revokeObjectURL(removed.previewUrl);
    setAttachments((current) => current.filter((item) => item.id !== id));
  }, []);

  /** 上传当前附件快照，成功后返回可用于生成的类型化输入；失败时保留选择。 */
  const uploadAttachments = useCallback(
    () =>
      Promise.all(
        latest.current.map(async (attachment): Promise<ContentPart> => {
          if (attachment.kind === 'document') {
            const uploaded = await mediaApi.uploadMedia(
              attachment.file,
              documentMimeType(attachment.file),
            );
            const document = await mediaApi.createDocument(uploaded.media.id, attachment.file.name);
            return { type: 'document', media: document.media, name: document.name };
          }
          const metadata = await mediaMetadata(attachment.file, attachment.kind);
          const resource = await mediaApi.uploadMedia(attachment.file);
          return { type: attachment.kind, media: resource.media, ...metadata } as ContentPart;
        }),
      ),
    [],
  );
  const getAttachments = useCallback(() => latest.current, []);
  return {
    attachments,
    getAttachments,
    selectAttachments,
    removeAttachment,
    clearAttachments,
    uploadAttachments,
  };
}
