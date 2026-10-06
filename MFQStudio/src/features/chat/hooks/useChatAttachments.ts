/** Manage pending attachments and previews, converting browser files into server message parts. */
import { i18n } from '../../../i18n';
import type { TFunction } from 'i18next';
import { useCallback, useEffect, useRef, useState } from 'react';
import { mediaApi } from '../../../shared/api/resources/media';
import type { ContentPart } from '../../../shared/api/types';
import {
  isTextDocument,
  MAX_DOCUMENT_BYTES,
  MAX_ATTACHMENTS,
  documentMimeType,
  mediaMetadata,
  type PendingAttachment,
} from '../attachments';
/** Isolate attachment selections by session and release preview URLs on session changes or unmount. */
export function useChatAttachments(
  sessionId: string | null,
  connectionRevision: number,
  onError: (message: string | null) => void,
  t: TFunction = i18n.t.bind(i18n),
) {
  const [attachments, setAttachments] = useState<PendingAttachment[]>([]);
  const latest = useRef(attachments);
  latest.current = attachments;
  const translate = useRef(t);
  translate.current = t;
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
/** Validate document size and attachment type, allowing up to eight pending files per session. */
  const selectAttachments = useCallback(
    (files: FileList | null) => {
      if (!files) return;
      onError(null);
      const next: PendingAttachment[] = [];
      let omitted = 0;
      for (const file of Array.from(files)) {
        if (latest.current.length + next.length >= MAX_ATTACHMENTS) {
          omitted += 1;
          continue;
        }
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
          onError(translate.current('chat:useChatAttachments.unsupportedAttachment', { name: file.name }));
          continue;
        }
        if (kind === 'document' && file.size > MAX_DOCUMENT_BYTES) {
          onError(translate.current('chat:useChatAttachments.oversizedDocument', { limit: MAX_DOCUMENT_BYTES / (1024 * 1024), name: file.name }));
          continue;
        }
        next.push({
          id: crypto.randomUUID(),
          file,
          kind,
          previewUrl: kind === 'document' ? '' : URL.createObjectURL(file),
        });
      }
      latest.current = [...latest.current, ...next];
      setAttachments(latest.current);
      if (omitted) onError(translate.current('chat:useChatAttachments.upToAttachmentsPerMessageFilesWereNotAdded', { maxAttachments: MAX_ATTACHMENTS, omitted: omitted }));
    },
    [onError],
  );
/** Remove an attachment and release its object URL. */
  const removeAttachment = useCallback((id: string) => {
    const removed = latest.current.find((item) => item.id === id);
    if (removed?.previewUrl) URL.revokeObjectURL(removed.previewUrl);
    latest.current = latest.current.filter((item) => item.id !== id);
    setAttachments(latest.current);
  }, []);
/** Upload the current attachment snapshot and return typed generation input; retain selections on failure. */
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
