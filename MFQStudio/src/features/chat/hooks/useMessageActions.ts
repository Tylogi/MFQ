/** Manage saved-message editing, regeneration, and confirmed tool execution without owning global runtime state. */
import { useMemo, useRef, useState } from 'react';
import type { EditDraft } from '../SavedMessageList';
import { sessionsApi } from '../../../shared/api/resources/sessions';
import { connectionsApi } from '../../../shared/api/resources/connections';
import type { ContentPart, Message, Session } from '../../../shared/api/types';
import { textParts, isMediaPart } from '../messageParts';
import type { useConversationSessions } from './useConversationSessions';
import { errorMessage } from '../../../app/formatters';

interface MessageActionOptions {
  conversation: ReturnType<typeof useConversationSessions>;
  blocked: boolean;
/** Continue generation using the rewound session revision; tool input must not append an optimistic user message. */
  generate: (
    session: Session,
    input: ContentPart[],
    optimistic?: boolean,
    role?: 'user' | 'tool',
  ) => Promise<void>;
/** Lock chat actions while writing history and release the lock when the operation ends. */
  setBusy: (busy: boolean) => void;
}
/** Provide explicit business actions to the message list and reject stale writes using the active session ID. */
export function useMessageActions({
  conversation,
  blocked,
  generate,
  setBusy,
}: MessageActionOptions) {
  const latest = useRef({ conversation, blocked, generate, setBusy });
  latest.current = { conversation, blocked, generate, setBusy };

  return useMemo(() => {
    const current = (sessionId: string) =>
      latest.current.conversation.activeIdRef.current === sessionId;
/** Rewind history after editing a user message, preserving attachments and continuing generation. */
    async function saveEdit(
      message: Message,
      draftText: string,
      onCommitted: () => void,
    ): Promise<boolean> {
      const { conversation, blocked, generate, setBusy } = latest.current;
      const { active, messages, setMessages, setSessions, setResponses, setError } = conversation;
      if (!active || blocked) return false;
      const index = messages.findIndex((item) => item.id === message.id);
      const text = draftText.trim();
      if (index < 0 || (!text && !message.parts.some(isMediaPart))) return false;
      setBusy(true);
      try {
        const rewound = await sessionsApi.rewindSession(active.id, active.revision, message.id, false);
        if (!current(active.id)) return false;
        const parts: ContentPart[] = [
          ...(text ? [{ type: 'text' as const, text }] : []),
          ...message.parts.filter((part) => isMediaPart(part) || part.type === 'document'),
        ];
        const previous = messages.slice(0, index);
        setSessions((items) =>
          items.map((session) => (session.id === rewound.id ? rewound : session)),
        );
        setMessages([...previous, { ...message, parts }]);
        setResponses((items) =>
          Object.fromEntries(
            Object.entries(items).filter(([id]) => previous.some((item) => item.id === id)),
          ),
        );
        onCommitted();
        await generate(rewound, parts, false);
        return true;
      } catch (cause) {
        if (current(active.id)) setError(errorMessage(cause));
        return false;
      } finally {
        setBusy(false);
      }
    }
/** Regenerate from the user input before an assistant answer, removing later history and using the server revision. */
    async function regenerate(message: Message) {
      const { conversation, blocked, generate, setBusy } = latest.current;
      const { active, messages, setMessages, setSessions, setResponses, setError } = conversation;
      if (!active || blocked || message.role !== 'assistant') return;
      const index = messages.findIndex((item) => item.id === message.id);
      const user = messages
        .slice(0, index)
        .reverse()
        .find((item) => item.role === 'user');
      if (!user?.parts.length) return;
      setBusy(true);
      try {
        const rewound = await sessionsApi.rewindSession(active.id, active.revision, user.id, false);
        if (!current(active.id)) return;
        const previous = messages.slice(0, messages.indexOf(user) + 1);
        setSessions((items) =>
          items.map((session) => (session.id === rewound.id ? rewound : session)),
        );
        setMessages(previous);
        setResponses((items) =>
          Object.fromEntries(
            Object.entries(items).filter(([id]) => previous.some((item) => item.id === id)),
          ),
        );
        await generate(rewound, user.parts, false);
      } catch (cause) {
        if (current(active.id)) setError(errorMessage(cause));
      } finally {
        setBusy(false);
      }
    }
/** Copy message text and reasoning; keep the page mounted and show a readable error on failure. */
    async function copyMessage(message: Message) {
      const { setError } = latest.current.conversation;
      try {
        const parts = textParts(message);
        await navigator.clipboard.writeText(
          [parts.reasoning, parts.text].filter(Boolean).join('\n\n'),
        );
      } catch (cause) {
        setError(errorMessage(cause));
      }
    }
/** Execute a tool only after user confirmation, append its result at the session revision, and continue generation. */
    async function executeToolCalls(message: Message) {
      const { conversation, blocked, generate, setBusy } = latest.current;
      const { active, setError } = conversation;
      if (!active || blocked) return;
      const calls = message.parts.filter(
        (part): part is Extract<ContentPart, { type: 'tool_call' }> => part.type === 'tool_call',
      );
      if (!calls.length) return;
      setBusy(true);
      try {
        const results = await Promise.all(
          calls.map(async (call) => ({
            call,
            result: await connectionsApi.callMcpTool(call.name, call.arguments),
          })),
        );
        if (!current(active.id)) return;
        let updated = await sessionsApi.getSession(active.id);
        const parts = results.map(({ call, result }): ContentPart => ({
          type: 'tool_result',
          call_id: call.call_id,
          result: result.structured_content ?? result.content,
          is_error: result.is_error,
        }));
        for (const part of parts.slice(0, -1)) {
          updated = (await sessionsApi.appendMessage(updated.id, updated.revision, 'tool', [part])).session;
          if (!current(active.id)) return;
        }
        await generate(updated, [parts.at(-1)!], false, 'tool');
      } catch (cause) {
        if (current(active.id)) setError(errorMessage(cause));
      } finally {
        setBusy(false);
      }
    }
    return { saveEdit, regenerate, copyMessage, executeToolCalls };
  }, []);
}
