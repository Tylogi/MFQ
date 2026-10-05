/** Bind voice capture, clip persistence, and session ownership for independent use by chat business logic. */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { RealtimeAudioController, type VoiceState, deleteVoiceClip, pruneVoiceClips, saveVoiceClip } from '../../realtimeAudio';
import {
  VOICE_HISTORY_KEY,
  loadVoiceHistory,
  type VoiceMessage,
  type LiveVoiceOutput,
} from './history';
import { useSettings } from '../settings/SettingsProvider';
import { errorMessage } from '../../app/formatters';
import { resetVoiceLevel, setVoiceLevel } from './voiceLevelStore';

/** Preserve voice state across routes and stop the old controller when the session changes or the connection resets. */
export function useVoiceConversation(
  activeId: string | null,
  connectionRevision: number,
  setError: (message: string) => void,
) {
  const { settings } = useSettings();
  const [voiceMessages, setVoiceMessages] = useState<VoiceMessage[]>(loadVoiceHistory);
  const [voiceState, setVoiceState] = useState<VoiceState>('idle');
  const [liveVoice, setLiveVoice] = useState<LiveVoiceOutput | null>(null);
  const voiceRef = useRef<RealtimeAudioController | null>(null);
  const acceptVoiceLevel = useRef(false);
  const voiceClipWrites = useRef(new Map<string, Promise<void>>());
  const initialClipPruning = useRef<Promise<void> | null>(null);
  const discardedSessions = useRef(new Set<string>());
  const discardedMessages = useRef(new Set<string>());

  /** Wait for in-flight writes before removing audio no longer referenced by local history. */
  const removeClips = useCallback(async (messages: VoiceMessage[]) => {
    const ids = [...new Set(messages.map((message) => message.audioId ?? `voice-${message.id}`))];
    await Promise.all(ids.map(async (id) => {
      await voiceClipWrites.current.get(id)?.catch(() => undefined);
      await deleteVoiceClip(id);
    }));
  }, []);

  /** Remove a deleted session's local messages and clips, including writes still in progress. */
  const removeSessionVoiceHistory = useCallback(async (sessionId: string) => {
    discardedSessions.current.add(sessionId);
    const removed = voiceMessages.filter((message) => message.sessionId === sessionId);
    setVoiceMessages((current) => current.filter((message) => message.sessionId !== sessionId));
    await removeClips(removed);
  }, [voiceMessages, removeClips]);

  useEffect(() => {
    const stable = voiceMessages.filter(
      (message) => !message.pending && (message.text.trim() || message.audioId),
    );
    try {
      localStorage.setItem(VOICE_HISTORY_KEY, JSON.stringify(stable.slice(-200)));
    } catch {
      /* Insufficient storage quota does not affect the current voice session. */
    }
    if (!initialClipPruning.current) {
      const retainedIds = new Set(stable.slice(-200)
        .map((message) => message.audioId)
        .filter((id): id is string => Boolean(id)));
      initialClipPruning.current = pruneVoiceClips(retainedIds)
        .catch((cause) => setError(errorMessage(cause)));
    }
    if (stable.length > 200) {
      const evicted = stable.slice(0, -200);
      const evictedIds = new Set(evicted.map((message) => message.id));
      for (const id of evictedIds) discardedMessages.current.add(id);
      setVoiceMessages((current) => current.filter((message) => !evictedIds.has(message.id)));
      void removeClips(evicted).catch((cause) => setError(errorMessage(cause)));
    }
  }, [voiceMessages, removeClips, setError]);
  useEffect(() => {
    voiceRef.current?.setPlayback(settings.playbackEnabled);
  }, [settings.playbackEnabled]);
  useEffect(() => {
    acceptVoiceLevel.current = false;
    resetVoiceLevel();
    void voiceRef.current?.stop();
    setLiveVoice(null);
  }, [activeId, connectionRevision]);
  useEffect(() => {
    const controller = new RealtimeAudioController(
      {
        onState: (state) => {
          if (voiceRef.current === controller && (state === 'connecting' || state === 'listening')) {
            acceptVoiceLevel.current = true;
          }
          setVoiceState(state);
        },
        onLevel: (level) => {
          if (voiceRef.current === controller && acceptVoiceLevel.current) setVoiceLevel(level);
        },
        onText: (sessionId, text) =>
          setLiveVoice((current) =>
            text ? { sessionId, text } : current?.sessionId === sessionId ? null : current,
          ),
        onError: (message) => setError(message),
        onInputStart: ({ id, sessionId }) => {
          if (discardedSessions.current.has(sessionId)) return;
          setVoiceMessages((current) => [
            ...current,
            {
              id,
              sessionId,
              role: 'user',
              text: '',
              pending: true,
              created_at: new Date().toISOString(),
            },
          ]);
        },
        onInputEnd: ({ id, sessionId, audio }) => {
          if (discardedSessions.current.has(sessionId) || discardedMessages.current.has(id)) return;
          const persist = async () => {
            if (!audio) {
              setVoiceMessages((current) => current.filter((message) => message.id !== id));
              return;
            }
            const audioId = `voice-${id}`;
            await initialClipPruning.current;
            await saveVoiceClip(audioId, audio);
            if (discardedSessions.current.has(sessionId) || discardedMessages.current.has(id)) {
              await deleteVoiceClip(audioId);
              return;
            }
            setVoiceMessages((current) =>
              current.map((message) =>
                message.id === id && message.sessionId === sessionId
                  ? { ...message, audioId, pending: false }
                  : message,
              ),
            );
          };
          const audioId = `voice-${id}`;
          const write = persist();
          voiceClipWrites.current.set(audioId, write);
          void write.catch((cause) => setError(errorMessage(cause))).finally(() => {
            if (voiceClipWrites.current.get(audioId) === write) voiceClipWrites.current.delete(audioId);
          });
        },
        onTurn: ({ id, sessionId, text, audio }) => {
          if (discardedSessions.current.has(sessionId) || discardedMessages.current.has(id)) return;
          setVoiceMessages((current) => {
            const existing = current.find((message) => message.id === id);
            if (existing) {
              return current.map((message) => (message.id === id ? { ...message, text } : message));
            }
            return [
              ...current,
              {
                id,
                sessionId,
                role: 'assistant',
                text,
                created_at: new Date().toISOString(),
              },
            ];
          });
          if (!audio) return;
          const audioId = `voice-${id}`;
          const previous = voiceClipWrites.current.get(audioId) ?? Promise.resolve();
          const persist = previous
            .catch(() => undefined)
            .then(async () => {
              await initialClipPruning.current;
              await saveVoiceClip(audioId, audio);
              if (discardedSessions.current.has(sessionId) || discardedMessages.current.has(id)) {
                await deleteVoiceClip(audioId);
                return;
              }
              setVoiceMessages((current) =>
                current.map((message) => (message.id === id ? { ...message, audioId } : message)),
              );
            });
          voiceClipWrites.current.set(audioId, persist);
          void persist
            .catch((cause) => setError(errorMessage(cause)))
            .finally(() => {
              if (voiceClipWrites.current.get(audioId) === persist) {
                voiceClipWrites.current.delete(audioId);
              }
            });
        },
      },
      settings.playbackEnabled,
      settings.fullDuplex,
    );
    voiceRef.current = controller;
    acceptVoiceLevel.current = true;
    return () => {
      acceptVoiceLevel.current = false;
      resetVoiceLevel();
      if (voiceRef.current === controller) voiceRef.current = null;
      void controller.stop();
    };
    // The controller reads current request settings when capture starts.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connectionRevision]);
  return useMemo(() => ({
    voiceRef,
    voiceMessages,
    setVoiceMessages,
    removeSessionVoiceHistory,
    voiceState,
    liveVoice,
  }), [voiceMessages, removeSessionVoiceHistory, voiceState, liveVoice]);
}
