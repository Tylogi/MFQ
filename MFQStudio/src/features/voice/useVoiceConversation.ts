/** 将语音采集、片段持久化与会话归属绑定，供聊天业务独立使用。 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { RealtimeAudioController, type VoiceState, saveVoiceClip } from '../../realtimeAudio';
import {
  VOICE_HISTORY_KEY,
  loadVoiceHistory,
  type VoiceMessage,
  type LiveVoiceOutput,
} from './history';
import { useSettings } from '../settings/SettingsProvider';
import { errorMessage } from '../../app/formatters';
import { resetVoiceLevel, setVoiceLevel } from './voiceLevelStore';

/** 保留跨路由语音状态，按会话切换和连接重置停止旧控制器。 */
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
  useEffect(() => {
    const stable = voiceMessages.filter(
      (message) => !message.pending && (message.text.trim() || message.audioId),
    );
    try {
      localStorage.setItem(VOICE_HISTORY_KEY, JSON.stringify(stable.slice(-200)));
    } catch {
      /* 存储额度不足不影响当前语音会话。 */
    }
  }, [voiceMessages]);
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
          const persist = async () => {
            if (!audio) {
              setVoiceMessages((current) => current.filter((message) => message.id !== id));
              return;
            }
            const audioId = `voice-${id}`;
            await saveVoiceClip(audioId, audio);
            setVoiceMessages((current) =>
              current.map((message) =>
                message.id === id && message.sessionId === sessionId
                  ? { ...message, audioId, pending: false }
                  : message,
              ),
            );
          };
          void persist().catch((cause) => setError(errorMessage(cause)));
        },
        onTurn: ({ id, sessionId, text, audio }) => {
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
              await saveVoiceClip(audioId, audio);
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
    voiceState,
    liveVoice,
  }), [voiceMessages, voiceState, liveVoice]);
}
