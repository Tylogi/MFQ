/** Verify deleted and evicted voice messages release their local audio clips. */
import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { useVoiceConversation } from './useVoiceConversation';
import { VOICE_HISTORY_KEY, type VoiceMessage } from './history';
import { deleteVoiceClip, pruneVoiceClips, saveVoiceClip } from '../../realtimeAudio';
import type { RealtimeCallbacks } from './realtimeTypes';

let controllerCallbacks: RealtimeCallbacks | null = null;
vi.mock('../../realtimeAudio', () => ({
  deleteVoiceClip: vi.fn().mockResolvedValue(undefined),
  pruneVoiceClips: vi.fn().mockResolvedValue(undefined),
  saveVoiceClip: vi.fn().mockResolvedValue(undefined),
  RealtimeAudioController: class {
    constructor(callbacks: RealtimeCallbacks) { controllerCallbacks = callbacks; }
    setPlayback() {}
    stop = async () => {};
  },
}));
vi.mock('../settings/SettingsProvider', () => ({
  useSettings: () => ({ settings: { playbackEnabled: false, fullDuplex: false } }),
}));

/** Build a persisted voice message with a matching local clip ID. */
function message(id: string, sessionId: string): VoiceMessage {
  return { id, sessionId, role: 'assistant', text: id, audioId: `voice-${id}`, created_at: '' };
}

beforeEach(() => {
  localStorage.clear();
  controllerCallbacks = null;
  vi.mocked(deleteVoiceClip).mockReset().mockResolvedValue(undefined);
  vi.mocked(pruneVoiceClips).mockReset().mockResolvedValue(undefined);
  vi.mocked(saveVoiceClip).mockReset().mockResolvedValue(undefined);
});

it('deletes the clips of a removed session while keeping other sessions', async () => {
  localStorage.setItem(VOICE_HISTORY_KEY, JSON.stringify([message('a', 'session-a'), message('b', 'session-b')]));
  const { result } = renderHook(() => useVoiceConversation(null, 0, vi.fn()));
  await act(async () => result.current.removeSessionVoiceHistory('session-a'));
  expect(result.current.voiceMessages.map((item) => item.id)).toEqual(['b']);
  expect(deleteVoiceClip).toHaveBeenCalledWith('voice-a');
  expect(deleteVoiceClip).not.toHaveBeenCalledWith('voice-b');
});

it('deletes clips evicted by the 200-message history limit', async () => {
  const history = Array.from({ length: 201 }, (_, index) => message(String(index), 'session-a'));
  const { result } = renderHook(() => useVoiceConversation(null, 0, vi.fn()));
  act(() => result.current.setVoiceMessages(history));
  await waitFor(() => expect(deleteVoiceClip).toHaveBeenCalledWith('voice-0'));
  expect(JSON.parse(localStorage.getItem(VOICE_HISTORY_KEY) ?? '[]')).toHaveLength(200);
});

it('prunes clips already orphaned before the app opens', async () => {
  const history = Array.from({ length: 201 }, (_, index) => message(String(index), 'session-a'));
  localStorage.setItem(VOICE_HISTORY_KEY, JSON.stringify(history));
  renderHook(() => useVoiceConversation(null, 0, vi.fn()));
  await waitFor(() => expect(pruneVoiceClips).toHaveBeenCalled());
  expect(pruneVoiceClips).toHaveBeenCalledWith(new Set(history.slice(-200).map((item) => item.audioId)));
});

it('waits for a pending clip write before deleting a removed session clip', async () => {
  let finishWrite!: () => void;
  vi.mocked(saveVoiceClip).mockImplementation(() => new Promise<void>((resolve) => { finishWrite = resolve; }));
  const { result } = renderHook(() => useVoiceConversation(null, 0, vi.fn()));
  act(() => controllerCallbacks?.onTurn({
    id: 'pending', sessionId: 'session-a', text: 'answer', audio: new Blob(['audio']),
  }));
  await waitFor(() => expect(saveVoiceClip).toHaveBeenCalledWith('voice-pending', expect.any(Blob)));
  let cleanup!: Promise<void>;
  act(() => { cleanup = result.current.removeSessionVoiceHistory('session-a'); });
  expect(deleteVoiceClip).not.toHaveBeenCalled();
  await act(async () => {
    finishWrite();
    await cleanup;
  });
  expect(deleteVoiceClip).toHaveBeenCalledWith('voice-pending');
  expect(result.current.voiceMessages).toEqual([]);
});
