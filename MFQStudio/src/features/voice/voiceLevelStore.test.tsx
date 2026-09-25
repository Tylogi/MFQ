/** 验证语音音量独立订阅与控制器切换时的复位行为。 */
import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { RealtimeCallbacks } from './realtimeTypes';
import { useVoiceConversation } from './useVoiceConversation';
import { getVoiceLevel, resetVoiceLevel, setVoiceLevel, useVoiceLevel } from './voiceLevelStore';

const mockControllers = vi.hoisted(
  () =>
    [] as Array<{
      callbacks: RealtimeCallbacks;
      stop: () => Promise<void>;
    }>,
);

vi.mock('../../realtimeAudio', () => ({
  RealtimeAudioController: class {
    stop: () => Promise<void>;

    constructor(callbacks: RealtimeCallbacks) {
      this.stop = vi.fn(async () => {
        await Promise.resolve();
        callbacks.onLevel(0);
      });
      mockControllers.push({ callbacks, stop: this.stop });
    }

    setPlayback() {}
  },
  saveVoiceClip: vi.fn(),
}));

vi.mock('../settings/SettingsProvider', () => ({
  useSettings: () => ({ settings: { playbackEnabled: true, fullDuplex: false } }),
}));

describe('语音音量独立订阅', () => {
  beforeEach(() => {
    mockControllers.length = 0;
    resetVoiceLevel();
  });

  afterEach(() => resetVoiceLevel());

  it('仅在音量改变时更新订阅者，不触发会话 hook 重渲染', () => {
    let conversationRenders = 0;
    let levelRenders = 0;
    const conversation = renderHook(() => {
      conversationRenders += 1;
      return useVoiceConversation('session-a', 1, vi.fn());
    });
    const level = renderHook(() => {
      levelRenders += 1;
      return useVoiceLevel();
    });
    const before = conversationRenders;

    act(() => mockControllers[0].callbacks.onLevel(0.6));
    expect(level.result.current).toBe(0.6);
    expect(conversationRenders).toBe(before);
    const afterChange = levelRenders;

    act(() => mockControllers[0].callbacks.onLevel(0.6));
    expect(levelRenders).toBe(afterChange);
    expect(conversationRenders).toBe(before);
    conversation.unmount();
    level.unmount();
  });

  it('会话与连接切换立即复位，并忽略旧控制器的迟到音量', async () => {
    const level = renderHook(useVoiceLevel);
    const conversation = renderHook(
      ({ sessionId, revision }) => useVoiceConversation(sessionId, revision, vi.fn()),
      { initialProps: { sessionId: 'session-a', revision: 1 } },
    );

    act(() => mockControllers[0].callbacks.onLevel(0.8));
    expect(level.result.current).toBe(0.8);

    conversation.rerender({ sessionId: 'session-b', revision: 1 });
    expect(level.result.current).toBe(0);
    expect(mockControllers[0].stop).toHaveBeenCalled();

    act(() => mockControllers[0].callbacks.onLevel(0.4));
    expect(level.result.current).toBe(0);
    act(() => mockControllers[0].callbacks.onState('connecting'));
    act(() => mockControllers[0].callbacks.onLevel(0.4));
    expect(level.result.current).toBe(0.4);

    conversation.rerender({ sessionId: 'session-b', revision: 2 });
    expect(level.result.current).toBe(0);
    expect(mockControllers[0].stop).toHaveBeenCalled();
    expect(mockControllers).toHaveLength(2);

    act(() => mockControllers[0].callbacks.onLevel(0.9));
    expect(level.result.current).toBe(0);
    act(() => mockControllers[1].callbacks.onLevel(0.3));
    expect(level.result.current).toBe(0.3);

    await act(async () => {
      await Promise.resolve();
    });
    expect(level.result.current).toBe(0.3);
    conversation.unmount();
    expect(level.result.current).toBe(0);
    level.unmount();
  });

  it('控制器停止发布零音量，卸载后旧回调无法重新写入', async () => {
    const level = renderHook(useVoiceLevel);
    const conversation = renderHook(() => useVoiceConversation('session-a', 1, vi.fn()));
    act(() => mockControllers[0].callbacks.onLevel(0.7));

    await act(async () => {
      await mockControllers[0].stop();
    });
    expect(level.result.current).toBe(0);

    act(() => mockControllers[0].callbacks.onLevel(0.5));
    expect(getVoiceLevel()).toBe(0.5);
    conversation.unmount();
    expect(level.result.current).toBe(0);
    act(() => mockControllers[0].callbacks.onLevel(0.9));
    expect(level.result.current).toBe(0);
    level.unmount();
  });

  it('未挂载订阅者时仍可更新快照并跳过重复写入', () => {
    setVoiceLevel(0.2);
    expect(getVoiceLevel()).toBe(0.2);
    resetVoiceLevel();
    expect(getVoiceLevel()).toBe(0);
  });
});
