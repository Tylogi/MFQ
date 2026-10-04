/** Verify defaults when reading chat mode across domains and when switching active sessions. */
import { act, renderHook } from '@testing-library/react';
import { beforeEach, expect, it } from 'vitest';
import type { Session } from '../../../shared/api/types';
import { useConversationStore } from '../state/conversationStore';
import { useActiveSessionMode } from './useActiveSessionMode';

beforeEach(() => useConversationStore.getState().reset());

it('verifies useActiveSessionMode test behavior 1', () => {
  const { result } = renderHook(useActiveSessionMode);
  expect(result.current).toBe('text');

  const voice = { id: 'voice', mode: 'voice' } as Session;
  const duplex = { id: 'duplex', mode: 'full_duplex' } as Session;
  act(() => {
    const store = useConversationStore.getState();
    store.loadSessions(store.epoch, [voice, duplex]);
  });
  expect(result.current).toBe('voice');

  act(() => useConversationStore.getState().setActiveId('duplex'));
  expect(result.current).toBe('full_duplex');

  act(() => useConversationStore.getState().reset());
  expect(result.current).toBe('text');
});
