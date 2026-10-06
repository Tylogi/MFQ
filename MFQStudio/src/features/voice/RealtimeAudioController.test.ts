/** Verify text submissions remain pending until the realtime session can actually send them. */
import { beforeEach, expect, it, vi } from 'vitest';
import { RealtimeAudioController } from './RealtimeAudioController';
import type { RealtimeCallbacks, RealtimeSessionConfig } from './realtimeTypes';

vi.mock('../../shared/api/client', () => ({ runtimeRealtimeUrl: () => 'ws://localhost/realtime' }));
vi.mock('./AudioDevices', () => ({
  AudioDevices: class {
    capturing = false;
    ensureOutput = async () => {};
    close = async () => {};
    stopCapture = async () => {};
    stopPlayback = () => {};
  },
}));

const sockets: FakeWebSocket[] = [];

class FakeWebSocket {
  static OPEN = 1;
  readyState = 0;
  onmessage: ((event: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  send = vi.fn();

  constructor(_url: string) {
    sockets.push(this);
  }

  close() {
    this.readyState = 3;
    this.onclose?.();
  }

  created() {
    this.readyState = FakeWebSocket.OPEN;
    this.onmessage?.({ data: JSON.stringify({ type: 'session.created' }) });
  }
}

const config: RealtimeSessionConfig = {
  sessionId: 'chat-1', systemPrompt: '', temperature: 0.7, topP: 0.9,
  topK: 40, repetitionPenalty: 1,
};
const callbacks: RealtimeCallbacks = {
  onState: vi.fn(), onLevel: vi.fn(), onText: vi.fn(), onError: vi.fn(),
  onInputStart: vi.fn(), onInputEnd: vi.fn(), onTurn: vi.fn(),
};

beforeEach(() => {
  sockets.length = 0;
  vi.stubGlobal('WebSocket', FakeWebSocket);
  vi.clearAllMocks();
});

it('resolves text submission only after session creation and a successful socket send', async () => {
  const controller = new RealtimeAudioController(callbacks, false, false);
  const submitted = controller.submitText(' hello ', config);
  let resolved = false;
  void submitted.then(() => { resolved = true; });
  await vi.waitFor(() => expect(sockets).toHaveLength(1));
  expect(resolved).toBe(false);
  expect(sockets[0].send).not.toHaveBeenCalled();

  sockets[0].created();
  await submitted;
  expect(sockets[0].send).toHaveBeenCalledWith(JSON.stringify({
    type: 'input.append', input: { text: 'hello', max_new_speak_tokens: 20 },
  }));
  await controller.stop();
});

it('rejects text submission when the socket fails before session creation', async () => {
  const controller = new RealtimeAudioController(callbacks, false, false);
  const submitted = controller.submitText('hello', config);
  await vi.waitFor(() => expect(sockets).toHaveLength(1));
  sockets[0].onerror?.();
  await expect(submitted).rejects.toThrow('Voice connection failed');
  expect(sockets[0].send).not.toHaveBeenCalled();
});

it('rejects text submission when the socket closes before session creation', async () => {
  const controller = new RealtimeAudioController(callbacks, false, false);
  const submitted = controller.submitText('hello', config);
  await vi.waitFor(() => expect(sockets).toHaveLength(1));
  sockets[0].close();
  await expect(submitted).rejects.toThrow('Voice connection closed');
  expect(sockets[0].send).not.toHaveBeenCalled();
});

it.each([
  { input_sample_rate: 48000 },
  { output_sample_rate: 44100 },
])('rejects incompatible advertised audio rates before opening transport: %j', async (rates) => {
  const controller = new RealtimeAudioController(callbacks, false, false);
  await expect(controller.submitText('hello', {
    ...config, capabilities: { available: true, ...rates },
  })).rejects.toThrow('Unsupported voice');
  expect(sockets).toHaveLength(0);
});
