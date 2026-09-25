/** 验证音频设备资源独立后的播放调度与录音失败清理。 */
import { beforeEach, expect, it, vi } from 'vitest';
import { AudioDevices } from './AudioDevices';

const contexts: FakeContext[] = [];
const sources: Array<{
  start: ReturnType<typeof vi.fn>;
  stop: ReturnType<typeof vi.fn>;
  disconnect: ReturnType<typeof vi.fn>;
  buffer: unknown;
  onended: (() => void) | null;
  connect: ReturnType<typeof vi.fn>;
}> = [];
const stopTrack = vi.fn();
let failWorklet = false;

class FakeContext {
  sampleRate: number;
  currentTime = 1;
  destination = {};
  resume = vi.fn(async () => undefined);
  close = vi.fn(async () => undefined);
  audioWorklet = {
    addModule: vi.fn(async () => {
      if (failWorklet) throw new Error('worklet unavailable');
    }),
  };
  constructor(options?: { sampleRate?: number }) {
    this.sampleRate = options?.sampleRate ?? 48_000;
    contexts.push(this);
  }
  createBuffer(_channels: number, length: number, rate: number) {
    return { duration: length / rate, copyToChannel: vi.fn() };
  }
  createBufferSource() {
    const source = {
      start: vi.fn(),
      stop: vi.fn(),
      disconnect: vi.fn(),
      buffer: null as unknown,
      onended: null as (() => void) | null,
      connect: vi.fn(),
    };
    sources.push(source);
    return source;
  }
  createMediaStreamSource() {
    return { connect: (target: unknown) => target, disconnect: vi.fn() };
  }
  createGain() {
    return { gain: { value: 1 }, connect: vi.fn() };
  }
}

beforeEach(() => {
  contexts.length = 0;
  sources.length = 0;
  failWorklet = false;
  stopTrack.mockClear();
  vi.stubGlobal('AudioContext', FakeContext);
  vi.stubGlobal(
    'AudioWorkletNode',
    class {
      port = { onmessage: null };
      connect(target: unknown) {
        return target;
      }
      disconnect = vi.fn();
    },
  );
  vi.stubGlobal('navigator', {
    mediaDevices: { getUserMedia: vi.fn(async () => ({ getTracks: () => [{ stop: stopTrack }] })) },
  });
});

it('连续音频块按样本时长串行排队，停止会释放所有源和输出上下文', async () => {
  const audio = new AudioDevices();
  await audio.ensureOutput();
  audio.play(new Float32Array(2400), 24000);
  audio.play(new Float32Array(4800), 24000);
  expect(sources[0].start).toHaveBeenCalledWith(1.2);
  expect(sources[1].start.mock.calls[0][0]).toBeCloseTo(1.3);
  await audio.close();
  expect(sources.every((source) => source.stop.mock.calls.length === 1)).toBe(true);
  expect(sources.every((source) => source.disconnect.mock.calls.length === 1)).toBe(true);
  expect(contexts[0].close).toHaveBeenCalledOnce();
});

it('录音工作线程加载失败时立即释放麦克风以及输入输出上下文', async () => {
  failWorklet = true;
  const audio = new AudioDevices();
  await expect(audio.startCapture(vi.fn())).rejects.toThrow('worklet unavailable');
  expect(audio.capturing).toBe(false);
  expect(stopTrack).toHaveBeenCalledOnce();
  expect(contexts).toHaveLength(2);
  expect(contexts.every((context) => context.close.mock.calls.length === 1)).toBe(true);
});

it('半双工结束采集只关闭输入设备，仍可播放服务返回的音频', async () => {
  const audio = new AudioDevices();
  await audio.startCapture(vi.fn());
  expect(audio.capturing).toBe(true);
  await audio.stopCapture();
  expect(audio.capturing).toBe(false);
  expect(stopTrack).toHaveBeenCalledOnce();
  expect(contexts[1].close).not.toHaveBeenCalled();
  audio.play(new Float32Array(2400), 24000);
  expect(sources[0].start).toHaveBeenCalled();
  await audio.close();
});
