/** 独立管理浏览器麦克风、AudioWorklet 与音频播放资源，不参与响应轮次协议。 */
import { StreamingLinearResampler } from './audioCodec';

const INPUT_RATE = 16_000;
const OUTPUT_RATE = 24_000;
const PLAYBACK_DELAY_SECONDS = 0.2;

/** 拥有采集与播放设备资源，控制器仅接收原始输入并提交播放样本。 */
export class AudioDevices {
  private stream: MediaStream | null = null;
  private inputContext: AudioContext | null = null;
  private outputContext: AudioContext | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private capture: AudioWorkletNode | null = null;
  private resampler: StreamingLinearResampler | null = null;
  private playAt = 0;
  private playing = new Set<AudioBufferSourceNode>();

  /** 当前是否持有麦克风采集上下文。 */
  get capturing(): boolean {
    return this.inputContext !== null;
  }

  /** 初始化或恢复播放上下文，必须在用户触发的音频操作中调用。 */
  async ensureOutput(): Promise<void> {
    this.outputContext ??= new AudioContext({ sampleRate: OUTPUT_RATE });
    await this.outputContext.resume();
  }

  /** 请求麦克风并接入工作线程；失败时释放已分配的设备资源。 */
  async startCapture(onSamples: (samples: Float32Array) => void): Promise<void> {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      try {
        this.inputContext = new AudioContext({ sampleRate: INPUT_RATE });
      } catch {
        this.inputContext = new AudioContext();
      }
      this.resampler = new StreamingLinearResampler(this.inputContext.sampleRate, INPUT_RATE);
      await Promise.all([this.inputContext.resume(), this.ensureOutput()]);
      await this.inputContext.audioWorklet.addModule('/pcm-capture-worklet.js');
      this.source = this.inputContext.createMediaStreamSource(this.stream);
      this.capture = new AudioWorkletNode(this.inputContext, 'pcm-capture');
      const silent = this.inputContext.createGain();
      silent.gain.value = 0;
      this.source.connect(this.capture).connect(silent).connect(this.inputContext.destination);
      this.capture.port.onmessage = (event: MessageEvent<Float32Array>) => onSamples(event.data);
    } catch (cause) {
      await this.close();
      throw cause;
    }
  }

  /** 保持输入分块相位，把设备样本转换为协议要求的 16kHz 数据。 */
  convert(samples: Float32Array): Float32Array {
    return this.resampler?.push(samples) ?? new Float32Array(samples);
  }

  /** 释放麦克风、节点和输入上下文；半双工结束输入时保留输出设备。 */
  async stopCapture(): Promise<void> {
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.source?.disconnect();
    if (this.capture) this.capture.port.onmessage = null;
    this.capture?.disconnect();
    await this.inputContext?.close().catch(() => undefined);
    this.inputContext = null;
    this.resampler = null;
    this.source = null;
    this.capture = null;
  }

  /** 将音频块串行排入播放队列，并在播放结束后释放节点引用。 */
  play(samples: Float32Array, sampleRate: number): void {
    const context = this.outputContext;
    if (!samples.length || !context) return;
    const buffer = context.createBuffer(1, samples.length, sampleRate);
    buffer.copyToChannel(new Float32Array(samples), 0);
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    this.playAt = Math.max(this.playAt, context.currentTime + PLAYBACK_DELAY_SECONDS);
    source.start(this.playAt);
    this.playAt += buffer.duration;
    this.playing.add(source);
    source.onended = () => {
      this.playing.delete(source);
      source.disconnect();
    };
  }

  /** 停止所有待播放样本，但允许后续继续使用输出设备。 */
  stopPlayback(): void {
    for (const source of this.playing) {
      try {
        source.stop();
      } catch {
        /* 音频可能已经自然结束。 */
      }
      source.disconnect();
    }
    this.playing.clear();
    this.playAt = this.outputContext?.currentTime ?? 0;
  }

  /** 完整关闭输入与输出资源，多次调用可安全复用。 */
  async close(): Promise<void> {
    await this.stopCapture();
    this.stopPlayback();
    await this.outputContext?.close().catch(() => undefined);
    this.outputContext = null;
    this.playAt = 0;
  }
}
