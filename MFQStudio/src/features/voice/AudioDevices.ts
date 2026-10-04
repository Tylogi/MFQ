/** Manage browser microphone, AudioWorklet, and playback resources independently of the response-turn protocol. */
import { StreamingLinearResampler } from './audioCodec';

const INPUT_RATE = 16_000;
const OUTPUT_RATE = 24_000;
const PLAYBACK_DELAY_SECONDS = 0.2;

/** Own capture and playback resources; the controller only receives raw input and submits playback samples. */
export class AudioDevices {
  private stream: MediaStream | null = null;
  private inputContext: AudioContext | null = null;
  private outputContext: AudioContext | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private capture: AudioWorkletNode | null = null;
  private resampler: StreamingLinearResampler | null = null;
  private playAt = 0;
  private playing = new Set<AudioBufferSourceNode>();

  /** Whether a microphone capture context is currently held. */
  get capturing(): boolean {
    return this.inputContext !== null;
  }

  /** Initialize or resume the playback context; must be called from a user-initiated audio action. */
  async ensureOutput(): Promise<void> {
    this.outputContext ??= new AudioContext({ sampleRate: OUTPUT_RATE });
    await this.outputContext.resume();
  }

  /** Request microphone access and connect the worklet, releasing allocated device resources on failure. */
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

  /** Preserve phase across input chunks while converting device samples to the protocol's required 16 kHz format. */
  convert(samples: Float32Array): Float32Array {
    return this.resampler?.push(samples) ?? new Float32Array(samples);
  }

  /** Release the microphone, nodes, and input context; retain the output device when ending half-duplex input. */
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

  /** Queue audio chunks serially and release node references after playback completes. */
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

  /** Stop all queued samples while keeping the output device available for later use. */
  stopPlayback(): void {
    for (const source of this.playing) {
      try {
        source.stop();
      } catch {
        /* The audio may have already ended naturally. */
      }
      source.disconnect();
    }
    this.playing.clear();
    this.playAt = this.outputContext?.currentTime ?? 0;
  }

  /** Fully close input and output resources; safe to call multiple times. */
  async close(): Promise<void> {
    await this.stopCapture();
    this.stopPlayback();
    await this.outputContext?.close().catch(() => undefined);
    this.outputContext = null;
    this.playAt = 0;
  }
}
