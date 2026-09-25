/** 提供实时语音使用的 PCM 编解码、连续重采样及 WAV 封装。 */
/** 将 Float32 视图的有效字节编码为 Base64，支持带偏移的大音频块。 */
export function float32ToBase64(values: Float32Array): string {
  const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
  }
  return btoa(binary);
}

/** 将服务端 Base64 PCM 数据恢复为 Float32 样本数组。 */
export function base64ToFloat32(value: string): Float32Array {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return new Float32Array(bytes.buffer);
}

/** 保存相邻块的采样位置，在线性插值时保持跨块重采样连续。 */
export class StreamingLinearResampler {
  private buffer = new Float32Array(0);
  /** 使用目标采样率作为分母累积相位，避免小数步长误差随分块变化。 */
  private position = 0;

  constructor(
    private readonly sourceRate: number,
    private readonly targetRate: number,
  ) {}

  /** 追加采集块并返回可插值的样本；末尾不足一对的样本留给下一块。 */
  push(input: Float32Array): Float32Array {
    if (!input.length) return new Float32Array(0);
    if (this.sourceRate === this.targetRate) return new Float32Array(input);

    const joined = new Float32Array(this.buffer.length + input.length);
    joined.set(this.buffer);
    joined.set(input, this.buffer.length);
    this.buffer = joined;

    let count = 0;
    for (let cursor = this.position; Math.floor(cursor / this.targetRate) + 1 < joined.length; cursor += this.sourceRate) {
      count += 1;
    }
    const output = new Float32Array(count);
    for (let index = 0; index < count; index += 1) {
      const left = Math.floor(this.position / this.targetRate);
      const mix = (this.position % this.targetRate) / this.targetRate;
      output[index] = joined[left] * (1 - mix) + joined[left + 1] * mix;
      this.position += this.sourceRate;
    }

    const discard = Math.min(joined.length, Math.floor(this.position / this.targetRate));
    if (discard > 0) {
      this.buffer = joined.slice(discard);
      this.position -= discard * this.targetRate;
    }
    return output;
  }
}

/** 将多个单声道浮点音频块裁幅并编码为 16 位小端 PCM WAV。 */
export function wavBlob(chunks: Float32Array[], sampleRate: number): Blob {
  const sampleCount = chunks.reduce((total, chunk) => total + chunk.length, 0);
  const buffer = new ArrayBuffer(44 + sampleCount * 2);
  const view = new DataView(buffer);
  const writeText = (offset: number, value: string) => {
    for (let index = 0; index < value.length; index += 1) {
      view.setUint8(offset + index, value.charCodeAt(index));
    }
  };
  writeText(0, "RIFF");
  view.setUint32(4, 36 + sampleCount * 2, true);
  writeText(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeText(36, "data");
  view.setUint32(40, sampleCount * 2, true);
  let offset = 44;
  for (const chunk of chunks) {
    for (const raw of chunk) {
      const sample = Math.max(-1, Math.min(1, raw));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      offset += 2;
    }
  }
  return new Blob([buffer], { type: "audio/wav" });
}
