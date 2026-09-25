/** 验证实时语音编码、WAV 格式和跨块重采样的数值边界。 */
import { describe, expect, it } from 'vitest';
import { base64ToFloat32, float32ToBase64, StreamingLinearResampler, wavBlob } from './audioCodec';

/** 使用浏览器文件读取接口读取 WAV，兼容 jsdom 的 Blob 实现。 */
function readBlob(blob: Blob): Promise<ArrayBuffer> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as ArrayBuffer);
    reader.onerror = () => reject(reader.error);
    reader.readAsArrayBuffer(blob);
  });
}

describe('PCM Base64', () => {
  it('保留浮点样本的原始字节，包括带偏移视图和大于编码分片的数据', () => {
    const storage = Float32Array.from({ length: 25_000 }, (_, index) => Math.sin(index / 20));
    storage[12] = -0;
    storage[13] = Number.POSITIVE_INFINITY;
    storage[14] = Number.NaN;
    const view = storage.subarray(7, -9);
    const decoded = base64ToFloat32(float32ToBase64(view));
    const originalBytes = new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
    expect(new Uint8Array(decoded.buffer)).toEqual(originalBytes);
    expect(decoded.length).toBe(view.length);
  });

  it('支持空音频，拒绝不能表示完整 Float32 的字节序列', () => {
    expect(base64ToFloat32(float32ToBase64(new Float32Array(0)))).toHaveLength(0);
    expect(() => base64ToFloat32(btoa('abc'))).toThrow('multiple of 4');
  });
});

describe('StreamingLinearResampler', () => {
  it.each([[48_000, 16_000], [44_100, 16_000], [16_000, 24_000]])('从 %i Hz 到 %i Hz 时分块结果与整块一致', (sourceRate, targetRate) => {
    const input = Float32Array.from({ length: 1201 }, (_, index) => Math.sin(index / 13));
    const expected = new StreamingLinearResampler(sourceRate, targetRate).push(input);
    const resampler = new StreamingLinearResampler(sourceRate, targetRate);
    const actual: number[] = [];
    const sizes = [1, 7, 32, 3, 47, 128, 11];
    let offset = 0;
    let block = 0;
    while (offset < input.length) {
      const end = Math.min(input.length, offset + sizes[block % sizes.length]);
      actual.push(...resampler.push(input.subarray(offset, end)));
      expect(resampler.push(new Float32Array(0))).toHaveLength(0);
      offset = end;
      block += 1;
    }
    expect(actual).toHaveLength(expected.length);
    actual.forEach((sample, index) => expect(sample).toBeCloseTo(expected[index], 5));
  });

  it('插值会保留块尾样本，等待下一块到达后继续输出', () => {
    const resampler = new StreamingLinearResampler(2, 4);
    expect(resampler.push(new Float32Array([0]))).toHaveLength(0);
    expect(Array.from(resampler.push(new Float32Array([1])))).toEqual([0, 0.5]);
    expect(Array.from(resampler.push(new Float32Array([0])))).toEqual([1, 0.5]);
  });

  it('同采样率时复制输入，避免采集缓冲复用改写已排队的音频', () => {
    const input = new Float32Array([0.2, 0.8]);
    const output = new StreamingLinearResampler(16_000, 16_000).push(input);
    expect(output).toEqual(input);
    expect(output).not.toBe(input);
    input.fill(0);
    expect(output[1]).toBeCloseTo(0.8);
  });
});

describe('WAV', () => {
  it('写入单声道 PCM 头、合并块并把超出范围的样本裁到 16 位边界', async () => {
    const blob = wavBlob([new Float32Array([-2, -1, -0.5]), new Float32Array([0, 0.5, 1, 2])], 24_000);
    expect(blob.type).toBe('audio/wav');
    const buffer = await readBlob(blob);
    const view = new DataView(buffer);
    const text = (start: number, length: number) => String.fromCharCode(...new Uint8Array(buffer, start, length));
    expect(text(0, 4)).toBe('RIFF');
    expect(text(8, 8)).toBe('WAVEfmt ');
    expect(text(36, 4)).toBe('data');
    expect(view.getUint32(4, true)).toBe(buffer.byteLength - 8);
    expect(view.getUint32(16, true)).toBe(16);
    expect(view.getUint16(20, true)).toBe(1);
    expect(view.getUint16(22, true)).toBe(1);
    expect(view.getUint32(24, true)).toBe(24_000);
    expect(view.getUint32(28, true)).toBe(48_000);
    expect(view.getUint16(32, true)).toBe(2);
    expect(view.getUint16(34, true)).toBe(16);
    expect(view.getUint32(40, true)).toBe(14);
    expect(Array.from({ length: 7 }, (_, index) => view.getInt16(44 + index * 2, true)))
      .toEqual([-32768, -32768, -16384, 0, 16383, 32767, 32767]);
  });

  it('没有样本时仍返回结构完整的空 WAV', async () => {
    const buffer = await readBlob(wavBlob([], 16_000));
    expect(buffer.byteLength).toBe(44);
    expect(new DataView(buffer).getUint32(40, true)).toBe(0);
  });
});
