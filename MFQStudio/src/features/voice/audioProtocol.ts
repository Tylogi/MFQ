/** Define the current realtime PCM protocol and reject incompatible advertised audio formats. */
import type { RealtimeCapabilities } from '../../shared/api/types';

export const INPUT_RATE = 16_000;
export const OUTPUT_RATE = 24_000;

/** Check reported sample rates before recording; older servers may omit these protocol fields. */
export function validateAudioProtocol(capabilities?: RealtimeCapabilities | null): void {
  if (capabilities?.input_sample_rate != null && capabilities.input_sample_rate !== INPUT_RATE) {
    throw new Error(`Unsupported voice input sample rate: ${capabilities.input_sample_rate} Hz; expected ${INPUT_RATE} Hz.`);
  }
  if (capabilities?.output_sample_rate != null && capabilities.output_sample_rate !== OUTPUT_RATE) {
    throw new Error(`Unsupported voice output sample rate: ${capabilities.output_sample_rate} Hz; expected ${OUTPUT_RATE} Hz.`);
  }
}
