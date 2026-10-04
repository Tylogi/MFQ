/** Normalize inference performance metrics, handling field priority and missing values across backends. */


export interface PrefillMetricLike {
  prompt_tokens?: number;
  prefill_tokens?: number;
  ttft_ms?: number;
  prefill_ms?: number;
  prefill_tps?: number;
  model_prefill_ms?: number;
  complete_prefill_ms?: number;
  complete_prefill_tps?: number;
}

/** Calculate prefill latency and throughput according to server metric priority. */
export function displayPrefillMetric(metrics?: PrefillMetricLike | null): {
  milliseconds: number | undefined;
  tokensPerSecond: number | undefined;
} {
  if (!metrics) return { milliseconds: undefined, tokensPerSecond: undefined };
  const languageMilliseconds = Number(metrics.prefill_ms);
  const modelMilliseconds = Number(metrics.model_prefill_ms);
  const milliseconds =
    Number.isFinite(languageMilliseconds) && languageMilliseconds > 0
      ? languageMilliseconds
      : Number.isFinite(modelMilliseconds) && modelMilliseconds > 0
        ? modelMilliseconds
        : undefined;
  const reported = Number(metrics.prefill_tps);
  if (Number.isFinite(reported) && reported > 0) {
    return { milliseconds, tokensPerSecond: reported };
  }
  const tokens = Number(metrics.prefill_tokens ?? metrics.prompt_tokens);
  return {
    milliseconds,
    tokensPerSecond:
      milliseconds !== undefined && Number.isFinite(tokens) && tokens > 0
        ? (tokens * 1000) / milliseconds
        : undefined,
  };
}

/** Prefer the primary metric and fall back to a valid positive metric when it is unavailable. */
export function preferPositiveMetric(primary: unknown, fallback: unknown): number | undefined {
  const preferred = Number(primary);
  if (Number.isFinite(preferred) && preferred > 0) return preferred;
  const fallbackNumber = Number(fallback);
  return Number.isFinite(fallbackNumber) ? fallbackNumber : undefined;
}
