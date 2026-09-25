/** 归一化推理性能指标，处理各后端上报字段的优先级和缺失值。 */


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

/** 按服务指标优先级计算预填充耗时及吞吐量。 */
export function displayPrefillMetric(metrics?: PrefillMetricLike | null): {
  milliseconds: number | undefined;
  tokensPerSecond: number | undefined;
} {
  if (!metrics) return { milliseconds: undefined, tokensPerSecond: undefined };
  const nativeMilliseconds = Number(metrics.ttft_ms);
  if (Number.isFinite(nativeMilliseconds) && nativeMilliseconds > 0) {
    const tokens = Number(metrics.prefill_tokens ?? metrics.prompt_tokens);
    return {
      milliseconds: nativeMilliseconds,
      tokensPerSecond:
        Number.isFinite(tokens) && tokens > 0
          ? (tokens * 1000) / nativeMilliseconds
          : undefined,
    };
  }
  const modelMilliseconds = Number(metrics.model_prefill_ms);
  const languageMilliseconds = Number(metrics.prefill_ms);
  const milliseconds =
    Number.isFinite(modelMilliseconds) && modelMilliseconds > 0
      ? modelMilliseconds
      : Number.isFinite(languageMilliseconds) && languageMilliseconds > 0
        ? languageMilliseconds
        : undefined;
  const tokens = Number(metrics.prefill_tokens ?? metrics.prompt_tokens);
  if (milliseconds !== undefined && Number.isFinite(tokens) && tokens > 0) {
    return { milliseconds, tokensPerSecond: (tokens * 1000) / milliseconds };
  }
  const reported = Number(metrics.prefill_tps);
  return {
    milliseconds,
    tokensPerSecond: Number.isFinite(reported) ? reported : undefined,
  };
}

/** 优先选取主指标，缺失时回退到有效的正数指标。 */
export function preferPositiveMetric(primary: unknown, fallback: unknown): number | undefined {
  const preferred = Number(primary);
  if (Number.isFinite(preferred) && preferred > 0) return preferred;
  const fallbackNumber = Number(fallback);
  return Number.isFinite(fallbackNumber) ? fallbackNumber : undefined;
}
