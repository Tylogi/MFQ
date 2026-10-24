import type { RuntimeConcurrencyPolicy, RuntimeConcurrencySettings } from '../../shared/api/types';

export type ConcurrencyDefaults = { [K in keyof RuntimeConcurrencySettings]: number | null };

export const DEFAULT_CONCURRENCY: ConcurrencyDefaults = {
  max_requests: null, max_queued_requests: null, queue_timeout_seconds: null,
};

/** Omitted fields inherit the server default; never clamp saved preferences to a different server. */
export function concurrencyOverrides(value: ConcurrencyDefaults): Partial<RuntimeConcurrencySettings> {
  return Object.fromEntries(Object.entries(value).filter(([, number]) => number != null));
}

export function validConcurrency(value: ConcurrencyDefaults, policy: RuntimeConcurrencyPolicy): boolean {
  return Object.entries(value).every(([field, number]) => {
    if (number == null) return true;
    const key = field as keyof RuntimeConcurrencySettings;
    const limit = policy.limits[key];
    return Number.isFinite(number) && number >= (key === 'max_requests' ? 1 : 0)
      && (key === 'queue_timeout_seconds' || Number.isSafeInteger(number))
      && (limit == null || number <= limit && (key !== 'queue_timeout_seconds' || number > 0));
  });
}
