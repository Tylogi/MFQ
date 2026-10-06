/** Centralize display formatting for errors, numbers, capacity, and duration. */
import { i18n } from '../i18n';
import { ApiError } from '../shared/api/client';
/** Convert server or generic errors into user-visible messages. */
export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) return `${error.code}: ${error.message}`;
  if (error instanceof Error) return error.message;
  return i18n.t('common:unknownError');
}
/** Format a finite number, returning a placeholder when missing or invalid. */
export function formatNumber(value: unknown, digits = 0): string {
  const number = Number(value);
  return Number.isFinite(number)
    ? new Intl.NumberFormat(i18n.resolvedLanguage, { maximumFractionDigits: digits }).format(number)
    : "--";
}
/** Format a local timestamp with a full calendar date and 24-hour time, preserving seconds. */
export function formatDateTime(value: string | number | Date): string {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return '--';
  return new Intl.DateTimeFormat(i18n.resolvedLanguage, {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hourCycle: 'h23',
  }).format(date);
}
/** Convert bytes into a readable binary capacity unit. */
export function formatBytes(value: unknown): string {
  const bytes = Number(value);
  if (!Number.isFinite(bytes) || bytes <= 0) return "--";
  if (bytes >= 2 ** 40) return `${formatNumber(bytes / 2 ** 40, 1)} TB`;
  if (bytes >= 2 ** 30) return `${formatNumber(bytes / 2 ** 30, 1)} GB`;
  if (bytes >= 2 ** 20) return `${formatNumber(bytes / 2 ** 20, 0)} MB`;
  return `${formatNumber(bytes / 2 ** 10, 0)} KB`;
}
/** Convert seconds into a concise duration in days, hours, minutes, and seconds. */
export function formatDuration(value: unknown): string {
  const seconds = Math.max(0, Math.floor(Number(value) || 0));
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (days) return `${days}d ${hours}h`;
  if (hours) return `${hours}h ${minutes}m`;
  return `${minutes}m ${seconds % 60}s`;
}
