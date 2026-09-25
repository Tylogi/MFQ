/** 统一应用的错误、数值、容量和时长展示格式。 */
import { ApiError } from '../shared/api/client';

/** 将服务端或普通异常转换为用户可见的错误信息。 */
export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) return `${error.code}: ${error.message}`;
  if (error instanceof Error) return error.message;
  return "Unknown error";
}

/** 格式化有限数值，缺失或无效时返回占位符。 */
export function formatNumber(value: unknown, digits = 0): string {
  const number = Number(value);
  return Number.isFinite(number)
    ? new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(number)
    : "--";
}

/** 将字节数转换为可读的二进制容量单位。 */
export function formatBytes(value: unknown): string {
  const bytes = Number(value);
  if (!Number.isFinite(bytes) || bytes <= 0) return "--";
  if (bytes >= 2 ** 40) return `${formatNumber(bytes / 2 ** 40, 1)} TB`;
  if (bytes >= 2 ** 30) return `${formatNumber(bytes / 2 ** 30, 1)} GB`;
  if (bytes >= 2 ** 20) return `${formatNumber(bytes / 2 ** 20, 0)} MB`;
  return `${formatNumber(bytes / 2 ** 10, 0)} KB`;
}

/** 将秒数转换为天、小时、分钟与秒的简洁时长。 */
export function formatDuration(value: unknown): string {
  const seconds = Math.max(0, Math.floor(Number(value) || 0));
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (days) return `${days}d ${hours}h`;
  if (hours) return `${hours}h ${minutes}m`;
  return `${minutes}m ${seconds % 60}s`;
}
