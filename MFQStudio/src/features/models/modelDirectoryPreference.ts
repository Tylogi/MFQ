import { getApiBaseUrl } from '../../shared/api/client';

function storageKey() {
  return `mfq.studio.model-directory.v1:${getApiBaseUrl() || window.location.origin}`;
}

export function readModelDirectory(): string {
  try {
    return localStorage.getItem(storageKey())?.trim() || '/';
  } catch {
    return '/';
  }
}

export function saveModelDirectory(path: string): void {
  try {
    localStorage.setItem(storageKey(), path);
  } catch {}
}
