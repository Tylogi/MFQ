/** 模型文件夹偏好按服务器隔离；没有设置时从文件系统根目录开始。 */
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
  } catch {
    // 禁用浏览器存储不应阻止本次选择或模型注册。
  }
}
