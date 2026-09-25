/**
 * 全局轻量通知（Toast）状态管理，负责维护提示队列与自动销毁。
 */
import { create } from 'zustand';

/** 通知提示类型。 */
export type ToastType = 'error' | 'success' | 'info' | 'warning';

/** 单条通知数据结构。 */
export interface ToastItemData {
  /** 唯一标识符。 */
  id: string;
  /** 每次展示时递增，用于重置计时并识别过期的关闭回调。 */
  revision: number;
  /** 通知类型。 */
  type: ToastType;
  /** 通知主要内容。 */
  message: string;
  /** 可选的通知标题。 */
  title?: string;
  /** 停留时间（毫秒），小于等于 0 表示常驻直至手动关闭。 */
  duration?: number;
}

/** 触发通知时的输入参数。 */
export interface ShowToastOptions {
  /** 可选指定 ID；若相同则替换已有通知。 */
  id?: string;
  /** 通知类型，默认为 'info'。 */
  type?: ToastType;
  /** 提示消息文本。 */
  message: string;
  /** 可选标题。 */
  title?: string;
  /** 持续时间（毫秒）；未指定时按类型默认（错误 5000ms，其他 3500ms）。 */
  duration?: number;
}

interface ToastState {
  /** 当前待展示的通知列表。 */
  toasts: ToastItemData[];
  /**
   * 推送新通知，返回通知唯一 ID。
   *
   * @param options 通知配置项
   * @returns 分配的通知 ID
   */
  showToast: (options: ShowToastOptions) => string;
  /**
   * 关闭并移除指定 ID 的通知。
   *
   * @param id 要关闭的通知 ID
   * @param revision 可选的展示版本，避免旧定时器关闭同 ID 的新通知
   */
  dismissToast: (id: string, revision?: number) => void;
  /** 清空所有当前显示的通知。 */
  clearToasts: () => void;
}

let toastIdCounter = 0;
let toastRevisionCounter = 0;

/** Zustand 全局通知 Store。 */
export const useToastStore = create<ToastState>()((set) => ({
  toasts: [],
  showToast: (options) => {
    const id = options.id ?? `toast-${Date.now()}-${++toastIdCounter}`;
    const defaultDuration = options.type === 'error' ? 5000 : 3500;
    const duration = options.duration !== undefined ? options.duration : defaultDuration;
    const item: ToastItemData = {
      id,
      revision: ++toastRevisionCounter,
      type: options.type ?? 'info',
      message: options.message,
      title: options.title,
      duration,
    };
    set((state) => ({
      toasts: [...state.toasts.filter((t) => t.id !== id), item],
    }));
    return id;
  },
  dismissToast: (id, revision) => {
    set((state) => ({
      toasts: state.toasts.filter((t) => t.id !== id || (revision !== undefined && t.revision !== revision)),
    }));
  },
  clearToasts: () => set({ toasts: [] }),
}));

/**
 * 快捷命令式 Toast 调用工具，支持在任意组件或普通函数中直接调用。
 */
export const toast = {
  /**
   * 触发错误类型通知。
   *
   * @param message 错误信息
   * @param options 可选标题或显示时长
   */
  error: (message: string, options?: { title?: string; duration?: number; id?: string }) =>
    useToastStore.getState().showToast({ ...options, type: 'error', message }),

  /**
   * 触发成功类型通知。
   *
   * @param message 成功提示信息
   * @param options 可选标题或显示时长
   */
  success: (message: string, options?: { title?: string; duration?: number; id?: string }) =>
    useToastStore.getState().showToast({ ...options, type: 'success', message }),

  /**
   * 触发普通信息类型通知。
   *
   * @param message 提示信息
   * @param options 可选标题或显示时长
   */
  info: (message: string, options?: { title?: string; duration?: number; id?: string }) =>
    useToastStore.getState().showToast({ ...options, type: 'info', message }),

  /**
   * 触发警告类型通知。
   *
   * @param message 警告信息
   * @param options 可选标题或显示时长
   */
  warning: (message: string, options?: { title?: string; duration?: number; id?: string }) =>
    useToastStore.getState().showToast({ ...options, type: 'warning', message }),

  /**
   * 通用自定义通知。
   *
   * @param options 完整通知配置项
   */
  show: (options: ShowToastOptions) => useToastStore.getState().showToast(options),

  /**
   * 手动关闭指定通知。
   *
   * @param id 通知 ID
   */
  dismiss: (id: string) => useToastStore.getState().dismissToast(id),

  /** 清空所有通知。 */
  clear: () => useToastStore.getState().clearToasts(),
};
