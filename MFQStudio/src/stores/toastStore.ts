/**
 * Global lightweight notification (Toast) state management for maintaining the notification queue and auto-dismissal.
 */
import { create } from 'zustand';
import type { DisplayMessage } from '../i18n/messages';

/** Notification type. */
export type ToastType = 'error' | 'success' | 'info' | 'warning';

/** Data structure for a single notification. */
export interface ToastItemData {
  /** Unique identifier. */
  id: string;
  /** Incremented on each display to reset timing and identify stale close callbacks. */
  revision: number;
  /** Notification type. */
  type: ToastType;
  /** Main notification content. */
  message: DisplayMessage;
  /** Optional notification title. */
  title?: DisplayMessage;
  /** Display duration in milliseconds; values at or below zero keep it visible until manually closed. */
  duration?: number;
}

/** Input options for triggering a notification. */
export interface ShowToastOptions {
  /** Optional ID; an existing notification with the same ID is replaced. */
  id?: string;
  /** Notification type; defaults to 'info'. */
  type?: ToastType;
  /** Notification message text. */
  message: DisplayMessage;
  /** Optional title. */
  title?: DisplayMessage;
  /** Duration in milliseconds; defaults by type when omitted (5000 ms for errors, 3500 ms otherwise). */
  duration?: number;
}

interface ToastState {
  /** Notifications currently queued for display. */
  toasts: ToastItemData[];
  /**
   * Push a new notification and return its unique ID.
   *
   * @param options Notification options.
   * @returns The assigned notification ID.
   */
  showToast: (options: ShowToastOptions) => string;
  /**
   * Close and remove the notification with the specified ID.
   *
   * @param id ID of the notification to close.
   * @param revision Optional display revision to prevent an old timer from closing a newer notification with the same ID.
   */
  dismissToast: (id: string, revision?: number) => void;
  /** Clear all currently displayed notifications. */
  clearToasts: () => void;
}

let toastIdCounter = 0;
let toastRevisionCounter = 0;

/** Global Zustand notification store. */
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
 * Shortcut for imperative Toast calls, usable directly from any component or regular function.
 */
export const toast = {
  /**
   * Trigger an error notification.
   *
   * @param message Error message.
   * @param options Optional title or display duration.
   */
  error: (message: DisplayMessage, options?: { title?: DisplayMessage; duration?: number; id?: string }) =>
    useToastStore.getState().showToast({ ...options, type: 'error', message }),

  /**
   * Trigger a success notification.
   *
   * @param message Success message.
   * @param options Optional title or display duration.
   */
  success: (message: DisplayMessage, options?: { title?: DisplayMessage; duration?: number; id?: string }) =>
    useToastStore.getState().showToast({ ...options, type: 'success', message }),

  /**
   * Trigger an informational notification.
   *
   * @param message Notification message.
   * @param options Optional title or display duration.
   */
  info: (message: DisplayMessage, options?: { title?: DisplayMessage; duration?: number; id?: string }) =>
    useToastStore.getState().showToast({ ...options, type: 'info', message }),

  /**
   * Trigger a warning notification.
   *
   * @param message Warning message.
   * @param options Optional title or display duration.
   */
  warning: (message: DisplayMessage, options?: { title?: DisplayMessage; duration?: number; id?: string }) =>
    useToastStore.getState().showToast({ ...options, type: 'warning', message }),

  /**
   * Trigger a custom notification.
   *
   * @param options Complete notification options.
   */
  show: (options: ShowToastOptions) => useToastStore.getState().showToast(options),

  /**
   * Manually close the specified notification.
   *
   * @param id Notification ID.
   */
  dismiss: (id: string) => useToastStore.getState().dismissToast(id),

  /** Clear all notifications. */
  clear: () => useToastStore.getState().clearToasts(),
};
