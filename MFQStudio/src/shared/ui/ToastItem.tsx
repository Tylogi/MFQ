/**
 * Individual notification item supporting accessible semantics, status icons, countdown, and pause-on-hover.
 */
import { useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { displayMessage } from '../../i18n/messages';
import {
  CheckCircleIcon,
  InfoIcon,
  WarningCircleIcon,
  WarningIcon,
  XIcon,
} from '@phosphor-icons/react';
import type { ToastItemData } from '../../stores/toastStore';

export interface ToastItemProps {
  /** Notification data currently being displayed. */
  toast: ToastItemData;
  /** Callback to close this notification. */
  onDismiss: (id: string, revision?: number) => void;
}

/**
 * Render a single Toast notification with an entrance animation and accessibility attributes.
 *
 * @param props Component properties
 */
export function ToastItem({ toast, onDismiss }: ToastItemProps) {
  const { t } = useTranslation();
  const { id, revision, type, message, title, duration } = toast;
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const remainingTimeRef = useRef<number>(duration ?? 0);
  const startTimeRef = useRef<number>(Date.now());
  const hoveredRef = useRef(false);

  /** Start the auto-dismiss timer. */
  const startTimer = () => {
    if (!duration || duration <= 0) return;
    startTimeRef.current = Date.now();
    timerRef.current = setTimeout(() => {
      onDismiss(id, revision);
    }, remainingTimeRef.current);
  };

  /** Clear the timer. */
  const clearTimer = () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  };

  useEffect(() => {
    remainingTimeRef.current = duration ?? 0;
    if (!hoveredRef.current) startTimer();
    return () => clearTimer();
  }, [id, revision, duration, onDismiss]);

  /** Pause the countdown when the pointer enters. */
  const handleMouseEnter = () => {
    if (hoveredRef.current) return;
    hoveredRef.current = true;
    if (!duration || duration <= 0) return;
    clearTimer();
    const elapsed = Date.now() - startTimeRef.current;
    remainingTimeRef.current = Math.max(remainingTimeRef.current - elapsed, 1000);
  };

  /** Resume the countdown when the pointer leaves. */
  const handleMouseLeave = () => {
    hoveredRef.current = false;
    if (!duration || duration <= 0) return;
    startTimer();
  };

  return (
    <div
      className={`studio-toast studio-toast-${type}`}
      role={type === 'error' ? 'alert' : 'status'}
      aria-live={type === 'error' ? 'assertive' : 'polite'}
      aria-atomic="true"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <div className="studio-toast-icon" aria-hidden="true">
        {type === 'error' && <WarningCircleIcon size={18} weight="fill" />}
        {type === 'success' && <CheckCircleIcon size={18} weight="fill" />}
        {type === 'warning' && <WarningIcon size={18} weight="fill" />}
        {type === 'info' && <InfoIcon size={18} weight="fill" />}
      </div>
      <div className="studio-toast-content">
        {title && <div className="studio-toast-title">{displayMessage(title, t)}</div>}
        <div className="studio-toast-message">{displayMessage(message, t)}</div>
      </div>
      <button
        type="button"
        className="studio-toast-close"
        onClick={() => onDismiss(id, revision)}
        aria-label={t('common:dismissNotification')}
      >
        <XIcon size={14} aria-hidden="true" />
      </button>
    </div>
  );
}
