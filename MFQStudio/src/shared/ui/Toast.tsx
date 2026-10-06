/**
 * Global floating notification container that observes the Toast queue and mounts at the application root.
 */
import { useToastStore } from '../../stores/toastStore';
import { useTranslation } from 'react-i18next';
import { ToastItem } from './ToastItem';
import './primitives.css';

export { ToastItem } from './ToastItem';
export type { ToastItemProps } from './ToastItem';

/**
 * Render the application-wide shared Toast container at the root of the app shell.
 */
export function ToastContainer() {
  const { t } = useTranslation();
  const toasts = useToastStore((state) => state.toasts);
  const dismissToast = useToastStore((state) => state.dismissToast);

  if (toasts.length === 0) return null;

  return (
    <div
      className="studio-toast-container"
      role="region"
      aria-label={t('common:notifications')}
      tabIndex={-1}
    >
      {toasts.map((item) => (
        <ToastItem key={item.id} toast={item} onDismiss={dismissToast} />
      ))}
    </div>
  );
}
