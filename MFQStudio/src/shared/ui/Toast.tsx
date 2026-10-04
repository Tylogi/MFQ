/**
 * Global floating notification container that observes the Toast queue and mounts at the application root.
 */
import { useToastStore } from '../../stores/toastStore';
import { ToastItem } from './ToastItem';
import './primitives.css';

export { ToastItem } from './ToastItem';
export type { ToastItemProps } from './ToastItem';

/**
 * Render the application-wide shared Toast container at the root of the app shell.
 */
export function ToastContainer() {
  const toasts = useToastStore((state) => state.toasts);
  const dismissToast = useToastStore((state) => state.dismissToast);

  if (toasts.length === 0) return null;

  return (
    <div
      className="studio-toast-container"
      role="region"
      aria-label="通知提示"
      tabIndex={-1}
    >
      {toasts.map((item) => (
        <ToastItem key={item.id} toast={item} onDismiss={dismissToast} />
      ))}
    </div>
  );
}
