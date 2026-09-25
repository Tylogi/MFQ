/**
 * 全局浮层通知容器组件，监听 Toast 队列并挂载于应用顶层。
 */
import { useToastStore } from '../../stores/toastStore';
import { ToastItem } from './ToastItem';
import './primitives.css';

export { ToastItem } from './ToastItem';
export type { ToastItemProps } from './ToastItem';

/**
 * 渲染全应用共享的 Toast 浮动容器，挂载于应用外壳根层级。
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
