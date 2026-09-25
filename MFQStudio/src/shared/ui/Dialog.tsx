/** 提供沿用 Studio 样式的受控模态弹窗和外部触发器焦点恢复。 */
import * as DialogPrimitive from '@radix-ui/react-dialog';
import { XIcon } from '@phosphor-icons/react';
import { useRef, type ReactNode, type RefObject } from 'react';

interface DialogProps {
  open: boolean;
  /** 用户按 Esc、点击遮罩或关闭按钮时通知父级更新受控状态。 */
  onOpenChange: (open: boolean) => void;
  title: string;
  description?: string;
  closeLabel: string;
  className?: string;
  children: ReactNode;
  /** 异步打开弹窗时可显式记录触发按钮，关闭后恢复键盘焦点。 */
  returnFocusRef?: RefObject<HTMLElement | null>;
}

/** 渲染带标题关联、焦点约束和关闭恢复的弹窗；内容沿用现有布局类。 */
export function Dialog({
  open,
  onOpenChange,
  title,
  description,
  closeLabel,
  className = '',
  children,
  returnFocusRef,
}: DialogProps) {
  const previousFocus = useRef<HTMLElement | null>(null);

  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="dialog-backdrop">
          <DialogPrimitive.Content
            className={`studio-dialog ${className}`.trim()}
            {...(!description ? { 'aria-describedby': undefined } : {})}
            onOpenAutoFocus={() => {
              previousFocus.current =
                document.activeElement instanceof HTMLElement ? document.activeElement : null;
            }}
            onCloseAutoFocus={(event) => {
              const target = returnFocusRef?.current ?? previousFocus.current;
              if (target?.isConnected) {
                event.preventDefault();
                target.focus({ preventScroll: true });
              }
            }}
          >
            <header>
              <div>
                <DialogPrimitive.Title asChild>
                  <h2>{title}</h2>
                </DialogPrimitive.Title>
                {description && (
                  <DialogPrimitive.Description asChild>
                    <p>{description}</p>
                  </DialogPrimitive.Description>
                )}
              </div>
              <DialogPrimitive.Close asChild>
                <button type="button" aria-label={closeLabel}>
                  <XIcon size={18} aria-hidden="true" />
                </button>
              </DialogPrimitive.Close>
            </header>
            {children}
          </DialogPrimitive.Content>
        </DialogPrimitive.Overlay>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}
