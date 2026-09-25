/**
 * 为原生按钮提供同时支持鼠标悬停和键盘聚焦的工具提示组件。
 */
import * as TooltipPrimitive from '@radix-ui/react-tooltip';
import type { ReactElement } from 'react';
import './primitives.css';

/** 工具提示组件属性。 */
export interface TooltipProps {
  /** 悬停或聚焦时显示的提示文字。 */
  content: string;
  /** 子元素必须接受 DOM 属性和 ref，原生按钮可直接使用。 */
  children: ReactElement;
}

/** 全局工具提示上下文 Provider，统一由应用外壳挂载配置全局延迟。 */
export const TooltipProvider = TooltipPrimitive.Provider;

/**
 * 在不新增按钮或布局包装元素的情况下为操作按钮显示提示。
 *
 * @param props 工具提示属性
 */
export function Tooltip({ content, children }: TooltipProps) {
  return (
    <TooltipPrimitive.Root>
      <TooltipPrimitive.Trigger asChild>{children}</TooltipPrimitive.Trigger>
      <TooltipPrimitive.Portal>
        <TooltipPrimitive.Content className="studio-tooltip" sideOffset={6} collisionPadding={12}>
          {content}
          <TooltipPrimitive.Arrow className="studio-tooltip-arrow" />
        </TooltipPrimitive.Content>
      </TooltipPrimitive.Portal>
    </TooltipPrimitive.Root>
  );
}
