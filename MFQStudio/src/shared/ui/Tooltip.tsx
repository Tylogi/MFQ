/**
 * Provide native buttons with tooltips that respond to both pointer hover and keyboard focus.
 */
import * as TooltipPrimitive from '@radix-ui/react-tooltip';
import type { ReactElement } from 'react';
import './primitives.css';

/** Tooltip component properties. */
export interface TooltipProps {
  /** Tooltip text shown on hover or focus. */
  content: string;
  /** The child must accept DOM properties and a ref; native buttons work directly. */
  children: ReactElement;
}

/** Global tooltip context provider, mounted by the app shell to configure a shared delay. */
export const TooltipProvider = TooltipPrimitive.Provider;

/**
 * Show a tooltip for an action button without adding button or layout wrapper elements.
 *
 * @param props Tooltip properties
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
