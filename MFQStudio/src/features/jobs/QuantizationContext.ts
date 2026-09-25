/** 页面范围内共享量化状态，避免将任务操作提升到应用根组件。 */
import { createContext, useContext } from 'react';
import type { useQuantizationWorkspace } from './useQuantizationWorkspace';
export const QuantizationContext = createContext<ReturnType<
  typeof useQuantizationWorkspace
> | null>(null);
/** 获取当前量化工作台状态，仅允许在量化页面内部调用。 */
export function useQuantization() {
  const value = useContext(QuantizationContext);
  if (!value) throw new Error('QuantizationContext is required');
  return value;
}
