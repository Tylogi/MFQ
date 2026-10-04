/** Share quantization state within the page instead of lifting job actions to the application root. */
import { createContext, useContext } from 'react';
import type { useQuantizationWorkspace } from './useQuantizationWorkspace';
export const QuantizationContext = createContext<ReturnType<
  typeof useQuantizationWorkspace
> | null>(null);
/** Get the current quantization workspace state; call only inside the quantization page. */
export function useQuantization() {
  const value = useContext(QuantizationContext);
  if (!value) throw new Error('QuantizationContext is required');
  return value;
}
