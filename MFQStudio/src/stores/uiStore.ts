/**
 * MFQ Studio cross-page interface state, centralizing app-shell and temporary navigation interactions.
 */

import { create } from 'zustand';

interface UiState {
  sidebarOpen: boolean;
  /** Close mobile navigation on route changes or backdrop clicks. */
  closeSidebar: () => void;
  /** Show mobile navigation when the user clicks the menu button. */
  openSidebar: () => void;
  /** Toggle visibility based on the current sidebar state. */
  toggleSidebar: () => void;
}

/** Provide shared app-shell state and avoid passing sidebar controls through page components. */
export const useUiStore = create<UiState>()((set) => ({
  sidebarOpen: false,
  closeSidebar: () => set({ sidebarOpen: false }),
  openSidebar: () => set({ sidebarOpen: true }),
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
}));
