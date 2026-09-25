/**
 * MFQ Studio 跨页面界面状态，集中管理应用外壳与临时导航交互。
 */

import { create } from 'zustand';

interface UiState {
  sidebarOpen: boolean;
  /** 路由跳转或遮罩点击时关闭移动端导航。 */
  closeSidebar: () => void;
  /** 用户点击菜单按钮时显示移动端导航。 */
  openSidebar: () => void;
  /** 在当前侧栏状态基础上切换显示。 */
  toggleSidebar: () => void;
}

/** 提供应用外壳共享状态，避免页面组件层层传递侧栏控制函数。 */
export const useUiStore = create<UiState>()((set) => ({
  sidebarOpen: false,
  closeSidebar: () => set({ sidebarOpen: false }),
  openSidebar: () => set({ sidebarOpen: true }),
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
}));
