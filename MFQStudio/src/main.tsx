/**
 * MFQ Studio Web 与 Tauri 共用的 React 启动入口，负责路由和顶层错误隔离。
 */

import { Component, StrictMode, type ErrorInfo, type ReactNode } from 'react';
import { createRoot } from 'react-dom/client';
import { createBrowserRouter, createHashRouter, RouterProvider } from 'react-router';

import App from './App';
import { FailureView } from './app/FailurePage';
import { isStudio } from './studio';
import './styles.css';

interface AppErrorBoundaryState {
  error: Error | null;
}

class AppErrorBoundary extends Component<{ children: ReactNode }, AppErrorBoundaryState> {
  state: AppErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): AppErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("MFQ Studio failed to render", error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    const chinese = navigator.language.toLowerCase().startsWith("zh");
    return (
      <main className="fatal-workspace">
        <FailureView
          code="APP / 03"
          description={chinese
            ? '应用界面遇到了意外问题。服务可能仍在运行，可以重新载入界面后继续。'
            : 'The interface encountered an unexpected problem. The service may still be running; reload to continue.'}
          detail={this.state.error.message || this.state.error.name}
          detailLabel={chinese ? '查看错误详情' : 'View error details'}
          kind="render"
          leaveLabel={chinese ? '返回概览' : 'Back to overview'}
          onLeave={() => window.location.assign(isStudio() ? '#/' : '/')}
          onRetry={() => window.location.reload()}
          retryLabel={chinese ? '重新载入' : 'Reload'}
          title={chinese ? '界面暂时无法显示' : 'Interface unavailable'}
        />
      </main>
    );
  }
}

// 浏览器清理旧版根路径哈希链接；桌面打包资源仍使用哈希路由。
if (!isStudio() && window.location.pathname === '/' && window.location.hash.startsWith('#/')) {
  window.history.replaceState(window.history.state, '', window.location.hash.slice(1));
}

const createRouter = isStudio() ? createHashRouter : createBrowserRouter;
const router = createRouter([
  {
    path: '*',
    element: <App />,
  },
]);

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <AppErrorBoundary>
      <RouterProvider router={router} />
    </AppErrorBoundary>
  </StrictMode>,
);
