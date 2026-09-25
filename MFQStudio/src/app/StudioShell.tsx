/** 应用外壳组织导航、路由出口及连接状态，不持有业务表单。 */
import { Component, Suspense, useEffect, type ReactNode } from 'react';
import { useLocation, Outlet } from 'react-router';
import * as TooltipPrimitive from '@radix-ui/react-tooltip';
import { useRuntime } from './RuntimeProvider';
import { useSettings } from '../features/settings/SettingsProvider';
import { ToastContainer } from '../shared/ui/Toast';
import { FailurePage } from './FailurePage';
import { LoadingPage } from './LoadingPage';
import { StudioSidebar } from './StudioSidebar';
import { RuntimeAlerts } from './RuntimeAlerts';
import { resolveStudioLocation, isStudioPath } from '../navigation';

/** 隔离单个业务路由的渲染错误，切换路径后恢复其他页面。 */
class PageErrorBoundary extends Component<{ children: ReactNode }, { detail: string | null }> {
  state = { detail: null as string | null };
  static getDerivedStateFromError(error: unknown) {
    return { detail: error instanceof Error ? error.message : String(error) };
  }
  render() {
    if (this.state.detail !== null)
      return (
        <FailurePage detail={this.state.detail} kind="render"
          onRetry={() => this.setState({ detail: null })} />
      );
    return this.props.children;
  }
}

/** 渲染固定导航及嵌套路由，保留未就绪时的直达页和 404 行为。 */
export function StudioShell() {
  const location = useLocation();
  const { tr } = useSettings();
  const { connectionError, refreshError, jobStreamErrors, ready, reloadService } = useRuntime();
  const currentLocation = resolveStudioLocation(location.pathname);
  const view = isStudioPath(location.pathname) ? currentLocation.view : 'not-found';
  const pageAvailable = ready || view === 'not-found' ||
    location.pathname === '/runtime' || location.pathname === '/settings';
  const connectionProblem = ready ? connectionError : connectionError ?? refreshError;
  const hasAlerts = pageAvailable && Boolean(
    connectionProblem || (ready && (refreshError || Object.keys(jobStreamErrors).length)),
  );
  useEffect(() => {
    const title = location.pathname === '/'
      ? 'Overview' : location.pathname.slice(1).replaceAll('-', ' ');
    document.title = `${title} · MFQ Studio`;
  }, [location.pathname]);

  return (
    <TooltipPrimitive.Provider delayDuration={300}>
      <div className="app-shell">
        <ToastContainer />
        <a className="skip-link" href="#studio-main">
          {tr('跳到主要内容', 'Skip to main content')}
        </a>
        <StudioSidebar />
        <main className={`${view === 'chat' ? 'workspace chat-workspace' : 'workspace'}${hasAlerts ? ' has-runtime-alerts' : ''}`}
          id="studio-main" tabIndex={-1}>
          {hasAlerts && <RuntimeAlerts connectionProblem={connectionProblem} />}
          {pageAvailable ? (
            <PageErrorBoundary key={location.pathname}>
              <Suspense fallback={<LoadingPage />}>
                <Outlet />
              </Suspense>
            </PageErrorBoundary>
          ) : connectionProblem ? (
            <FailurePage detail={connectionProblem} kind="connection"
              onRetry={() => void reloadService()} />
          ) : <LoadingPage />}
        </main>
      </div>
    </TooltipPrimitive.Provider>
  );
}
