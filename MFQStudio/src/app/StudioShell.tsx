/** Organize navigation, route outlets, and connection state without owning business forms. */
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
/** Isolate rendering errors to an individual business route and restore other pages after navigation. */
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
/** Render fixed navigation and nested routes, retaining direct-page access before readiness and 404 behavior. */
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
