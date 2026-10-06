/**
 * Shared React entry point for MFQ Studio Web and Tauri, responsible for routing and top-level error isolation.
 */
import { i18n } from './i18n';
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
    const t = i18n.t.bind(i18n);
    return (
      <main className="fatal-workspace">
        <FailureView
          code="APP / 03"
          description={t('app:main.theInterfaceEncounteredAnUnexpectedProblemTheServiceMayStillBeRunning')}
          detail={this.state.error.message || this.state.error.name}
          detailLabel={t('app:main.viewErrorDetails')}
          kind="render"
          leaveLabel={t('app:main.backToOverview')}
          onLeave={() => window.location.assign(isStudio() ? '#/' : '/')}
          onRetry={() => window.location.reload()}
          retryLabel={t('app:main.reload')}
          title={t('app:main.interfaceUnavailable')}
        />
      </main>
    );
  }
}

// Clean up legacy root-path hash links in browsers; packaged desktop assets still use hash routing.
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
