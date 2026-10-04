/** Provide a consistent workspace loading page for app startup, route changes, and service connections. */
import { useSettings } from '../features/settings/SettingsProvider';
/** Display a consistent loading state; the animation does not indicate actual completion percentage. */
export function LoadingPage() {
  const { tr } = useSettings();

  return (
    <section aria-labelledby="loading-title" className="failure-page loading-page" role="status">
      <div className="failure-page-content">
        <div className="failure-page-heading">
          <span className="failure-page-symbol loading-page-symbol" aria-hidden="true" />
        </div>
        <h1 id="loading-title">{tr('正在加载中', 'Loading…')}</h1>
      </div>
      <div className="failure-page-footer" aria-hidden="true">
        <img src="/mfq-mark.svg" alt="" />
        <span>MFQ Studio</span>
      </div>
    </section>
  );
}
