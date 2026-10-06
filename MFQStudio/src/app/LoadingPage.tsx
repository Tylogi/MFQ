/** Provide a consistent workspace loading page for app startup, route changes, and service connections. */
import { useTranslation } from 'react-i18next';
/** Display a consistent loading state; the animation does not indicate actual completion percentage. */
export function LoadingPage() {
  const { t } = useTranslation();

  return (
    <section aria-labelledby="loading-title" className="failure-page loading-page" role="status">
      <div className="failure-page-content">
        <div className="failure-page-heading">
          <span className="failure-page-symbol loading-page-symbol" aria-hidden="true" />
        </div>
        <h1 id="loading-title">{t('app:loadingPage.loading')}</h1>
      </div>
      <div className="failure-page-footer" aria-hidden="true">
        <img src="/mfq-mark.svg" alt="" />
        <span>MFQ Studio</span>
      </div>
    </section>
  );
}
