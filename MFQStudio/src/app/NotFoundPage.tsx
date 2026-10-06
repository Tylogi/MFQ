/** 404 page for unregistered routes, retaining workspace navigation and providing a clear way back. */
import { useTranslation } from 'react-i18next';
import { ArrowLeftIcon, HouseIcon } from '@phosphor-icons/react';
import { useNavigate } from 'react-router';
/** Display a standalone page for unknown routes without redirecting or rewriting the user-entered address. */
export function NotFoundPage() {
  const navigate = useNavigate();
  const { t } = useTranslation();

  return (
    <section aria-labelledby="not-found-title" className="failure-page not-found-page">
      <div className="failure-page-content">
        <div className="failure-page-heading">
          <span className="failure-page-code">404 / NOT FOUND</span>
        </div>
        <h1 id="not-found-title">{t('app:notFoundPage.pageNotFound')}</h1>
        <p className="failure-page-description">
          {t('app:notFoundPage.thereIsNoPageAtThisAddressCheckTheUrlOrReturn')}
        </p>
        <div className="failure-page-actions">
          <button className="failure-page-primary" onClick={() => navigate('/')} type="button">
            <HouseIcon size={16} />
            {t('app:notFoundPage.backToOverview')}
          </button>
          <button
            className="failure-page-secondary"
            onClick={() => {
              if (window.history.state?.idx > 0) navigate(-1);
              else navigate('/');
            }}
            type="button"
          >
            <ArrowLeftIcon size={16} />
            {t('app:notFoundPage.goBack')}
          </button>
        </div>
      </div>
      <div className="failure-page-footer" aria-hidden="true">
        <img src="/mfq-mark.svg" alt="" />
        <span>MFQ Studio</span>
      </div>
    </section>
  );
}
