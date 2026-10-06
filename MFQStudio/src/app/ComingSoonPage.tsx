/** Present a standalone placeholder while the quantization workbench is being prepared. */
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router';
import './coming-soon.css';

/** Show localized availability information without loading the unfinished workbench. */
export function ComingSoonPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();

  return (
    <section className="coming-soon-page" aria-labelledby="coming-soon-title">
      <div className="coming-soon-content">
        <span className="coming-soon-label">{t('app:labLayout.quantization')}</span>
        <h1 id="coming-soon-title">{t('app:comingSoon.title')}</h1>
        <p>{t('app:comingSoon.description')}</p>
        <button className="failure-page-secondary" onClick={() => navigate('/')} type="button">
          {t('app:notFoundPage.backToOverview')}
        </button>
      </div>
    </section>
  );
}
