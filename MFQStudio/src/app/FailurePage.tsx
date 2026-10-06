/** Display blocking workspace failures consistently, with retry and leave-page actions. */
import { useTranslation } from 'react-i18next';
import { ArrowLeftIcon, ArrowClockwiseIcon, PlugsIcon, WarningCircleIcon } from '@phosphor-icons/react';
import { useNavigate } from 'react-router';

interface FailurePageProps {
  kind: 'connection' | 'render';
  detail?: string | null;
  onRetry: () => void;
}

interface FailureViewProps {
  kind: 'connection' | 'render';
  code: string;
  title: string;
  description: string;
  retryLabel: string;
  leaveLabel: string;
  detailLabel: string;
  detail?: string | null;
  onRetry: () => void;
  onLeave: () => void;
}
/** Render the failure page without application context so the top-level error boundary can safely use it. */
export function FailureView({ kind, code, title, description, retryLabel, leaveLabel, detailLabel, detail, onRetry, onLeave }: FailureViewProps) {
  return (
    <section aria-labelledby="failure-title" className="failure-page" role="alert">
      <div className="failure-page-content">
        <div className="failure-page-heading">
          <span className="failure-page-symbol" aria-hidden="true">
            {kind === 'connection' ? <PlugsIcon size={27} weight="regular" /> : <WarningCircleIcon size={27} weight="regular" />}
          </span>
          <span className="failure-page-code">{code}</span>
        </div>
        <h1 id="failure-title">{title}</h1>
        <p className="failure-page-description">{description}</p>
        <div className="failure-page-actions">
          <button className="failure-page-primary" onClick={onRetry} type="button">
            <ArrowClockwiseIcon size={16} />
            {retryLabel}
          </button>
          <button
            className="failure-page-secondary"
            onClick={onLeave}
            type="button"
          >
            <ArrowLeftIcon size={16} />
            {leaveLabel}
          </button>
        </div>
        {detail && (
          <details className="failure-page-details">
            <summary>{detailLabel}</summary>
            <pre>{detail}</pre>
          </details>
        )}
      </div>
      <div className="failure-page-footer" aria-hidden="true">
        <img src="/mfq-mark.svg" alt="" />
        <span>MFQ Studio</span>
      </div>
    </section>
  );
}
/** Show a recoverable failure page in the main workspace, allowing connection failures to navigate to server settings. */
export function FailurePage({ kind, detail, onRetry }: FailurePageProps) {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const connection = kind === 'connection';

  return (
    <FailureView
      code={connection ? 'CONNECTION / 01' : 'PAGE / 02'}
      description={connection
        ? t('app:failurePage.workspaceDataIsUnavailableCheckTheServiceOrConnectionSettingsThenTry')
        : t('app:failurePage.somethingWentWrongOnThisPageTryAgainOrReturnToThe')}
      detail={detail}
      detailLabel={t('app:failurePage.viewErrorDetails')}
      kind={kind}
      leaveLabel={connection ? t('app:failurePage.connectionSettings') : t('app:failurePage.backToOverview')}
      onLeave={() => navigate(connection ? '/runtime' : '/')}
      onRetry={onRetry}
      retryLabel={connection ? t('app:failurePage.reconnect') : t('app:failurePage.retryPage')}
      title={connection ? t('app:failurePage.serviceUnavailable') : t('app:failurePage.pageUnavailable')}
    />
  );
}
