/** Display blocking workspace failures consistently, with retry and leave-page actions. */
import { ArrowLeftIcon, ArrowClockwiseIcon, PlugsIcon, WarningCircleIcon } from '@phosphor-icons/react';
import { useNavigate } from 'react-router';
import { useSettings } from '../features/settings/SettingsProvider';

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
  const { tr } = useSettings();
  const navigate = useNavigate();
  const connection = kind === 'connection';

  return (
    <FailureView
      code={connection ? 'CONNECTION / 01' : 'PAGE / 02'}
      description={connection
        ? tr('无法获取工作区数据。请检查服务状态或连接配置，然后重新连接。', 'Workspace data is unavailable. Check the service or connection settings, then try again.')
        : tr('当前页面遇到了意外问题。可以重试，或先返回概览继续使用。', 'Something went wrong on this page. Try again, or return to the overview.')}
      detail={detail}
      detailLabel={tr('查看错误详情', 'View error details')}
      kind={kind}
      leaveLabel={connection ? tr('连接设置', 'Connection settings') : tr('返回概览', 'Back to overview')}
      onLeave={() => navigate(connection ? '/runtime' : '/')}
      onRetry={onRetry}
      retryLabel={connection ? tr('重新连接', 'Reconnect') : tr('重试页面', 'Retry page')}
      title={connection ? tr('服务暂时无法连接', 'Service unavailable') : tr('页面暂时无法显示', 'Page unavailable')}
    />
  );
}
