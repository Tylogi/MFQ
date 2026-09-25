/** 统一展示工作区阻断性故障，并提供重试及离开当前页面的操作。 */
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

/** 不依赖应用上下文地渲染失败页，使顶层异常边界也能安全使用。 */
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

/** 在主工作区显示可恢复的失败页面，连接故障允许前往服务器设置。 */
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
