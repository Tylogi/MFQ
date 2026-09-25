/** 为应用启动、路由切换和服务连接提供一致的工作区加载页面。 */
import { useSettings } from '../features/settings/SettingsProvider';

/** 显示统一的加载状态，动效不表示实际完成比例。 */
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
