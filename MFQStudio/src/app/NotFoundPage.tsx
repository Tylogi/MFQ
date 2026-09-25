/** 未注册地址的 404 页面，保留工作区导航并提供明确的返回入口。 */
import { ArrowLeftIcon, HouseIcon } from '@phosphor-icons/react';
import { useNavigate } from 'react-router';
import { useSettings } from '../features/settings/SettingsProvider';

/** 在未知路由展示独立页面，不重定向或改写用户输入的地址。 */
export function NotFoundPage() {
  const navigate = useNavigate();
  const { tr } = useSettings();

  return (
    <section aria-labelledby="not-found-title" className="failure-page not-found-page">
      <div className="failure-page-content">
        <div className="failure-page-heading">
          <span className="failure-page-code">404 / NOT FOUND</span>
        </div>
        <h1 id="not-found-title">{tr('页面不存在', 'Page not found')}</h1>
        <p className="failure-page-description">
          {tr('这个地址没有对应的页面。请检查网址，或返回概览继续使用。', 'There is no page at this address. Check the URL or return to the overview.')}
        </p>
        <div className="failure-page-actions">
          <button className="failure-page-primary" onClick={() => navigate('/')} type="button">
            <HouseIcon size={16} />
            {tr('返回概览', 'Back to overview')}
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
            {tr('返回上一页', 'Go back')}
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
