/** 模型仓库页面负责检索、详情与下载任务提交。 */
import { FormEvent, useEffect, useState } from 'react';
import { modelsApi } from '../../shared/api/resources/models';
import { jobsApi } from '../../shared/api/resources/jobs';
import type { HubModelSummary, HubModelInfo, JobKindResource } from '../../shared/api/types';
import { Icon } from '../../app/display';
import { PanelDeck } from '../../app/PanelDeck';
import { errorMessage, formatNumber } from '../../app/formatters';
import { useSettings } from '../settings/SettingsProvider';
import { useRuntime } from '../../app/RuntimeProvider';
import { useNavigate } from 'react-router';
import { parseHubReference } from './hubReference';
import { toast } from '../../stores/toastStore';
/** 隔离模型仓库的请求与草稿，将下载任务登记到共享运行时。 */
export function ModelHubPage() {
  const { tr } = useSettings();
  const [busy, setBusy] = useState(false);
  const panelLabels = {
    collapse: tr('折叠面板', 'Collapse panel'),
    expand: tr('展开面板', 'Expand panel'),
  };
  const { addJob } = useRuntime();
  const navigate = useNavigate();
  const [hubProvider, setHubProvider] = useState<HubModelSummary['provider']>('modelscope');
  const [hubQuery, setHubQuery] = useState('');
  const [hubResults, setHubResults] = useState<HubModelSummary[]>([]);
  const [hubModel, setHubModel] = useState<HubModelInfo | null>(null);
  const [jobKinds, setJobKinds] = useState<JobKindResource[]>([]);
  useEffect(() => {
    let active = true;
    void jobsApi
      .jobKinds()
      .then((items) => {
        if (active) setJobKinds(items);
      })
      .catch((cause) => {
        if (active) {
          toast.error(errorMessage(cause));
        }
      });
    return () => {
      active = false;
    };
  }, []);
  /** 按关键词或仓库链接检索模型。 */
  async function searchHub(event: FormEvent) {
    event.preventDefault();
    const query = hubQuery.trim();
    if (!query || busy) return;
    setBusy(true);
    try {
      const reference = parseHubReference(query, hubProvider);
      if (reference) {
        const info = await modelsApi.hubModelInfo(
          reference.provider,
          reference.repoId,
          reference.revision,
        );
        setHubProvider(reference.provider);
        setHubResults([info]);
        setHubModel(info);
        return;
      }
      const results = await modelsApi.searchHubModels(hubProvider, query);
      setHubResults(results);
      setHubModel(null);
    } catch (cause) {
      toast.error(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }
  /** 读取所选仓库的版本、文件与大小信息。 */
  async function inspectHubModel(item: HubModelSummary) {
    if (busy) return;
    setBusy(true);
    try {
      setHubModel(await modelsApi.hubModelInfo(item.provider, item.repo_id));
    } catch (cause) {
      toast.error(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }
  /** 创建下载任务，并打开该任务的进度页面。 */
  async function downloadHubModel() {
    if (!hubModel || busy) return;
    const name =
      hubModel.repo_id
        .split('/')
        .pop()
        ?.replace(/[^A-Za-z0-9_.-]/g, '-') || 'model';
    const repositoryPath = hubModel.repo_id
        .split('/')
        .map((part) => part.replace(/[^A-Za-z0-9_.-]+/g, '-') || 'model')
        .join('/');
    setBusy(true);
    try {
      const created = await jobsApi.createJob(`download.${hubModel.provider}`, {
        repo_id: hubModel.repo_id,
        destination: `models/${hubModel.provider}/${repositoryPath || name}`,
        revision: hubModel.revision,
        expected_bytes: hubModel.total_bytes || null,
      });
      addJob(created);
      toast.success(tr('下载任务已提交', 'Download job submitted'));
      navigate('/quantization', { state: { jobId: created.id } });
    } catch (cause) {
      toast.error(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <PanelDeck labels={panelLabels} page="lab-models">
        <div key="hubs">
          <section className="dashboard-panel hub-panel">
            <div className="panel-heading">
              <div>
                <h2>{tr('模型仓库', 'Model hubs')}</h2>
                <p>
                  {tr(
                    '搜索、粘贴仓库链接并启动可续传下载',
                    'Search or paste a repository link to start a resumable download',
                  )}
                </p>
              </div>
            </div>
            <form aria-busy={busy} className="hub-search" onSubmit={searchHub}>
              <select
                onChange={(event) =>
                  setHubProvider(event.target.value as HubModelSummary['provider'])
                }
                value={hubProvider}
              >
                <option value="modelscope">ModelScope</option>
                <option value="huggingface">Hugging Face</option>
              </select>
              <input
                onChange={(event) => setHubQuery(event.target.value)}
                placeholder={tr('模型名称、仓库或链接', 'Model, repository, or URL')}
                value={hubQuery}
              />
              <button disabled={busy || !hubQuery.trim()} type="submit">
                {busy ? tr('搜索中…', 'Searching…') : tr('查找', 'Find')}
              </button>
            </form>
            {hubQuery.trim() && hubResults.length === 0 && !hubModel && (
              <div className="inline-empty hub-empty">
                {tr(
                  '没有匹配的模型仓库。检查名称、组织名或仓库链接。',
                  'No model hub matches. Check the model name, organization, or repository URL.',
                )}
              </div>
            )}
            {hubResults.length > 0 && (
              <div className="hub-results">
                {hubResults.map((item) => (
                  <button
                    className={hubModel?.repo_id === item.repo_id ? 'active' : ''}
                    key={`${item.provider}:${item.repo_id}`}
                    onClick={() => void inspectHubModel(item)}
                    type="button"
                  >
                    <div>
                      <strong>{item.repo_id}</strong>
                      <small>
                        {formatNumber(item.downloads)} downloads · {formatNumber(item.likes)} likes
                      </small>
                    </div>
                    <span>
                      {item.total_bytes
                        ? `${formatNumber(item.total_bytes / 2 ** 30, 1)} GB`
                        : '--'}
                    </span>
                  </button>
                ))}
              </div>
            )}
            {hubModel && (
              <div className="hub-detail">
                <div>
                  <strong>{hubModel.repo_id}</strong>
                  <small>
                    {hubModel.revision} · {hubModel.files.length} files ·{' '}
                    {formatNumber(hubModel.total_bytes / 2 ** 30, 2)} GB
                  </small>
                </div>
                <button
                  disabled={
                    busy || !jobKinds.some((item) => item.kind === `download.${hubModel.provider}`)
                  }
                  onClick={() => void downloadHubModel()}
                  type="button"
                >
                  <Icon name="download" size={14} />
                  {tr('下载', 'Download')}
                </button>
              </div>
            )}
          </section>
        </div>
      </PanelDeck>
    </>
  );
}
