/** 评测页面独立管理结果比较、数据集注册及按需加载。 */
import { FormEvent, useEffect, useState } from 'react';
import { evaluationsApi } from '../../shared/api/resources/evaluations';
import { Icon } from '../../app/display';
import { PanelDeck } from '../../app/PanelDeck';
import { errorMessage, formatNumber } from '../../app/formatters';
import { useSettings } from '../settings/SettingsProvider';
import type { DatasetResource, EvaluationResult, EvaluationComparison } from '../../shared/api/types';
import { toast } from '../../stores/toastStore';
/** 挂载时加载评测资源，提交与错误状态仅影响当前页面。 */
export function EvaluationsPage() {
  const { tr } = useSettings();
  const [busy, setBusy] = useState(false);
  const panelLabels = {
    collapse: tr('折叠面板', 'Collapse panel'),
    expand: tr('展开面板', 'Expand panel'),
  };
  const [datasets, setDatasets] = useState<DatasetResource[]>([]);
  const [evaluations, setEvaluations] = useState<EvaluationResult[]>([]);
  const [selectedEvaluations, setSelectedEvaluations] = useState<string[]>([]);
  const [evaluationComparison, setEvaluationComparison] = useState<EvaluationComparison | null>(
    null,
  );
  const [datasetDraft, setDatasetDraft] = useState({
    name: '',
    artifact_uri: '',
    kind: 'custom' as DatasetResource['kind'],
  });
  useEffect(() => {
    let active = true;
    void Promise.all([evaluationsApi.datasets(), evaluationsApi.evaluations()])
      .then(([nextDatasets, nextEvaluations]) => {
        if (active) {
          setDatasets(nextDatasets);
          setEvaluations(nextEvaluations);
        }
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
  /** 注册数据集后重新读取服务器资源列表。 */
  async function registerDataset(event: FormEvent) {
    event.preventDefault();
    if (busy || !datasetDraft.name.trim() || !datasetDraft.artifact_uri.trim()) return;
    setBusy(true);
    try {
      await evaluationsApi.createDataset({
        name: datasetDraft.name.trim(),
        kind: datasetDraft.kind,
        artifact_uri: datasetDraft.artifact_uri.trim(),
      });
      setDatasetDraft({ name: '', artifact_uri: '', kind: 'custom' });
      setDatasets(await evaluationsApi.datasets());
      toast.success(tr('数据集已注册', 'Dataset registered'));
    } catch (cause) {
      toast.error(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }
  /** 比较所选结果并由服务器校验可比性。 */
  async function compareSelectedEvaluations() {
    if (selectedEvaluations.length < 2 || busy) return;
    setBusy(true);
    try {
      setEvaluationComparison(await evaluationsApi.compareEvaluations(selectedEvaluations));
    } catch (cause) {
      toast.error(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <PanelDeck labels={panelLabels} page="lab-evaluations">
        <section className="dashboard-panel evaluation-panel" key="results">
          <div className="panel-heading">
            <div>
              <h2>{tr('评测结果', 'Evaluation results')}</h2>
              <p>
                {tr(
                  '只允许数据集与运行参数一致的结果对比',
                  'Comparison requires matching datasets and execution parameters',
                )}
              </p>
            </div>
            <b>{evaluations.length}</b>
          </div>
          <div className="evaluation-list">
            {evaluations.length === 0 ? (
              <div className="inline-empty">
                {tr(
                  '还没有评测结果。先注册数据集并运行评测任务。',
                  'No evaluation results yet. Register a dataset and run an evaluation job first.',
                )}
              </div>
            ) : (
              evaluations.map((item) => (
                <label key={item.id}>
                  <input
                    checked={selectedEvaluations.includes(item.id)}
                    onChange={(event) =>
                      setSelectedEvaluations((current) =>
                        event.target.checked
                          ? [...current, item.id]
                          : current.filter((id) => id !== item.id),
                      )
                    }
                    type="checkbox"
                  />
                  <div>
                    <strong>{item.model_id}</strong>
                    <small>
                      {item.kind} · {new Date(item.created_at).toLocaleString()}
                    </small>
                  </div>
                  <span>
                    {Object.entries(item.metrics)
                      .filter(([, value]) => typeof value === 'number')
                      .slice(0, 2)
                      .map(([name, value]) => `${name} ${formatNumber(Number(value), 3)}`)
                      .join(' · ')}
                  </span>
                </label>
              ))
            )}
          </div>
          <button
            className="panel-action"
            disabled={busy || selectedEvaluations.length < 2}
            onClick={() => void compareSelectedEvaluations()}
            type="button"
          >
            {tr('对比所选结果', 'Compare selected')}
          </button>
          {evaluationComparison && (
            <div className="comparison-table">
              <header>
                <span>{tr('模型', 'Model')}</span>
                {evaluationComparison.metrics.map((metric) => (
                  <b key={metric}>{metric}</b>
                ))}
              </header>
              {evaluationComparison.rows.map((row) => (
                <div key={row.evaluation.id}>
                  <strong>{row.evaluation.model_id}</strong>
                  {evaluationComparison.metrics.map((metric) => (
                    <span key={metric}>
                      {formatNumber(Number(row.evaluation.metrics[metric]), 4)}
                      <small>
                        {row.deltas[metric] == null
                          ? ''
                          : ` ${Number(row.deltas[metric]) >= 0 ? '+' : ''}${formatNumber(Number(row.deltas[metric]), 4)}`}
                      </small>
                    </span>
                  ))}
                </div>
              ))}
            </div>
          )}
        </section>
        <section className="dashboard-panel dataset-panel" key="datasets">
          <div className="panel-heading">
            <div>
              <h2>{tr('数据集', 'Datasets')}</h2>
              <p>
                {tr('可复现的文件哈希与来源清单', 'Reproducible file hashes and source manifests')}
              </p>
            </div>
            <b>{datasets.length}</b>
          </div>
          <form className="dataset-form" onSubmit={registerDataset}>
            <input
              onChange={(event) =>
                setDatasetDraft((current) => ({ ...current, name: event.target.value }))
              }
              placeholder={tr('名称', 'Name')}
              value={datasetDraft.name}
            />
            <select
              onChange={(event) =>
                setDatasetDraft((current) => ({
                  ...current,
                  kind: event.target.value as DatasetResource['kind'],
                }))
              }
              value={datasetDraft.kind}
            >
              <option value="custom">Custom</option>
              <option value="wikitext2">WikiText-2</option>
            </select>
            <input
              onChange={(event) =>
                setDatasetDraft((current) => ({ ...current, artifact_uri: event.target.value }))
              }
              placeholder="workspace://datasets/corpus.txt"
              value={datasetDraft.artifact_uri}
            />
            <button disabled={busy} type="submit">
              {tr('注册', 'Register')}
            </button>
          </form>
          <div className="dataset-list">
            {datasets.length === 0 ? (
              <div className="inline-empty">
                {tr(
                  '还没有数据集。注册一个文件或工作区资源后即可开始评测。',
                  'No datasets yet. Register a file or workspace resource to start evaluating.',
                )}
              </div>
            ) : (
              datasets.map((item) => (
                <div key={item.id}>
                  <div>
                    <strong>{item.name}</strong>
                    <small>
                      {item.kind} · {formatNumber(item.byte_size / 2 ** 20, 2)} MiB ·{' '}
                      {item.sha256.slice(0, 12)}
                    </small>
                  </div>
                  <button
                    aria-label={tr('删除数据集', 'Delete dataset')}
                    onClick={() =>
                      void evaluationsApi
                        .deleteDataset(item.id)
                        .then(() =>
                          setDatasets((current) => current.filter((entry) => entry.id !== item.id)),
                        )
                        .catch((cause) => toast.error(errorMessage(cause)))
                    }
                    type="button"
                  >
                    <Icon name="trash" size={13} />
                  </button>
                </div>
              ))
            )}
          </div>
        </section>
      </PanelDeck>
    </>
  );
}
