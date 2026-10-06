/** Display registered datasets and evaluation results, refreshing when evaluation jobs finish. */
import { i18n } from '../../i18n';
import { localized } from '../../i18n/messages';
import { useTranslation } from 'react-i18next';
import { FormEvent, useEffect, useState } from 'react';
import { evaluationsApi } from '../../shared/api/resources/evaluations';
import { Icon } from '../../app/display';
import { PanelDeck } from '../../app/PanelDeck';
import { errorMessage, formatNumber } from '../../app/formatters';
import type { DatasetResource, EvaluationResult, EvaluationComparison } from '../../shared/api/types';
import { toast } from '../../stores/toastStore';
import { ModelVendorMark } from '../../app/ModelVendorMark';
import { useJobStore } from '../../stores/jobStore';

/** Render evaluation results and reload them when a relevant background job reaches a terminal state. */
export function EvaluationsPage() {
  const { t } = useTranslation();
  const [busy, setBusy] = useState(false);
  const panelLabels = {
    collapse: t('evaluations:evaluationsPage.collapsePanel'),
    expand: t('evaluations:evaluationsPage.expandPanel'),
  };
  const [datasets, setDatasets] = useState<DatasetResource[]>([]);
  const [evaluations, setEvaluations] = useState<EvaluationResult[]>([]);
  const finishedEvaluationJobs = useJobStore((state) => state.jobs
    .filter((job) =>
      (job.kind === 'evaluate.perplexity' || job.kind === 'benchmark.kernel') &&
      ['succeeded', 'failed', 'cancelled', 'interrupted'].includes(job.status))
    .map((job) => `${job.id}:${job.status}`)
    .sort()
    .join(','));
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
  }, [finishedEvaluationJobs]);
  /** Register a dataset and reload the dataset list after the server accepts it. */
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
      toast.success(localized('evaluations:evaluationsPage.datasetRegistered'));
    } catch (cause) {
      toast.error(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }
  /** Compare the selected evaluation records using the server's comparison contract. */
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
              <h2>{t('evaluations:evaluationsPage.evaluationResults')}</h2>
              <p>
                {t('evaluations:evaluationsPage.comparisonRequiresMatchingDatasetsAndExecutionParameters')}
              </p>
            </div>
            <b>{evaluations.length}</b>
          </div>
          <div className="evaluation-list">
            {evaluations.length === 0 ? (
              <div className="inline-empty">
                {t('evaluations:evaluationsPage.noEvaluationResultsYetRegisterADatasetAndRunAnEvaluationJob')}
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
                      {item.kind} · {new Date(item.created_at).toLocaleString(i18n.resolvedLanguage)}
                    </small>
                  </div>
                  <span className="model-identity-trailing">
                    {Object.entries(item.metrics)
                      .filter(([, value]) => typeof value === 'number')
                      .slice(0, 2)
                      .map(([name, value]) => `${name} ${formatNumber(Number(value), 3)}`)
                      .join(' · ')}
                    <ModelVendorMark name={item.model_id} />
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
            {t('evaluations:evaluationsPage.compareSelected')}
          </button>
          {evaluationComparison && (
            <div className="comparison-table">
              <header>
                <span>{t('evaluations:evaluationsPage.model')}</span>
                {evaluationComparison.metrics.map((metric) => (
                  <b key={metric}>{metric}</b>
                ))}
              </header>
              {evaluationComparison.rows.map((row) => (
                <div key={row.evaluation.id}>
                  <strong className="model-identity-label">{row.evaluation.model_id}<ModelVendorMark name={row.evaluation.model_id} size={20} /></strong>
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
              <h2>{t('evaluations:evaluationsPage.datasets')}</h2>
              <p>
                {t('evaluations:evaluationsPage.reproducibleFileHashesAndSourceManifests')}
              </p>
            </div>
            <b>{datasets.length}</b>
          </div>
          <form className="dataset-form" onSubmit={registerDataset}>
            <input
              onChange={(event) =>
                setDatasetDraft((current) => ({ ...current, name: event.target.value }))
              }
              placeholder={t('evaluations:evaluationsPage.name')}
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
              {t('evaluations:evaluationsPage.register')}
            </button>
          </form>
          <div className="dataset-list">
            {datasets.length === 0 ? (
              <div className="inline-empty">
                {t('evaluations:evaluationsPage.noDatasetsYetRegisterAFileOrWorkspaceResourceToStartEvaluating')}
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
                    aria-label={t('evaluations:evaluationsPage.deleteDataset')}
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
