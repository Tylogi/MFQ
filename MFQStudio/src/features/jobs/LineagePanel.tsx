/** Display artifact lineage and verification records for the quantization workspace. */
import { i18n } from '../../i18n';
import { useQuantization } from './QuantizationContext';
/** Read the data and business actions required by this panel from page state. */
export function LineagePanel() {
  const { t, lineage } = useQuantization();
  return (
    <>
      <section className="dashboard-panel lineage-panel" key="lineage">
        <div className="panel-heading">
          <div>
            <h2>{t('jobs:lineagePanel.artifactLineage')}</h2>
            <p>
              {t('jobs:lineagePanel.sourcesProducingJobsResolvedParametersAndValidations')}
            </p>
          </div>
          <b>{lineage.length}</b>
        </div>
        <div className="lineage-list">
          {lineage.length === 0 ? (
            <div className="inline-empty">
              {t('jobs:lineagePanel.noArtifactLineageYetRunAQuantizationOrImportJobToPopulate')}
            </div>
          ) : (
            lineage.slice(0, 20).map((item) => (
              <details key={item.id}>
                <summary>
                  <div>
                    <strong>{item.artifact_name}</strong>
                    <small>
                      {item.producer_kind} · {new Date(item.created_at).toLocaleString(i18n.resolvedLanguage)}
                    </small>
                  </div>
                  <span>{item.validation_job_ids.length} checks</span>
                </summary>
                <dl>
                  <div>
                    <dt>URI</dt>
                    <dd>{item.artifact_uri}</dd>
                  </div>
                  <div>
                    <dt>{t('jobs:lineagePanel.sources')}</dt>
                    <dd>{item.source_uris.join(', ') || '--'}</dd>
                  </div>
                </dl>
                <pre>{JSON.stringify(item.parameters, null, 2)}</pre>
              </details>
            ))
          )}
        </div>
      </section>
    </>
  );
}
