/** 展示量化工作台的产物来源与验证记录。 */
import { useQuantization } from './QuantizationContext';
/** 从页面状态读取本面板所需数据与业务操作。 */
export function LineagePanel() {
  const { tr, lineage } = useQuantization();
  return (
    <>
      <section className="dashboard-panel lineage-panel" key="lineage">
        <div className="panel-heading">
          <div>
            <h2>{tr('产物谱系', 'Artifact lineage')}</h2>
            <p>
              {tr(
                '源产物、生成任务、默认后参数和验证记录',
                'Sources, producing jobs, resolved parameters, and validations',
              )}
            </p>
          </div>
          <b>{lineage.length}</b>
        </div>
        <div className="lineage-list">
          {lineage.length === 0 ? (
            <div className="inline-empty">
              {tr(
                '暂无产物谱系记录。运行量化或导入任务后会显示在这里。',
                'No artifact lineage yet. Run a quantization or import job to populate this view.',
              )}
            </div>
          ) : (
            lineage.slice(0, 20).map((item) => (
              <details key={item.id}>
                <summary>
                  <div>
                    <strong>{item.artifact_name}</strong>
                    <small>
                      {item.producer_kind} · {new Date(item.created_at).toLocaleString()}
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
                    <dt>{tr('源', 'Sources')}</dt>
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
