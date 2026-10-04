/** Display calibration import and artifact selection for the quantization workspace. */
import { useQuantization } from './QuantizationContext';
import { Icon } from '../../app/display';
/** Read the data and business actions required by this panel from page state. */
export function ImatrixPanel() {
  const {
    tr,
    busy,
    jobKinds,
    setSelectedJobKind,
    imatrixImporting,
    setPendingImatrix,
    imatrixInputRef,
    imatrixArtifacts,
    importImatrix,
  } = useQuantization();
  return (
    <>
      <section className="dashboard-panel imatrix-panel" key="imatrix">
        <div className="panel-heading">
          <div>
            <h2>Imatrix</h2>
            <p>
              {tr(
                '单独校准、导入，或在量化任务中先校准再使用',
                'Calibrate separately, import one, or collect it before quantization',
              )}
            </p>
          </div>
          <b>{imatrixArtifacts.length}</b>
        </div>
        <div className="imatrix-actions">
          <button
            disabled={busy || !jobKinds.some((item) => item.kind === 'calibrate.imatrix')}
            onClick={() => setSelectedJobKind('calibrate.imatrix')}
            type="button"
          >
            <Icon name="plus" size={11} />
            {tr('新建校准', 'New calibration')}
          </button>
          <button
            disabled={
              busy || imatrixImporting || !jobKinds.some((item) => item.kind === 'artifact.import')
            }
            onClick={() => imatrixInputRef.current?.click()}
            type="button"
          >
            <Icon name="upload" size={11} />
            {imatrixImporting ? tr('正在导入', 'Importing') : tr('导入文件', 'Import file')}
          </button>
          <input
            accept=".imatrix,.gguf,.dat,application/x-mfq-imatrix,application/octet-stream"
            hidden
            onChange={(event) => void importImatrix(event.target.files)}
            ref={imatrixInputRef}
            type="file"
          />
        </div>
        {imatrixArtifacts.length > 0 && (
          <div className="imatrix-list">
            {imatrixArtifacts.slice(0, 12).map((item) => (
              <button
                key={item.id}
                onClick={() => {
                  setPendingImatrix(item.artifact_uri.replace(/^workspace:\/\//, ''));
                  setSelectedJobKind('model.quantize');
                }}
                type="button"
              >
                <div>
                  <strong>{item.artifact_name}</strong>
                  <small>{item.artifact_uri}</small>
                </div>
                <span>{tr('用于量化', 'Use')}</span>
              </button>
            ))}
          </div>
        )}
      </section>
    </>
  );
}
