/** Display calibration import and artifact selection for the quantization workspace. */
import { useQuantization } from './QuantizationContext';
import { Icon } from '../../app/display';
/** Read the data and business actions required by this panel from page state. */
export function ImatrixPanel() {
  const {
    t,
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
              {t('jobs:imatrixPanel.calibrateSeparatelyImportOneOrCollectItBeforeQuantization')}
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
            {t('jobs:imatrixPanel.newCalibration')}
          </button>
          <button
            disabled={
              busy || imatrixImporting || !jobKinds.some((item) => item.kind === 'artifact.import')
            }
            onClick={() => imatrixInputRef.current?.click()}
            type="button"
          >
            <Icon name="upload" size={11} />
            {imatrixImporting ? t('jobs:imatrixPanel.importing') : t('jobs:imatrixPanel.importFile')}
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
                <span>{t('jobs:imatrixPanel.use')}</span>
              </button>
            ))}
          </div>
        )}
      </section>
    </>
  );
}
