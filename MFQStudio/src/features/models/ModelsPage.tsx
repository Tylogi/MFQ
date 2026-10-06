/** Compose loaded models, local checkpoints, and their lifecycle controls. */
import { useTranslation } from 'react-i18next';
import { Icon, ScreenHeader } from '../../app/display';
import { useModelCatalog } from './useModelCatalog';
import { ModelDirectoryDialog } from './ModelDirectoryDialog';
import { LoadedModels } from './LoadedModels';
import { ModelLoadPolicy } from './ModelLoadPolicy';
import { LocalCheckpoints } from './LocalCheckpoints';
import { formatBytes } from '../../app/formatters';

/** Compose loaded models, local checkpoints, and their lifecycle controls. */
export function ModelsPage() {
  const catalog = useModelCatalog();
  const { t } = useTranslation();
  const { artifacts, busy, availableModelNames, modelFilter, setModelFilter,
    openStudioPage, chooseModelDirectory } = catalog;
  const totalBytes = artifacts.reduce((sum, artifact) => sum + artifact.total_bytes, 0);
  return (
    <section className="dashboard-view">
      <ScreenHeader
        title={t('models:modelsPage.models')}
        subtitle={t('models:modelsPage.manageLocalModelsAndRuntimeInstances')}
        trailing={
          <>
            <button onClick={() => openStudioPage('lab', 'models')} type="button">
              <Icon name="download" size={14} />{t('models:modelsPage.modelDownloads')}
            </button>
            <button className="primary" disabled={busy}
              onClick={() => void chooseModelDirectory()} type="button">
              <Icon name="folder" size={14} />{t('models:modelsPage.addModel')}
            </button>
          </>
        }
      />
      <div className="model-workbench-summary">
        <div>
          <span>{t('models:modelsPage.loadedModels')}</span>
          <strong>{availableModelNames.length}</strong>
          <small>{t('models:modelsPage.readyForServing')}</small>
        </div>
        <div>
          <span>{t('models:modelsPage.localCheckpoints')}</span>
          <strong>{artifacts.length}</strong>
          <small>{t('models:modelsPage.registeredInMfq')}</small>
        </div>
        <div>
          <span>{t('models:modelsPage.registeredModelAssetsSize')}</span>
          <strong>{totalBytes === 0 ? '0 B' : formatBytes(totalBytes).replace(/\b(KB|MB|GB|TB)\b/g, (unit) => `${unit[0]}iB`)}</strong>
          <small>{t('models:modelsPage.totalFileSizeOfRegisteredModels')}</small>
        </div>
      </div>
      <div className="model-catalog-toolbar">
        <div>
          <h2>{t('models:modelsPage.modelAssets')}</h2>
          <span>{t('models:modelsPage.manageLocalAssetsAndLoadedModels')}</span>
        </div>
        <label>
          <span aria-hidden="true">/</span>
          <input aria-label={t('models:modelsPage.filterModels')}
            onChange={(event) => setModelFilter(event.target.value)}
            placeholder={t('models:modelsPage.filterByName')}
            value={modelFilter} />
        </label>
      </div>
      <LoadedModels catalog={catalog} />
      <ModelLoadPolicy catalog={catalog} />
      <LocalCheckpoints catalog={catalog} />
      <div className="model-workbench-links">
        <span>{t('models:modelsPage.customModelPrecision')}</span>
        <button onClick={() => openStudioPage('lab', 'quantization')} type="button">
          {t('models:modelsPage.openQuantizationWorkspace')}
          <Icon name="text-forward" size={14} />
        </button>
      </div>
      <ModelDirectoryDialog catalog={catalog} />
    </section>
  );
}
