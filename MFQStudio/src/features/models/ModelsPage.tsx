import { useSettings } from '../settings/SettingsProvider';
import { Icon, ScreenHeader } from '../../app/display';
import { useModelCatalog } from './useModelCatalog';
import { ModelDirectoryDialog } from './ModelDirectoryDialog';
import { LoadedModels } from './LoadedModels';
import { ModelLoadPolicy } from './ModelLoadPolicy';
import { LocalCheckpoints } from './LocalCheckpoints';
import { formatBytes } from '../../app/formatters';

export function ModelsPage() {
  const catalog = useModelCatalog();
  const { tr } = useSettings();
  const { artifacts, busy, availableModelNames,
    openStudioPage, chooseModelDirectory, modelFolderPath } = catalog;
  const totalBytes = artifacts.reduce((sum, artifact) => sum + artifact.total_bytes, 0);
  return (
    <section className="dashboard-view">
      <ScreenHeader
        title={tr('模型', 'Models')}
        subtitle={tr('管理本地模型与运行实例。', 'Manage local models and runtime instances.')}
        trailing={
          <>
            <button onClick={() => openStudioPage('lab', 'models')} type="button">
              <Icon name="download" size={14} />{tr('模型下载', 'Model downloads')}
            </button>
            <button className="primary" disabled={busy}
              onClick={() => void chooseModelDirectory()} type="button">
              <Icon name="folder" size={14} />{tr('添加模型', 'Add model')}
            </button>
          </>
        }
      />
      <div className="model-folder-location">
        <Icon name="folder" />
        <span>{tr('模型文件夹', 'Model folder')}</span>
        <code title={modelFolderPath}>{modelFolderPath}</code>
        <button disabled={busy} onClick={() => void chooseModelDirectory()} type="button">
          {tr('更改', 'Change')}
        </button>
      </div>
      <div className="model-workbench-summary">
        <div>
          <span>{tr('运行中的模型', 'Loaded models')}</span>
          <strong>{availableModelNames.length}</strong>
          <small>{tr('可直接用于服务', 'Ready for serving')}</small>
        </div>
        <div>
          <span>{tr('本地检查点', 'Local checkpoints')}</span>
          <strong>{artifacts.length}</strong>
          <small>{tr('已登记到 MFQ', 'Registered in MFQ')}</small>
        </div>
        <div>
          <span>{tr('注册模型资产总大小', 'Registered model assets size')}</span>
          <strong>{totalBytes === 0 ? '0 B' : formatBytes(totalBytes).replace(/\b(KB|MB|GB|TB)\b/g, (unit) => `${unit[0]}iB`)}</strong>
          <small>{tr('已登记模型的文件总大小', 'Total file size of registered models')}</small>
        </div>
      </div>
      <div className="model-catalog-toolbar">
        <div>
          <h2>{tr('模型资产', 'Model assets')}</h2>
          <span>{tr('管理本地资产与已载入模型', 'Manage local assets and loaded models')}</span>
        </div>
      </div>
      <LoadedModels catalog={catalog} />
      <ModelLoadPolicy catalog={catalog} />
      <LocalCheckpoints catalog={catalog} />
      <div className="model-workbench-links">
        <span>{tr('自定义模型精度', 'Custom model precision')}</span>
        <button onClick={() => openStudioPage('lab', 'quantization')} type="button">
          {tr('打开量化工作台', 'Open quantization workspace')}
          <Icon name="text-forward" size={14} />
        </button>
      </div>
      <ModelDirectoryDialog catalog={catalog} />
    </section>
  );
}
