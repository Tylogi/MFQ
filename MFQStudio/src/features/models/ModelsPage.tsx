/** 模型路由页面组合目录摘要、实例、加载策略及本地检查点。 */
import { useSettings } from '../settings/SettingsProvider';
import { Icon, ScreenHeader } from '../../app/display';
import { useModelCatalog } from './useModelCatalog';
import { ModelDirectoryDialog } from './ModelDirectoryDialog';
import { LoadedModels } from './LoadedModels';
import { ModelLoadPolicy } from './ModelLoadPolicy';
import { LocalCheckpoints } from './LocalCheckpoints';

/** 仅在模型页请求目录数据，并组合各展示面板。 */
export function ModelsPage() {
  const catalog = useModelCatalog();
  const { tr } = useSettings();
  const { model, artifacts, busy, availableModelNames, modelFilter, setModelFilter,
    openStudioPage, chooseModelDirectory } = catalog;
  return (
    <section className="dashboard-view">
      <ScreenHeader
        title={tr('模型', 'Models')}
        subtitle={tr('管理本地模型与运行实例。', 'Manage local models and runtime instances.')}
        trailing={
          <button className="primary" disabled={busy}
            onClick={() => void chooseModelDirectory()} type="button">
            <Icon name="folder" size={14} />{tr('添加模型', 'Add model')}
          </button>
        }
      />
      <div className="model-workbench-summary">
        <div>
          <span>{tr('运行中的模型', 'Loaded models')}</span>
          <strong>{availableModelNames.length}</strong>
          <small>{tr('可直接用于对话', 'Ready for chat')}</small>
        </div>
        <div>
          <span>{tr('本地检查点', 'Local checkpoints')}</span>
          <strong>{artifacts.length}</strong>
          <small>{tr('已登记到 MFQ', 'Registered in MFQ')}</small>
        </div>
        <div>
          <span>{tr('当前对话模型', 'Chat model')}</span>
          <strong title={model || undefined}>{model || tr('未选择', 'None')}</strong>
          <small>{model
            ? tr('切换会话模型不会重新注册资产', 'Switching keeps the registered asset')
            : tr('加载后从这里选择', 'Choose one after loading')}</small>
        </div>
        <div className="model-workbench-links">
          <button onClick={() => openStudioPage('lab', 'models')} type="button">
            <Icon name="download" size={13} />{tr('打开模型仓库', 'Open model hub')}
          </button>
          <button onClick={() => openStudioPage('lab', 'quantization')} type="button">
            <Icon name="memory" size={13} />{tr('去量化', 'Quantize')}
          </button>
        </div>
      </div>
      <div className="model-catalog-toolbar">
        <div>
          <h2>{tr('模型资产', 'Model assets')}</h2>
          <span>{tr('注册、加载和切换对话模型', 'Register, load, and switch chat models')}</span>
        </div>
        <label>
          <span aria-hidden="true">/</span>
          <input aria-label={tr('筛选模型', 'Filter models')}
            onChange={(event) => setModelFilter(event.target.value)}
            placeholder={tr('按名称筛选', 'Filter by name')}
            value={modelFilter} />
        </label>
      </div>
      <LoadedModels catalog={catalog} />
      <ModelLoadPolicy catalog={catalog} />
      <LocalCheckpoints catalog={catalog} />
      <ModelDirectoryDialog catalog={catalog} />
    </section>
  );
}
