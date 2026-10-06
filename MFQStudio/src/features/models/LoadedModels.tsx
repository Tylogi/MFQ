/** Render model availability and keep unload progress visible until the operation completes. */
import { useSettings } from '../settings/SettingsProvider';
import { Icon, SectionLabel, TMPanel, EmptyPanel } from '../../app/display';
import { formatNumber } from '../../app/formatters';
import type { useModelCatalog } from './useModelCatalog';
import { ModelVendorMark } from '../../app/ModelVendorMark';
import { ModelLoadProgress } from './ModelLoadProgress';

/** Display model actions with instance-specific unload feedback and duplicate-action protection. */
export function LoadedModels({ catalog }: { catalog: ReturnType<typeof useModelCatalog> }) {
  const { tr } = useSettings();
  const { busy, artifacts, availableModelNames, modelFilter, filteredInstances,
    unloadInstance, unloadingInstanceIds, chooseModelDirectory } = catalog;
  return (
    <>
      <SectionLabel
        title={tr('已加载模型', 'Loaded models')}
        subtitle={tr(`${availableModelNames.length} 个可用于推理`,
          `${availableModelNames.length} available for inference`)}
      />
      {filteredInstances.length > 0 ? (
        <TMPanel className="model-catalog-panel loaded-model-panel">
          <div className="model-list">
            {filteredInstances.map((instance) => {
              const unloading = instance.state === 'unloading' || unloadingInstanceIds.has(instance.id);
              const ready = !unloading && (instance.state === 'ready' || instance.state === 'busy');
              const stateLabel = unloading
                ? tr('正在卸载', 'Unloading')
                : instance.state === 'loading'
                  ? tr('加载中', 'Loading')
                  : instance.state === 'failed'
                    ? tr('失败', 'Failed')
                    : instance.state === 'busy'
                      ? tr('使用中', 'Busy')
                      : tr('就绪', 'Ready');
              return (
                <div className="model-row" key={instance.id}>
                  <span className={instance.state === 'failed'
                    ? 'model-state failed' : ready ? 'model-state active' : 'model-state'} />
                  <div>
                    <strong>{instance.model}</strong>
                    <small>
                      {stateLabel} · {formatNumber(instance.context_size)} ctx
                      {instance.pinned
                        ? ` · ${tr('固定', 'Pinned')}`
                        : instance.idle_ttl_seconds != null
                          ? ` · TTL ${instance.idle_ttl_seconds}s` : ''}
                    </small>
                    {instance.state === 'loading' && <ModelLoadProgress model={instance.model} />}
                  </div>
                  <div className="model-row-actions">
                    <ModelVendorMark name={instance.model} architecture={artifacts.find((item) => item.name === instance.model)?.architecture} />
                    <button
                      disabled={busy || unloading || instance.state !== 'ready'}
                      onClick={() => void unloadInstance(instance.id)}
                      type="button"
                    >
                      {unloading ? tr('正在卸载…', 'Unloading…') : tr('卸载', 'Unload')}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </TMPanel>
      ) : (
        <EmptyPanel
          icon="memory"
          title={tr(modelFilter ? '没有匹配的已加载模型' : '当前没有已加载模型',
            modelFilter ? 'No loaded model matches' : 'No models loaded')}
          message={tr('从本地检查点加载一个模型后即可开始对话。',
            'Load a local checkpoint to start chatting.')}
          action={
            <button className="primary screen-header-action" disabled={busy}
              onClick={() => void chooseModelDirectory()} type="button">
              <Icon name="folder" size={14} />{tr('添加模型', 'Add model')}
            </button>
          }
        />
      )}
    </>
  );
}
