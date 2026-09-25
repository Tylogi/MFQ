/** 已加载模型列表及切换、卸载入口。 */
import { useSettings } from '../settings/SettingsProvider';
import { Icon, SectionLabel, TMPanel, EmptyPanel } from '../../app/display';
import { formatNumber } from '../../app/formatters';
import type { useModelCatalog } from './useModelCatalog';

/** 展示当前实例及模型选择操作。 */
export function LoadedModels({ catalog }: { catalog: ReturnType<typeof useModelCatalog> }) {
  const { tr } = useSettings();
  const { model, busy, availableModelNames, modelFilter, filteredInstances,
    selectModel, unloadInstance, chooseModelDirectory } = catalog;
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
              const ready = instance.state === 'ready' || instance.state === 'busy';
              const selected = instance.model === model;
              const stateLabel = instance.state === 'loading'
                ? tr('加载中', 'Loading')
                : instance.state === 'unloading'
                  ? tr('卸载中', 'Unloading')
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
                  </div>
                  <div className="model-row-actions">
                    {ready && (
                      <button
                        className={selected ? 'selected' : ''}
                        disabled={busy || selected}
                        onClick={() => selectModel(instance.model)}
                        type="button"
                      >
                        {selected ? tr('当前', 'Current') : tr('用于对话', 'Use in chat')}
                      </button>
                    )}
                    <button
                      disabled={busy || instance.state !== 'ready'}
                      onClick={() => void unloadInstance(instance.id)}
                      type="button"
                    >
                      {tr('卸载', 'Unload')}
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
