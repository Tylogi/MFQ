/** 本地检查点目录列表及模型加载入口。 */
import { useSettings } from '../settings/SettingsProvider';
import { Icon, SectionLabel, TMPanel, EmptyPanel } from '../../app/display';
import { formatNumber } from '../../app/formatters';
import type { useModelCatalog } from './useModelCatalog';

/** 展示已登记检查点及加载能力。 */
export function LocalCheckpoints({ catalog }: { catalog: ReturnType<typeof useModelCatalog> }) {
  const { tr } = useSettings();
  const { runtime, artifacts, busy, instances, modelFilter,
    filteredArtifacts, unloadInstance, loadArtifact, chooseModelDirectory } = catalog;
  return (
    <>
      <SectionLabel
        title={tr('本地检查点', 'Local checkpoints')}
        subtitle={`${artifacts.length} ${tr('个本地模型', 'local models')}`}
      />
      {filteredArtifacts.length > 0 ? (
        <TMPanel className="model-catalog-panel model-library-panel">
          <div className="model-list">
            {filteredArtifacts.map((item) => {
              const instance = instances.find(
                (candidate) => candidate.model === item.name && candidate.state !== 'failed',
              );
              const loaded = Boolean(instance) || item.name === runtime?.model;
              const policy = instance?.pinned
                ? tr('固定', 'Pinned')
                : instance?.idle_ttl_seconds != null
                  ? `TTL ${instance.idle_ttl_seconds}s` : null;
              return (
                <div className="model-row" key={item.id}>
                  <span className={loaded ? 'model-state active'
                    : item.loadable ? 'model-state' : 'model-state failed'} />
                  <div>
                    <strong>{item.name}</strong>
                    <small>
                      {item.architecture} · {item.shard_count} shards ·{' '}
                      {formatNumber(item.total_bytes / 2 ** 30, 1)} GB{policy ? ` · ${policy}` : ''}
                    </small>
                  </div>
                  {instance ? (
                    <button disabled={busy || instance.state !== 'ready'}
                      onClick={() => void unloadInstance(instance.id)} type="button">
                      {tr('卸载', 'Unload')}
                    </button>
                  ) : loaded ? (
                    <em>{tr('已加载', 'Loaded')}</em>
                  ) : !item.loadable ? (
                    <em className="failed" title={item.error || undefined}>
                      {item.complete && item.format === 'hf'
                        ? tr('需先转换', 'Convert first') : tr('不可用', 'Invalid')}
                    </em>
                  ) : (
                    <button disabled={busy} onClick={() => void loadArtifact(item.name)} type="button">
                      {tr('加载', 'Load')}
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        </TMPanel>
      ) : (
        <EmptyPanel
          icon="folder"
          title={tr(modelFilter ? '没有匹配的本地模型' : '还没有本地模型',
            modelFilter ? 'No local model matches' : 'No local models yet')}
          message={tr('添加一个模型文件夹即可开始。', 'Add a model folder to get started.')}
          action={
            <button className="primary screen-header-action" disabled={busy}
              onClick={() => void chooseModelDirectory()} type="button">
              <Icon name="folder" size={14} />{tr('选择模型文件夹', 'Choose model folder')}
            </button>
          }
        />
      )}
    </>
  );
}
