import { useSettings } from '../settings/SettingsProvider';
import { Icon, SectionLabel, TMPanel, EmptyPanel } from '../../app/display';
import type { useModelCatalog } from './useModelCatalog';
import { ModelVendorMark } from '../../app/ModelVendorMark';
import { ModelLoadProgress } from './ModelLoadProgress';
import { formatNumber } from '../../app/formatters';
import { ModelMemoryPressure } from './ModelMemoryPressure';

export function LocalCheckpoints({ catalog }: { catalog: ReturnType<typeof useModelCatalog> }) {
  const { tr } = useSettings();
  const { runtime, artifacts, busy, instances, modelFilter,
    filteredArtifacts, unloadInstance, loadArtifact, chooseModelDirectory, openModelFiles } = catalog;
  const available = runtime?.runtime_memory_headroom_bytes;
  const gib = (bytes?: number | null) => bytes == null ? '—' : `${formatNumber(bytes / 2 ** 30, 1)} GiB`;
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
              return (
                <div className="model-row" key={item.id}>
                  <span className={loaded ? 'model-state active'
                    : item.loadable ? 'model-state' : 'model-state failed'} />
                  <div>
                    <strong>{item.name}</strong>
                    <small>{gib(item.total_bytes)} · {item.missing_shards
                      ? tr(`${item.shard_count} 个分片，缺 ${item.missing_shards} 片`, `${item.shard_count} shards, ${item.missing_shards} missing`)
                      : tr(`${item.shard_count} 个分片`, `${item.shard_count} shards`)}</small>
                    <small>{tr('预计常驻内存', 'Estimated resident memory')} {gib(item.estimated_resident_weight_bytes)}
                      {(item.ssd_ple_bytes ?? 0) > 0 && ` · SSD PLE ${gib(item.ssd_ple_bytes)}`}</small>
                    <ModelLoadProgress model={item.name} />
                  </div>
                  <div className="local-checkpoint-memory">
                    <small>{tr('剩余可用内存', 'Remaining available memory')} {gib(available)}</small>
                    <ModelMemoryPressure required={item.estimated_resident_weight_bytes} available={available}
                      label={tr('预计占剩余可用内存', 'Estimated share of remaining available memory')}
                      emptyLabel={tr('无可用内存', 'No available memory')} />
                  </div>
                  <div className="model-row-actions"><ModelVendorMark name={item.name} architecture={item.architecture} /><button className="model-files-action" disabled={busy} onClick={() => void openModelFiles(item.id)} type="button"><Icon name="folder" size={13} />{tr('模型文件', 'Model files')}</button>{instance ? (
                    <button disabled={busy || instance.state !== 'ready'}
                      onClick={() => void unloadInstance(instance.id)} type="button">
                      {tr('卸载', 'Unload')}
                    </button>
                  ) : loaded ? (
                    <em>{tr('已加载', 'Loaded')}</em>
                  ) : !item.loadable ? (
                    <em className="failed" title={item.error || undefined}>
                      {item.complete && item.format === 'hf'
                        ? tr('需先转换', 'Convert first') : item.missing_shards
                          ? tr('分片不全', 'Incomplete shards') : tr('不可用', 'Invalid')}
                    </em>
                  ) : (
                    <button disabled={busy} onClick={() => void loadArtifact(item.name)} type="button">
                      {tr('加载', 'Load')}
                    </button>
                  )}</div>
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
