import { useState } from 'react';
import { useSettings } from '../settings/SettingsProvider';
import { ListToolbar } from '../../shared/ui/ListToolbar';
import { Icon, SectionLabel, TMPanel, EmptyPanel } from '../../app/display';
import type { useModelCatalog } from './useModelCatalog';
import { ModelVendorMark } from '../../app/ModelVendorMark';
import { ModelLoadProgress } from './ModelLoadProgress';
import { formatNumber } from '../../app/formatters';
import { ModelPressureBars } from './ModelMemoryPressure';
import { hasSeparateVram } from '../runtime/memoryArchitecture';

export function LocalCheckpoints({ catalog }: { catalog: ReturnType<typeof useModelCatalog> }) {
  const { tr } = useSettings();
  const { runtime, artifacts, busy, instances, modelFilter,
    setModelFilter, filteredArtifacts, unloadInstance, loadArtifact, chooseModelDirectory, openModelFiles } = catalog;
  const [status, setStatus] = useState('all');
  const [architecture, setArchitecture] = useState('all');
  const [format, setFormat] = useState('all');
  const [sort, setSort] = useState('recent');
  const isLoaded = (name: string) => instances.some(item => item.model === name && item.state !== 'failed') || name === runtime?.model;
  const visibleArtifacts = filteredArtifacts.filter(item =>
    (architecture === 'all' || item.architecture === architecture) && (format === 'all' || item.format === format) &&
    (status === 'all' || status === 'loaded' && isLoaded(item.name) || status === 'loadable' && item.loadable ||
      status === 'incomplete' && !item.complete || status === 'unavailable' && !item.loadable))
    .sort((a, b) => sort === 'name' ? a.name.localeCompare(b.name) : sort === 'size'
      ? b.total_bytes - a.total_bytes : (Date.parse(b.modified_at) || 0) - (Date.parse(a.modified_at) || 0));
  const filterCount = [status, architecture, format].filter(value => value !== 'all').length;
  const available = runtime?.runtime_memory_headroom_bytes;
  const separate = hasSeparateVram(runtime);
  const system = separate ? { physical_memory_bytes: runtime?.host_memory_total_bytes, memory_pools: [
    { kind: 'vram' as const, capacity_bytes: runtime?.device_memory_total_bytes },
    { kind: 'ram' as const, capacity_bytes: runtime?.host_memory_total_bytes },
  ] } : undefined;
  const gib = (bytes?: number | null) => bytes == null ? '—' : `${formatNumber(bytes / 2 ** 30, 1)} GiB`;
  return (
    <>
      <SectionLabel
        title={tr('本地检查点', 'Local checkpoints')}
        subtitle={`${artifacts.length} ${tr('个本地模型', 'local models')}`}
      />
      <ListToolbar label={tr('搜索本地检查点', 'Search local checkpoints')} placeholder={tr('按模型名称搜索', 'Search model names')}
        query={modelFilter} onQueryChange={setModelFilter} count={visibleArtifacts.length} total={artifacts.length}
        activeFilters={filterCount} onReset={() => { setModelFilter(''); setStatus('all'); setArchitecture('all'); setFormat('all'); setSort('recent'); }}>
        <label>{tr('模型状态', 'Model status')}<select aria-label={tr('模型状态', 'Model status')} value={status} onChange={event => setStatus(event.target.value)}>
          <option value="all">{tr('全部', 'All')}</option><option value="loaded">{tr('已加载', 'Loaded')}</option>
          <option value="loadable">{tr('可直接加载', 'Loadable')}</option><option value="incomplete">{tr('分片不全', 'Incomplete shards')}</option><option value="unavailable">{tr('不可直接加载', 'Not loadable')}</option>
        </select></label>
        <label>{tr('架构', 'Architecture')}<select aria-label={tr('筛选架构', 'Filter architecture')} value={architecture} onChange={event => setArchitecture(event.target.value)}>
          <option value="all">{tr('全部', 'All')}</option>{[...new Set(artifacts.map(item => item.architecture).filter(Boolean))].sort().map(value => <option key={value} value={value}>{value}</option>)}
        </select></label>
        <label>{tr('格式', 'Format')}<select aria-label={tr('筛选格式', 'Filter format')} value={format} onChange={event => setFormat(event.target.value)}>
          <option value="all">{tr('全部', 'All')}</option><option value="mfq">MFQ</option><option value="hf">HuggingFace</option>
        </select></label>
        <label>{tr('排序', 'Sort')}<select aria-label={tr('检查点排序', 'Checkpoint sort')} value={sort} onChange={event => setSort(event.target.value)}>
          <option value="recent">{tr('最近修改', 'Recently modified')}</option><option value="name">{tr('模型名称', 'Model name')}</option><option value="size">{tr('文件大小', 'File size')}</option>
        </select></label>
      </ListToolbar>
      {visibleArtifacts.length > 0 ? (
        <TMPanel className="model-catalog-panel model-library-panel">
          <div className="model-list">
            {visibleArtifacts.map((item) => {
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
                    {!separate && <small>{tr('剩余可用内存', 'Remaining available memory')} {gib(available)}</small>}
                    <ModelPressureBars weights={item.estimated_resident_weight_bytes} roles={item.estimated_weight_bytes_by_role}
                      system={system} available={available}
                      label={tr('预计占剩余可用内存', 'Estimated share of remaining available memory')}
                      tr={tr} />
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
          title={tr(modelFilter || filterCount ? '没有匹配的本地模型' : '还没有本地模型',
            modelFilter || filterCount ? 'No local model matches' : 'No local models yet')}
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
