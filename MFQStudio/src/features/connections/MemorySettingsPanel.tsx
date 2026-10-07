import { useRuntime } from '../../app/RuntimeProvider';
import { useSettings } from '../settings/SettingsProvider';
import { SectionLabel, SettingRow, TMPanel } from '../../app/display';
import { formatBytes, formatNumber } from '../../app/formatters';
import { ModelContextSettings } from '../runtime/ModelContextSettings';
import { MemoryBudgetControls } from './MemoryBudgetControls';
import { PrefixCacheDirectory } from './PrefixCacheDirectory';
import { PrefixDiskBudgetControls } from './PrefixDiskBudgetControls';
export function MemorySettingsPanel() {
  const { runtime, instances } = useRuntime();
  const { tr } = useSettings();
  const loaded = instances.filter((item) => item.state === 'ready' || item.state === 'busy');
  const amounts = loaded.map((item) => item.memory?.resident_weight_bytes ?? item.resident_bytes);
  const memory = amounts.reduce<number>((sum, bytes) => sum + (bytes ?? 0), 0);
  const missing = amounts.some((bytes) => bytes == null);
  const residency = formatBytes(memory).replace(/\b(KB|MB|GB|TB)\b/g, (unit) => `${unit[0]}iB`);
  const diskBudget = Number(runtime?.prefix_cache_total_disk_max_bytes ?? runtime?.prefix_cache_disk_max_bytes ?? 0);
  const persistent = typeof runtime?.prefix_cache_total_disk_max_bytes === 'number'
    || typeof runtime?.prefix_cache_max_bytes === 'number';
  const hotOnly = runtime?.prefix_cache_mode === 'single_device_hot_prefix';
  return <>
        <SectionLabel title={tr('内存规划', 'Memory plan')} />
        <TMPanel className="server-settings-panel">
          <div className="setting-list">
            <MemoryBudgetControls residency={memory > 0 ? `${missing ? '≥ ' : ''}${residency}` : '--'} />
            <SettingRow
              title={tr('启动时预热专家缓存', 'Warm expert cache on launch')}
              detail={tr(
                'MFQ 会依据架构和当前内存压力自动预热可用专家槽位。',
                'MFQ warms available expert slots according to the architecture and current memory pressure.',
              )}
              trailing={<span className="server-managed-value">{tr('自动', 'Automatic')}</span>}
            />
            <SettingRow
              title={tr('前缀块大小', 'Prefix block size')}
              detail={tr(
                '较大的块有利于长提示词吞吐，较小的块提供更细粒度的部分前缀复用。',
                'Larger blocks favor long-prompt throughput; smaller blocks allow finer partial-prefix reuse.',
              )}
              trailing={
                <div className="server-unit-value">
                  <strong>{formatNumber(runtime?.prefix_cache_block_tokens || 256)}</strong>
                  <span>tokens</span>
                </div>
              }
            />
            <ModelContextSettings />
          </div>
        </TMPanel>
        <SectionLabel title={tr('持久化前缀缓存', 'Persistent Prefix cache')} />
        <TMPanel className="server-settings-panel">
          <div className="setting-list">
            <SettingRow
              title={tr('启用 SSD 层', 'Enable SSD tier')}
              detail={tr(
                '保存经过校验的增量前缀块，并允许服务重启后继续复用。',
                'Persist incremental, checksummed Prefix blocks and recover them after restart.',
              )}
              trailing={
                <input
                  aria-label={tr('启用 SSD 层', 'Enable SSD tier')}
                  checked={persistent && !hotOnly && diskBudget > 0}
                  disabled
                  readOnly
                  type="checkbox"
                />
              }
            />
            <PrefixCacheDirectory />
            <PrefixDiskBudgetControls budget={diskBudget} />
          </div>
        </TMPanel>
  </>;
}
