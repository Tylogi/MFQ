import { useRuntime } from '../../app/RuntimeProvider';
import { SectionLabel, SettingRow, TMPanel } from '../../app/display';
import { formatNumber } from '../../app/formatters';
import { useSettings } from '../settings/SettingsProvider';
import { PrefixCacheDirectory } from './PrefixCacheDirectory';
import { PrefixDiskBudgetControls } from './PrefixDiskBudgetControls';

export function PrefixCacheSettingsPanel() {
  const { runtime } = useRuntime();
  const { tr } = useSettings();
  const diskBudget = Number(runtime?.prefix_cache_total_disk_max_bytes ?? runtime?.prefix_cache_disk_max_bytes ?? 0);
  const persistent = typeof runtime?.prefix_cache_total_disk_max_bytes === 'number'
    || typeof runtime?.prefix_cache_max_bytes === 'number';
  const hotOnly = runtime?.prefix_cache_mode === 'single_device_hot_prefix';
  return <>
    <SectionLabel title={tr('持久化前缀缓存', 'Persistent prefix cache')} />
    <TMPanel className="server-settings-panel server-prefix-panel">
      <div className="setting-list">
        <SettingRow
          title={tr('前缀块大小', 'Prefix block size')}
          detail={tr(
            '较大的块有利于长提示词吞吐，较小的块提供更细粒度的部分前缀复用。',
            'Larger blocks favor long-prompt throughput; smaller blocks allow finer partial-prefix reuse.',
          )}
          trailing={<div className="server-unit-value">
            <strong>{formatNumber(runtime?.prefix_cache_block_tokens || 256)}</strong>
            <span>tokens</span>
          </div>}
        />
        <SettingRow
          title={tr('启用 SSD 层', 'Enable SSD tier')}
          detail={tr(
            '保存经过校验的增量前缀块，并允许服务重启后继续复用。',
            'Persist incremental, checksummed prefix blocks and recover them after restart.',
          )}
          trailing={<input
            aria-label={tr('启用 SSD 层', 'Enable SSD tier')}
            checked={persistent && !hotOnly && diskBudget > 0}
            disabled readOnly type="checkbox"
          />}
        />
        <PrefixCacheDirectory />
        <PrefixDiskBudgetControls budget={diskBudget} />
      </div>
    </TMPanel>
  </>;
}
