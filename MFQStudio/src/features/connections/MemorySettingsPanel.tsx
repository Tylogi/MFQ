/** 在连接页展示内存和缓存策略，并独立管理上下文重载操作。 */
import { useState } from 'react';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { useRuntime } from '../../app/RuntimeProvider';
import { useSettings } from '../settings/SettingsProvider';
import { SectionLabel, SettingRow, TMPanel } from '../../app/display';
import { errorMessage, formatBytes, formatNumber } from '../../app/formatters';
import { studioConfirm } from '../../studio';
import { toast } from '../../stores/toastStore';
/** 只订阅运行时与上下文设置，重载期间不阻塞其他连接配置。 */
export function MemorySettingsPanel() {
  const { runtime, refreshRuntime } = useRuntime();
  const { tr, contextSize, setContextSize } = useSettings();
  const [busy, setBusy] = useState(false);
  const memory = Number(
    runtime?.mlx_active_bytes ??
      runtime?.cuda_allocated_bytes ??
      runtime?.process_resident_bytes ??
      0,
  );
  const cache = Number(runtime?.mlx_cache_bytes ?? runtime?.cuda_reserved_bytes ?? 0);
  const hotBytes = Number(runtime?.prefix_cache_hot_bytes ?? runtime?.prefix_cache_bytes ?? 0);
  const diskBudget = Number(runtime?.prefix_cache_disk_max_bytes ?? 0);
  const persistent = typeof runtime?.prefix_cache_max_bytes === 'number';
  const hotOnly = runtime?.prefix_cache_mode === 'single_device_hot_prefix';
  /** 确认上下文变更后重新加载活动实例。 */
  async function reloadRuntime() {
    if (
      busy ||
      !(await studioConfirm(
        tr(
          `以 ${formatNumber(contextSize)} token 上下文重载模型？`,
          `Reload the model with a ${formatNumber(contextSize)} token context?`,
        ),
      ))
    )
      return;
    setBusy(true);
    try {
      await runtimeApi.reloadRuntime(contextSize, runtime?.instance_id);
      await refreshRuntime(true);
      toast.success(tr('模型重载成功', 'Model reloaded successfully'));
    } catch (cause) {
      toast.error(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }
  return <>
        <SectionLabel title={tr('内存规划', 'Memory plan')} />
        <TMPanel className="server-settings-panel">
          <div className="setting-list">
            <SettingRow
              title={tr('模型总驻留', 'Total model residency')}
              detail={tr(
                'MFQ 根据模型、专家缓存和设备可用内存自动规划驻留。',
                'MFQ plans model, expert-cache, and device residency from the available memory automatically.',
              )}
              trailing={
                <strong>
                  {memory + cache > 0 ? formatBytes(memory + cache) : tr('自动', 'Automatic')}
                </strong>
              }
            />
            <SettingRow
              title={tr('前缀 RAM 配额', 'Prefix RAM allowance')}
              detail={tr(
                'RAM 热层从统一内存预算中分配，剩余空间可用于模型和专家缓存。',
                'The RAM hot tier is allocated inside the unified budget, leaving the remainder for model and expert caching.',
              )}
              trailing={
                <strong>{hotBytes > 0 ? formatBytes(hotBytes) : tr('自动', 'Automatic')}</strong>
              }
            />
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
            <SettingRow
              title={tr('最大上下文', 'Maximum context')}
              detail={tr(
                '修改后重载当前模型；未加载模型时会作为下一次加载的默认值。',
                'Reloads the active model after a change; otherwise it becomes the next load default.',
              )}
              trailing={
                <div className="server-row-actions">
                  <div className="server-input-unit">
                    <input
                      aria-label={tr('最大上下文', 'Maximum context')}
                      className="server-number-input"
                      max={1048576}
                      min={512}
                      onChange={(event) => setContextSize(Number(event.target.value))}
                      type="number"
                      value={contextSize}
                    />
                    <span>tokens</span>
                  </div>
                  <button
                    disabled={busy || !runtime?.model}
                    onClick={() => void reloadRuntime()}
                    type="button"
                  >
                    {tr('重载', 'Reload')}
                  </button>
                </div>
              }
            />
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
                  checked={persistent && !hotOnly}
                  disabled
                  readOnly
                  type="checkbox"
                />
              }
            />
            <SettingRow
              title={tr('目录', 'Directory')}
              detail={tr(
                'MFQ 自动选择应用数据目录中的本地缓存位置。',
                'MFQ automatically uses a local cache location inside the application data directory.',
              )}
              trailing={<code>{tr('由 MFQ 管理', 'Managed by MFQ')}</code>}
            />
            <SettingRow
              title={tr('SSD 配额', 'SSD budget')}
              detail={tr(
                '基于 LRU 的回收策略会将磁盘占用控制在此上限内。',
                'Leaf-aware LRU keeps disk usage within this ceiling.',
              )}
              trailing={
                <div className="server-unit-value">
                  <strong>{diskBudget > 0 ? formatNumber(diskBudget / 2 ** 30, 0) : '--'}</strong>
                  <span>GiB</span>
                </div>
              }
            />
          </div>
        </TMPanel>
  </>;
}
