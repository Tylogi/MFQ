/** 资源页管理前缀缓存与可复用运行配置，按需读取配置和模型资产。 */
import { useState } from 'react';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { useRuntime } from '../../app/RuntimeProvider';
import { useSettings } from '../settings/SettingsProvider';
import { ScreenHeader, SectionLabel, TMPanel, UsageBar, EmptyPanel } from '../../app/display';
import { errorMessage, formatNumber } from '../../app/formatters';
import { studioConfirm } from '../../studio';
import { ToolsRoutingPanel } from '../connections/ToolsRoutingPanel';
import { RuntimeProfilesPanel } from './RuntimeProfilesPanel';
import { toast } from '../../stores/toastStore';

/** 管理资源页状态；离开页面后忽略迟到的资源加载结果。 */
export function CachePage() {
  const { runtime, refreshRuntime } = useRuntime();
  const { tr } = useSettings();
  const [busy, setBusy] = useState(false);
  const runtimeCache = Number(runtime?.mlx_cache_bytes ?? runtime?.cuda_reserved_bytes ?? 0);
  const prefixCacheQueries = Number(runtime?.prefix_cache_queries || 0);
  const prefixCacheHits = Number(runtime?.prefix_cache_hits || 0);
  const prefixCacheSnapshots = Number(runtime?.prefix_cache_snapshots || 0);
  const prefixCacheBytes = Number(runtime?.prefix_cache_bytes || 0);
  const prefixCacheDiskBytes = Number(runtime?.prefix_cache_disk_bytes || 0);
  const prefixCacheDiskBudget = Number(runtime?.prefix_cache_disk_max_bytes || 0);
  const prefixCacheHotBytes = Number(runtime?.prefix_cache_hot_bytes ?? prefixCacheBytes);
  const prefixCacheHitRate =
    prefixCacheQueries > 0 ? (prefixCacheHits / prefixCacheQueries) * 100 : 0;
  const prefixCacheHotOnly = runtime?.prefix_cache_mode === 'single_device_hot_prefix';
  const prefixCachePersistent = typeof runtime?.prefix_cache_max_bytes === 'number';
  const prefixCacheSupported = prefixCachePersistent || prefixCacheHotOnly;
  /** 提交缓存或配置操作，保留确认语义并刷新对应资源。 */
  async function clearRuntimeCache() {
    const snapshots = Number(runtime?.prefix_cache_snapshots || 0);
    if (busy || snapshots <= 0 || Number(runtime?.active_requests || 0) > 0) return;
    const hotPrefixOnly = runtime?.prefix_cache_mode === 'single_device_hot_prefix';
    const confirmation = hotPrefixOnly
      ? tr(
          `清除当前设备热前缀（${formatNumber(snapshots)} 个）？此操作不会删除聊天记录。`,
          `Clear ${formatNumber(snapshots)} device-hot prefix? Chat history will be kept.`,
        )
      : tr(
          `清除 ${formatNumber(snapshots)} 个 Session KV SSD 缓存块？此操作不会删除聊天记录。`,
          `Clear ${formatNumber(snapshots)} Session KV SSD cache blocks? Chat history will be kept.`,
        );
    if (!(await studioConfirm(confirmation))) return;
    setBusy(true);
    try {
      await runtimeApi.clearRuntimeCache(runtime?.instance_id);
      await refreshRuntime(false);
      toast.success(tr('缓存已清除', 'Cache cleared successfully'));
    } catch (cause) {
      toast.error(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="dashboard-view">
      <ScreenHeader
        title={tr('资源', 'Resources')}
        subtitle={tr(
          '检查内存层级、前缀缓存与运行配置。',
          'Inspect memory, prefix caching, and runtime profiles.',
        )}
      />

      <SectionLabel
        title={tr('内存层级', 'Memory hierarchy')}
        subtitle={tr('设备、内存与持久缓存', 'Device, memory, and persistent cache')}
      />
      {prefixCacheSupported ? (
        <TMPanel className="cache-panel">
          <div className="panel-heading">
            <div>
              <h2>Session KV cache</h2>
              <p>
                {prefixCacheHotOnly
                  ? tr(
                      '当前进程内的单条设备热前缀；切换会话或编辑 prompt 会重新 prefill',
                      'One device-hot prefix in this process; switching sessions or editing the prompt triggers a fresh prefill',
                    )
                  : tr(
                      'RAM 热缓存与可跨重启复用的 SSD 前缀块',
                      'RAM hot cache with persistent SSD prefix blocks',
                    )}
              </p>
            </div>
            <b>
              {prefixCacheQueries > 0
                ? `${formatNumber(prefixCacheHitRate, 1)}% hit`
                : tr('暂无查询', 'No queries')}
            </b>
          </div>
          {prefixCacheHotOnly ? (
            <>
              <UsageBar
                label={tr('RAM 热前缀', 'RAM hot prefix')}
                used={prefixCacheHotBytes}
                total={Math.max(prefixCacheHotBytes, runtimeCache || 0)}
              />
              <div className="cache-stats">
                <div>
                  <span>{tr('活动会话', 'Active sessions')}</span>
                  <strong>{formatNumber(runtime?.prefix_cache_sessions)}</strong>
                </div>
                <div>
                  <span>{tr('设备热前缀', 'Device-hot prefix')}</span>
                  <strong>{formatNumber(prefixCacheSnapshots)}</strong>
                  <small>{tr('单物理快照', 'one physical snapshot')}</small>
                </div>
                <div>
                  <span>{tr('缓存 tokens', 'Cached tokens')}</span>
                  <strong>{formatNumber(runtime?.prefix_cache_tokens)}</strong>
                </div>
                <div>
                  <span>{tr('复用 tokens', 'Reused tokens')}</span>
                  <strong>{formatNumber(runtime?.prefix_cache_hit_tokens)}</strong>
                </div>
                <div>
                  <span>{tr('命中 / 查询', 'Hits / queries')}</span>
                  <strong>
                    {formatNumber(prefixCacheHits)} / {formatNumber(prefixCacheQueries)}
                  </strong>
                </div>
                <div>
                  <span>{tr('持久性', 'Persistence')}</span>
                  <strong>{tr('进程生命周期', 'Process lifetime')}</strong>
                  <small>{tr('不落盘', 'not stored on disk')}</small>
                </div>
              </div>
            </>
          ) : (
            <>
              <div className="cache-usage-bars">
                <UsageBar
                  label={tr('SSD 前缀缓存', 'SSD prefix cache')}
                  used={prefixCacheDiskBytes}
                  total={Math.max(prefixCacheDiskBudget, prefixCacheDiskBytes)}
                />
                <UsageBar
                  label={tr('RAM 热层', 'RAM hot tier')}
                  used={prefixCacheHotBytes}
                  total={Math.max(prefixCacheHotBytes, runtimeCache || 0)}
                />
              </div>
              <div className="cache-stats">
                <div>
                  <span>{tr('活动会话', 'Active sessions')}</span>
                  <strong>{formatNumber(runtime?.prefix_cache_sessions)}</strong>
                </div>
                <div>
                  <span>{tr('SSD 块', 'SSD blocks')}</span>
                  <strong>
                    {formatNumber(runtime?.prefix_cache_disk_blocks ?? prefixCacheSnapshots)}
                  </strong>
                </div>
                <div>
                  <span>{tr('复用 tokens', 'Reused tokens')}</span>
                  <strong>{formatNumber(runtime?.prefix_cache_hit_tokens)}</strong>
                </div>
                <div>
                  <span>{tr('SSD 占用', 'SSD usage')}</span>
                  <strong>{formatNumber(prefixCacheDiskBytes / 2 ** 30, 2)} GB</strong>
                  <small>
                    {prefixCacheDiskBudget > 0
                      ? `${formatNumber(prefixCacheDiskBudget / 2 ** 30, 0)} GB ${tr('上限', 'limit')}`
                      : ''}
                  </small>
                </div>
                <div>
                  <span>{tr('RAM 热层', 'RAM hot tier')}</span>
                  <strong>{formatNumber(prefixCacheHotBytes / 2 ** 20, 1)} MB</strong>
                  <small>{formatNumber(runtime?.prefix_cache_hot_blocks)} blocks</small>
                </div>
                <div>
                  <span>{tr('待写入', 'Pending writes')}</span>
                  <strong>{formatNumber(runtime?.prefix_cache_pending_writes)}</strong>
                  <small>
                    {formatNumber(Number(runtime?.prefix_cache_pending_bytes || 0) / 2 ** 20, 1)} /{' '}
                    {formatNumber(
                      Number(runtime?.prefix_cache_pending_max_bytes || 0) / 2 ** 20,
                      0,
                    )}{' '}
                    MB · {formatNumber(runtime?.prefix_cache_deduplicated_writes)} deduplicated
                  </small>
                </div>
                <div>
                  <span>{tr('SSD 命中', 'SSD hits')}</span>
                  <strong>{formatNumber(runtime?.prefix_cache_disk_hits)}</strong>
                  <small>{formatNumber(runtime?.prefix_cache_hot_hits)} RAM hits</small>
                </div>
                <div>
                  <span>{tr('回收', 'Evictions')}</span>
                  <strong>{formatNumber(runtime?.prefix_cache_evictions)}</strong>
                  <small>{formatNumber(runtime?.prefix_cache_corrupt_blocks)} corrupt</small>
                </div>
              </div>
            </>
          )}
          {prefixCacheSnapshots > 0 && (
            <button
              className="panel-action danger"
              disabled={busy || Number(runtime?.active_requests || 0) > 0}
              onClick={() => void clearRuntimeCache()}
              type="button"
            >
              {tr('清除 Session KV 缓存', 'Clear Session KV cache')}
            </button>
          )}
        </TMPanel>
      ) : (
        <EmptyPanel
          icon="settings"
          title={tr('Runtime 诊断未启用', 'Runtime diagnostics are offline')}
          message={tr(
            '加载模型后可查看内存与前缀缓存状态。',
            'Load a model to inspect memory and prefix-cache state.',
          )}
        />
      )}
      <RuntimeProfilesPanel />
      <ToolsRoutingPanel />
    </section>
  );
}
