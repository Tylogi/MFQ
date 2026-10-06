/** Provide CachePage interface behavior. */
import { localized } from '../../i18n/messages';
import { useTranslation } from 'react-i18next';
import { useState } from 'react';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { useRuntime } from '../../app/RuntimeProvider';
import { ScreenHeader, SectionLabel, TMPanel, UsageBar, EmptyPanel } from '../../app/display';
import { errorMessage, formatNumber } from '../../app/formatters';
import { studioConfirm } from '../../studio';
import { ResourceMonitorPanel } from './ResourceMonitorPanel';
import { toast } from '../../stores/toastStore';

/** Display resource telemetry and clear the selected model's prefix cache on request. */
export function CachePage() {
  const { runtime, refreshRuntime } = useRuntime();
  const { t } = useTranslation();
  const [busy, setBusy] = useState(false);
  const prefixCacheRamBudget = Number(runtime?.prefix_cache_max_bytes ?? 0);
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
  const prefixCacheSupported = runtime?.prefix_cache_supported !== undefined
    ? Number(runtime.prefix_cache_supported) > 0
    : prefixCachePersistent || prefixCacheHotOnly;
  const prefixCacheUnavailableReason = Number(runtime?.prefix_cache_disabled_reason) === 2
    ? t('runtime:cachePage.sessionKvCacheIsUnavailableWithContinuousBatching')
    : t('runtime:cachePage.theCurrentModelDoesNotSupportSessionKvCache');
  async function clearRuntimeCache() {
    const snapshots = Number(runtime?.prefix_cache_snapshots || 0);
    if (busy || snapshots <= 0 || Number(runtime?.active_requests || 0) > 0) return;
    const hotPrefixOnly = runtime?.prefix_cache_mode === 'single_device_hot_prefix';
    const confirmation = hotPrefixOnly
      ? t('runtime:cachePage.clearDeviceHotPrefixChatHistoryWillBeKept', { count: snapshots })
      : t('runtime:cachePage.clearSessionKvSsdCacheBlocksChatHistoryWillBeKept', { count: snapshots });
    if (!(await studioConfirm(confirmation))) return;
    setBusy(true);
    try {
      await runtimeApi.clearRuntimeCache(runtime?.instance_id);
      await refreshRuntime(false);
      toast.success(localized('runtime:cachePage.cacheClearedSuccessfully'));
    } catch (cause) {
      toast.error(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="dashboard-view">
      <ScreenHeader
        title={t('runtime:cachePage.resources')}
        subtitle={t('runtime:cachePage.monitorComputeUtilizationWeightTrafficBandwidthAndPrefixCaching')}
      />

      <ResourceMonitorPanel />
      <SectionLabel title={t('runtime:cachePage.prefixCache')} />
      {prefixCacheSupported ? (
        <TMPanel className="cache-panel">
          <div className="panel-heading">
            <div>
              <h2>Session KV cache</h2>
              <p>
                {prefixCacheHotOnly
                  ? t('runtime:cachePage.oneDeviceHotPrefixInThisProcessSwitchingSessionsOrEditingThe')
                  : t('runtime:cachePage.ramHotCacheWithPersistentSsdPrefixBlocks')}
              </p>
            </div>
            <b>
              {prefixCacheQueries > 0
                ? `${formatNumber(prefixCacheHitRate, 1)}% hit`
                : t('runtime:cachePage.noQueries')}
            </b>
          </div>
          {prefixCacheHotOnly ? (
            <>
              <UsageBar
                label={t('runtime:cachePage.ramHotPrefix')}
                used={prefixCacheHotBytes}
                total={Math.max(prefixCacheHotBytes, prefixCacheRamBudget)}
              />
              <div className="cache-stats">
                <div>
                  <span>{t('runtime:cachePage.activeSessions')}</span>
                  <strong>{formatNumber(runtime?.prefix_cache_sessions)}</strong>
                </div>
                <div>
                  <span>{t('runtime:cachePage.deviceHotPrefix')}</span>
                  <strong>{formatNumber(prefixCacheSnapshots)}</strong>
                  <small>{t('runtime:cachePage.onePhysicalSnapshot')}</small>
                </div>
                <div>
                  <span>{t('runtime:cachePage.cachedTokens')}</span>
                  <strong>{formatNumber(runtime?.prefix_cache_tokens)}</strong>
                </div>
                <div>
                  <span>{t('runtime:cachePage.reusedTokens')}</span>
                  <strong>{formatNumber(runtime?.prefix_cache_hit_tokens)}</strong>
                </div>
                <div>
                  <span>{t('runtime:cachePage.hitsQueries')}</span>
                  <strong>
                    {formatNumber(prefixCacheHits)} / {formatNumber(prefixCacheQueries)}
                  </strong>
                </div>
                <div>
                  <span>{t('runtime:cachePage.persistence')}</span>
                  <strong>{t('runtime:cachePage.processLifetime')}</strong>
                  <small>{t('runtime:cachePage.notStoredOnDisk')}</small>
                </div>
              </div>
            </>
          ) : (
            <>
              <div className="cache-usage-bars">
                <UsageBar
                  label={t('runtime:cachePage.ssdPrefixCache')}
                  used={prefixCacheDiskBytes}
                  total={Math.max(prefixCacheDiskBudget, prefixCacheDiskBytes)}
                />
                <UsageBar
                  label={t('runtime:cachePage.ramHotTier')}
                  used={prefixCacheHotBytes}
                  total={Math.max(prefixCacheHotBytes, prefixCacheRamBudget)}
                />
              </div>
              <div className="cache-stats">
                <div>
                  <span>{t('runtime:cachePage.activeSessions')}</span>
                  <strong>{formatNumber(runtime?.prefix_cache_sessions)}</strong>
                </div>
                <div>
                  <span>{t('runtime:cachePage.ssdBlocks')}</span>
                  <strong>
                    {formatNumber(runtime?.prefix_cache_disk_blocks ?? prefixCacheSnapshots)}
                  </strong>
                </div>
                <div>
                  <span>{t('runtime:cachePage.reusedTokens')}</span>
                  <strong>{formatNumber(runtime?.prefix_cache_hit_tokens)}</strong>
                </div>
                <div>
                  <span>{t('runtime:cachePage.ssdUsage')}</span>
                  <strong>{formatNumber(prefixCacheDiskBytes / 2 ** 30, 2)} GiB</strong>
                  <small>
                    {prefixCacheDiskBudget > 0
                      ? `${formatNumber(prefixCacheDiskBudget / 2 ** 30, 0)} GiB ${t('runtime:cachePage.limit')}`
                      : ''}
                  </small>
                </div>
                <div>
                  <span>{t('runtime:cachePage.ramHotTier')}</span>
                  <strong>{formatNumber(prefixCacheHotBytes / 2 ** 20, 1)} MiB</strong>
                  <small>{formatNumber(runtime?.prefix_cache_hot_blocks)} blocks</small>
                </div>
                <div>
                  <span>{t('runtime:cachePage.pendingWrites')}</span>
                  <strong>{formatNumber(runtime?.prefix_cache_pending_writes)}</strong>
                  <small>
                    {formatNumber(Number(runtime?.prefix_cache_pending_bytes || 0) / 2 ** 20, 1)} /{' '}
                    {formatNumber(
                      Number(runtime?.prefix_cache_pending_max_bytes || 0) / 2 ** 20,
                      0,
                    )}{' '}
                    MiB · {formatNumber(runtime?.prefix_cache_deduplicated_writes)} deduplicated
                  </small>
                </div>
                <div>
                  <span>{t('runtime:cachePage.ssdHits')}</span>
                  <strong>{formatNumber(runtime?.prefix_cache_disk_hits)}</strong>
                  <small>{formatNumber(runtime?.prefix_cache_hot_hits)} RAM hits</small>
                </div>
                <div>
                  <span>{t('runtime:cachePage.evictions')}</span>
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
              {t('runtime:cachePage.clearSessionKvCache')}
            </button>
          )}
        </TMPanel>
      ) : (
        <EmptyPanel
          icon="settings"
          title={
            runtime
              ? t('runtime:cachePage.sessionKvCacheUnavailable')
              : t('runtime:cachePage.runtimeDiagnosticsAreOffline')
          }
          message={
            runtime
              ? prefixCacheUnavailableReason
              : t('runtime:cachePage.loadAModelToInspectMemoryAndPrefixCacheState')
          }
        />
      )}
    </section>
  );
}
