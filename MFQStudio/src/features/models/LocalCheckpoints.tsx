/** Render model availability and keep unload progress visible until the operation completes. */
import { useTranslation } from 'react-i18next';
import { Icon, SectionLabel, TMPanel, EmptyPanel } from '../../app/display';
import { formatNumber } from '../../app/formatters';
import type { useModelCatalog } from './useModelCatalog';
import { ModelVendorMark } from '../../app/ModelVendorMark';
import { ModelLoadProgress } from './ModelLoadProgress';

/** Display model actions with instance-specific unload feedback and duplicate-action protection. */
export function LocalCheckpoints({ catalog }: { catalog: ReturnType<typeof useModelCatalog> }) {
  const { t } = useTranslation();
  const { runtime, artifacts, busy, instances, modelFilter,
    filteredArtifacts, unloadInstance, unloadingInstanceIds, loadArtifact, chooseModelDirectory } = catalog;
  return (
    <>
      <SectionLabel
        title={t('models:localCheckpoints.localCheckpoints')}
        subtitle={`${artifacts.length} ${t('models:localCheckpoints.localModels')}`}
      />
      {filteredArtifacts.length > 0 ? (
        <TMPanel className="model-catalog-panel model-library-panel">
          <div className="model-list">
            {filteredArtifacts.map((item) => {
              const instance = instances.find(
                (candidate) => candidate.model === item.name && candidate.state !== 'failed',
              );
              const unloading = instance && (instance.state === 'unloading' || unloadingInstanceIds.has(instance.id));
              const loaded = Boolean(instance) || item.name === runtime?.model;
              const policy = instance?.pinned
                ? t('models:localCheckpoints.pinned')
                : instance?.idle_ttl_seconds != null
                  ? `TTL ${instance.idle_ttl_seconds}s` : null;
              return (
                <div className="model-row" key={item.id}>
                  <span className={loaded ? 'model-state active'
                    : item.loadable ? 'model-state' : 'model-state failed'} />
                  <div>
                    <strong>{item.name}</strong>
                    <small>
                      {item.architecture} · {item.missing_shards ? t('models:localCheckpoints.shardsMissing', { count: item.missing_shards }) : item.complete ? `${item.shard_count} ${t('models:localCheckpoints.shards')}` : t('models:localCheckpoints.invalidFile')} ·{' '}
                      {formatNumber(item.total_bytes / 2 ** 30, 1)} GB{policy ? ` · ${policy}` : ''}
                    </small>
                    <ModelLoadProgress model={item.name} />
                  </div>
                  <div className="model-row-actions"><ModelVendorMark name={item.name} architecture={item.architecture} />{instance ? (
                    <button disabled={busy || unloading || instance.state !== 'ready'}
                      onClick={() => void unloadInstance(instance.id)} type="button">
                      {unloading ? t('models:localCheckpoints.unloading') : t('models:localCheckpoints.unload')}
                    </button>
                  ) : loaded ? (
                    <em>{t('models:localCheckpoints.loaded')}</em>
                  ) : !item.loadable ? (
                    <em className="failed" title={item.error || undefined}>
                      {item.complete && item.format === 'hf'
                        ? t('models:localCheckpoints.convertFirst') : item.missing_shards
                          ? t('models:localCheckpoints.incompleteShards') : t('models:localCheckpoints.invalid')}
                    </em>
                  ) : (
                    <button disabled={busy} onClick={() => void loadArtifact(item.name)} type="button">
                      {t('models:localCheckpoints.load')}
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
          title={(modelFilter ? t('models:localCheckpoints.noLocalModelMatches') : t('models:localCheckpoints.noLocalModelsYet'))}
          message={t('models:localCheckpoints.addAModelFolderToGetStarted')}
          action={
            <button className="primary screen-header-action" disabled={busy}
              onClick={() => void chooseModelDirectory()} type="button">
              <Icon name="folder" size={14} />{t('models:localCheckpoints.chooseModelFolder')}
            </button>
          }
        />
      )}
    </>
  );
}
