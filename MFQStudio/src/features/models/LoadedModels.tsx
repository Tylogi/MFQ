/** Render model availability and keep unload progress visible until the operation completes. */
import { useTranslation } from 'react-i18next';
import { Icon, SectionLabel, TMPanel, EmptyPanel } from '../../app/display';
import { formatNumber } from '../../app/formatters';
import type { useModelCatalog } from './useModelCatalog';
import { ModelVendorMark } from '../../app/ModelVendorMark';
import { ModelLoadProgress } from './ModelLoadProgress';

/** Display model actions with instance-specific unload feedback and duplicate-action protection. */
export function LoadedModels({ catalog }: { catalog: ReturnType<typeof useModelCatalog> }) {
  const { t } = useTranslation();
  const { busy, artifacts, availableModelNames, modelFilter, filteredInstances,
    unloadInstance, unloadingInstanceIds, chooseModelDirectory } = catalog;
  return (
    <>
      <SectionLabel
        title={t('models:loadedModels.loadedModels')}
        subtitle={t('models:loadedModels.availableForInference', { count: availableModelNames.length })}
      />
      {filteredInstances.length > 0 ? (
        <TMPanel className="model-catalog-panel loaded-model-panel">
          <div className="model-list">
            {filteredInstances.map((instance) => {
              const unloading = instance.state === 'unloading' || unloadingInstanceIds.has(instance.id);
              const ready = !unloading && (instance.state === 'ready' || instance.state === 'busy');
              const stateLabel = unloading
                ? t('models:loadedModels.unloading')
                : instance.state === 'loading'
                  ? t('models:loadedModels.loading')
                  : instance.state === 'failed'
                    ? t('models:loadedModels.failed')
                    : instance.state === 'busy'
                      ? t('models:loadedModels.busy')
                      : t('models:loadedModels.ready');
              return (
                <div className="model-row" key={instance.id}>
                  <span className={instance.state === 'failed'
                    ? 'model-state failed' : ready ? 'model-state active' : 'model-state'} />
                  <div>
                    <strong>{instance.model}</strong>
                    <small>
                      {stateLabel} · {formatNumber(instance.context_size)} ctx
                      {instance.pinned
                        ? ` · ${t('models:loadedModels.pinned')}`
                        : instance.idle_ttl_seconds != null
                          ? ` · TTL ${instance.idle_ttl_seconds}s` : ''}
                    </small>
                    {instance.state === 'loading' && <ModelLoadProgress model={instance.model} />}
                  </div>
                  <div className="model-row-actions">
                    <ModelVendorMark name={instance.model} architecture={artifacts.find((item) => item.name === instance.model)?.architecture} />
                    <button
                      disabled={busy || unloading || instance.state !== 'ready'}
                      onClick={() => void unloadInstance(instance.id)}
                      type="button"
                    >
                      {unloading ? t('models:loadedModels.unloading2') : t('models:loadedModels.unload')}
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
          title={(modelFilter ? t('models:loadedModels.noLoadedModelMatches') : t('models:loadedModels.noModelsLoaded'))}
          message={t('models:loadedModels.loadALocalCheckpointToStartChatting')}
          action={
            <button className="primary screen-header-action" disabled={busy}
              onClick={() => void chooseModelDirectory()} type="button">
              <Icon name="folder" size={14} />{t('models:loadedModels.addModel')}
            </button>
          }
        />
      )}
    </>
  );
}
