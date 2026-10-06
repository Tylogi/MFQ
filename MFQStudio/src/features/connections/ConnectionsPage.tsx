/** Configure desktop and browser service connections using the actual connected service address. */
import { localized } from '../../i18n/messages';
import { useTranslation } from 'react-i18next';
import { useEffect, useState } from 'react';
import { Icon, ScreenHeader, SectionLabel, SettingRow, TMPanel } from '../../app/display';
import { errorMessage } from '../../app/formatters';
import {
  configureStudio,
  isStudio,
  saveStudioCredential,
  studioCredential,
  type StudioConfig,
} from '../../studio';
import { useRuntime } from '../../app/RuntimeProvider';
import { runtimeModelNames } from '../runtime/modelSelection';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { resolveServiceUrl, setApiToken, setBrowserServiceUrl } from '../../shared/api/client';
import { ToolsRoutingPanel } from './ToolsRoutingPanel';
import { MemorySettingsPanel } from './MemorySettingsPanel';
import { RuntimeProfilesPanel } from '../runtime/RuntimeProfilesPanel';
import { ModelAliasMapping } from './ModelAliasMapping';
import { toast } from '../../stores/toastStore';

function browserConfig(): StudioConfig {
  const address = resolveServiceUrl();
  const url = new URL(address);
  return {
    mode: ['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname) ? 'local' : 'remote',
    remote_url: address, local_service_port: Number(url.port) || (url.protocol === 'https:' ? 443 : 80),
  };
}

/** Display and save service connection settings, reconnecting only after a successful update. */
export function ConnectionsPage() {
  const { t } = useTranslation();
  const {
    runtime,
    models,
    instances,
    selectedModel,
    studio,
    reloadService,
  } = useRuntime();
  const [draft, setDraft] = useState<StudioConfig>(() => studio?.config ?? browserConfig());
  const [listeningPort, setListeningPort] = useState<number | null>(null);
  const [token, setToken] = useState('');
  const [credentialWritable, setCredentialWritable] = useState(false);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    if (studio) setDraft(studio.config);
    else {
      let disposed = false;
      void runtimeApi.runtimeListener().then((listener) => {
        if (!disposed) {
          setDraft((current) => ({ ...current, local_service_port: listener.port }));
          setListeningPort(listener.port);
        }
      }).catch(() => {});
      return () => { disposed = true; };
    }
  }, [studio]);
  useEffect(() => {
    let disposed = false;
    void studioCredential()
      .then((value) => {
        if (!disposed) setToken(value ?? '');
      })
      .catch((cause) => {
        if (!disposed) {
          toast.error(errorMessage(cause));
        }
      });
    return () => {
      disposed = true;
    };
  }, []);
  const active = Boolean(studio?.reachable ?? runtime);
  const modelNames = runtimeModelNames(models, instances);


  async function save() {
    if (!draft || busy) return;
    setBusy(true);
    try {
      if (!Number.isInteger(draft.local_service_port) || draft.local_service_port < 1 || draft.local_service_port > 65535) {
        throw new Error(t('connections:connectionsPage.portMustBeAnIntegerBetween1And65535'));
      }
      if (isStudio()) {
        if (studio?.config.mode === 'local' && draft.mode === 'local'
            && studio.config.local_service_port !== draft.local_service_port) {
          await runtimeApi.configureRuntimeListener(draft.local_service_port);
        }
        await configureStudio(draft);
        if (credentialWritable) await saveStudioCredential(token);
      } else {
        let address = draft.remote_url.trim().replace(/\/+$/, '').replace(/\/v1$/, '');
        if (draft.mode === 'local') {
          const current = browserConfig();
          if (current.mode === 'local') await runtimeApi.configureRuntimeListener(draft.local_service_port);
          const local = new URL(current.mode === 'local' ? current.remote_url : 'http://127.0.0.1');
          local.port = String(draft.local_service_port);
          address = local.toString().replace(/\/+$/, '');
        } else {
          const parsed = new URL(address);
          if (!['http:', 'https:'].includes(parsed.protocol) || parsed.username || parsed.password) {
            throw new Error(t('connections:connectionsPage.enterAnHttpOrHttpsServiceUrlWithoutCredentials'));
          }
        }
        setBrowserServiceUrl(address);
        if (credentialWritable) setApiToken(token);
        if (draft.mode === 'local' && window.location.port === String(listeningPort)
            && listeningPort !== draft.local_service_port) {
          const page = new URL(window.location.href);
          page.port = String(draft.local_service_port);
          window.location.assign(page.toString());
          return;
        }
      }
      const reconnected = await reloadService();
      setCredentialWritable(false);
      if (reconnected) toast.success(localized('connections:connectionsPage.serverSettingsSaved'));
    } catch (cause) {
      toast.error(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }


  return (
    <section className="dashboard-view">
      <ScreenHeader
        title={t('connections:connectionsPage.service')}
        subtitle={t('connections:connectionsPage.runtimeServiceConnectionsAndModelDefaults')}
      />
      <div className="server-page">
        {active && (
          <div className="server-active-notice">
            <Icon name="info" size={15} />
            <span>
              {t('connections:connectionsPage.theServerIsActiveNetworkChangesReconnectAsSoonAsTheyAre')}
            </span>
          </div>
        )}
        <SectionLabel title={t('connections:connectionsPage.runtime')} />
        <TMPanel className="server-settings-panel">
          <div className="setting-list">
            <SettingRow
              title={t('connections:connectionsPage.modelId')}
              detail={t('connections:connectionsPage.advertisedByV1ModelsAndAcceptedByChatCompletions')}
              trailing={
                <div className="server-row-actions server-model-control">
                  <ModelAliasMapping models={modelNames} selectedModel={selectedModel} />
                </div>
              }
            />
            <SettingRow
              title={t('connections:connectionsPage.bindAddress')}
              detail={t('connections:connectionsPage.localModeStaysOn127001RemoteModeConnectsTo')}
              trailing={
                <select
                  aria-label={t('connections:connectionsPage.bindAddress')}
                  disabled={busy || !draft}
                  onChange={(event) =>
                    setDraft(
                      (current) =>
                        current && { ...current, mode: event.target.value as StudioConfig['mode'] },
                    )
                  }
                  value={draft?.mode ?? 'local'}
                >
                  <option value="local">
                    {t('connections:connectionsPage.localOnly127001')}
                  </option>
                  <option value="remote">{t('connections:connectionsPage.remoteMfqServer')}</option>
                </select>
              }
            />
            {draft?.mode === 'remote' ? (
              <>
                <SettingRow
                  title={t('connections:connectionsPage.remoteEndpoint')}
                  detail={t('connections:connectionsPage.openaiCompatibleBaseUrlForTheRemoteMfqServer')}
                  trailing={
                    <input
                      aria-label={t('connections:connectionsPage.remoteEndpoint')}
                      className="server-wide-input"
                      disabled={busy}
                      onChange={(event) =>
                        setDraft(
                          (current) => current && { ...current, remote_url: event.target.value },
                        )
                      }
                      placeholder="https://host:port"
                      type="url"
                      value={draft.remote_url}
                    />
                  }
                />
                <SettingRow
                  title={t('connections:connectionsPage.apiKey')}
                  detail={(isStudio() ? t('connections:connectionsPage.theCredentialIsStoredOnlyInTheSystemCredentialVault') : t('connections:connectionsPage.theCredentialStaysOnlyInThisPageSMemory'))}
                  trailing={
                    <input
                      aria-label={t('connections:connectionsPage.apiKey')}
                      autoComplete="off"
                      className="server-wide-input"
                      disabled={busy}
                      onChange={(event) => {
                        setToken(event.target.value);
                        setCredentialWritable(true);
                      }}
                      placeholder={t('connections:connectionsPage.optional')}
                      type="password"
                      value={token}
                    />
                  }
                />
              </>
            ) : (
              <SettingRow
                title={t('connections:connectionsPage.port')}
                detail={t('connections:connectionsPage.tcpPortUsedByTheOpenaiCompatibleHttpServer')}
                trailing={
                  <input
                    aria-label={t('connections:connectionsPage.port')}
                    className="server-number-input"
                    disabled={busy || !draft}
                    max={65535}
                    min={1}
                    onChange={(event) =>
                      setDraft(
                        (current) =>
                          current && { ...current, local_service_port: Number(event.target.value) },
                      )
                    }
                    type="number"
                    value={draft?.local_service_port ?? 8090}
                  />
                }
              />
            )}
          </div>
        </TMPanel>
        <MemorySettingsPanel />
        <SectionLabel title={t('connections:connectionsPage.automation')} />
        <TMPanel className="server-settings-panel">
          <div className="setting-list">
            <SettingRow
              title={t('connections:connectionsPage.startServerWhenMfqStudioOpens')}
              detail={t('connections:connectionsPage.localModeRestoresTheServerAutomaticallyWithTheCurrentModelAndSaved')}
              trailing={
                <input
                  aria-label={t('connections:connectionsPage.startServerWhenMfqStudioOpens')}
                  checked={draft?.mode !== 'remote'}
                  disabled
                  readOnly
                  type="checkbox"
                />
              }
            />
          </div>
        </TMPanel>
        <div className="server-page-footer">
          <button
            className="primary"
            disabled={busy || !draft}
            onClick={() => void save()}
            type="button"
          >
            {t('connections:connectionsPage.saveServerSettings')}
          </button>
        </div>
      </div>
      <RuntimeProfilesPanel />
      <ToolsRoutingPanel />
    </section>
  );
}
