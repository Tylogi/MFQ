/** 迁移 Studio 纯前端源码与样式契约；全部属于过渡源码检查，不冒充行为验证。旧 Python 测试名逐项保留用于核对映射。 */
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const sourceRoot = resolve(process.cwd(), 'src');

/** 只读取产品实现，排除测试和类型声明，防止测试自身满足契约。 */
function readSources(...paths: string[]): string {
  function collect(location: string): string[] {
    if (statSync(location).isDirectory()) {
      return readdirSync(location).sort().flatMap((name) => collect(join(location, name)));
    }
    return /\.tsx?$/.test(location) && !/\.(test|spec|d)\.tsx?$/.test(location)
      ? [readFileSync(location, 'utf8').replace(/\r\n/g, '\n')]
      : [];
  }
  return paths.flatMap((path) => collect(join(sourceRoot, path))).join('\n');
}

/** 按真实导入顺序展开 CSS，供静态样式契约使用。 */
function readStyles(path = join(sourceRoot, 'styles.css')): string {
  return readFileSync(path, 'utf8').replace(/@import\s+['"]([^'"]+)['"];/g, (_match, target: string) =>
    readStyles(resolve(dirname(path), target)),
  );
}

const APP = readSources('App.tsx', 'app', 'features');
const APP_ENTRY = readSources('App.tsx');
const API = readSources('shared/api');
const MAIN = readSources('main.tsx');
const STYLES = readStyles();

describe('Studio 过渡源码与静态样式契约（非行为测试）', () => {
  it('test_studio_supports_local_and_remote_server_connections_with_voice_controls', () => {
    const models = readSources('features/models');
    const chat = readSources('features/chat', 'features/voice');
    expect(models).toContain('canUseNativeModelPicker');
    expect(models).toContain('selectLocalModelDirectory');
    expect(models).toContain('modelsApi.registerModelDirectory');
    expect(models).toContain('ModelDirectoryDialog');
    expect(models).toContain('jumpToModelDirectory');
    expect(models).toContain('listing.current_path');
    expect(chat).toContain('RealtimeAudioController');
    expect(chat).toContain('selectInteractionMode');
    expect(models).toContain('Browse folders on the MFQ Server host.');
  });

  it('test_studio_handles_a_running_server_without_a_loaded_model', () => {
    const sessions = readSources('features/chat/hooks/useConversationSessions.ts');
    const runtime = readSources('app/RuntimeProvider.tsx');
    expect(sessions).toContain('if (!selectedModel || transitioning) return');
    expect(sessions).toContain('modelAvailable');
    expect(sessions).toContain('historyLoadedId === activeId');
    expect(runtime).toContain('isRuntimeReady(status.runtime_state)');
    expect(runtime).toContain('Promise.resolve<RuntimeModel[]>([])');
    expect(APP).toContain('No model loaded');
  });

  it('test_studio_exposes_every_loaded_model_and_switches_chat_sessions_safely', () => {
    const sessions = readSources('features/chat/hooks/useConversationSessions.ts');
    const overview = readSources('features/runtime/OverviewPage.tsx');
    expect(overview).toContain('availableModelNames.map((name)');
    expect(APP).not.toContain('artifacts.slice(0, 8)');
    expect(sessions).toContain('.forkSession(active.id, null, true, active.title, selectedModel)');
    expect(sessions).toContain('setSelectedModel(session.model)');
    expect(sessions).toContain('setActiveId(id)');
    expect(sessions).toContain('active.model === selectedModel');
    expect(sessions).toContain('generationBusy');
    expect(sessions).toContain('controller.abort()');
    expect(readSources('features/chat/components/ChatPageHeader.tsx')).toContain('disabled={busy || conversation.transitioning}');
    expect(API).toContain('model?: string');
  });

  // test_studio_uses_selected_runtime_mtp_availability 已由 runtimeContracts.test.tsx
  // 的实例优先级、模型匹配、请求参数及 UI 禁用行为覆盖；字段类型见 shared/api/contracts/runtime.ts。

  it('test_model_lifecycle_actions_stay_on_the_models_page', () => {
    const load_body = APP.slice(APP.indexOf('async function loadArtifact('), APP.indexOf('async function finishModelRegistration('));
    const unload_body = APP.slice(APP.indexOf('async function unloadInstance('), APP.indexOf('const last = runtime?.last_request'));
    expect(load_body).toContain('navigate(STUDIO_PATHS.models)');
    expect(load_body).not.toContain('navigate(STUDIO_PATHS.quantization)');
    expect(unload_body).toContain('navigate(STUDIO_PATHS.models)');
    expect(unload_body).not.toContain('navigate(STUDIO_PATHS.quantization)');
  });

  it('test_studio_can_select_and_load_an_external_mfq_directory_in_local_mode', () => {
    const models = readSources('features/models');
    expect(models).toContain('selectLocalModelDirectory()');
    expect(models).toContain('finishModelRegistration(names)');
    expect(models).toContain('Choose model folder');
    expect(models).toContain('modelsApi.loadModel(');
  });

  it('test_studio_uses_native_confirmation_dialogs_for_destructive_actions', () => {
    expect(APP).not.toContain('window.confirm');
    expect((APP.split('await studioConfirm(').length - 1)).toBeGreaterThanOrEqual(6);
  });

  it('test_studio_has_a_render_error_boundary_instead_of_a_blank_window', () => {
    expect(MAIN).toContain('class AppErrorBoundary');
    expect(MAIN).toContain('static getDerivedStateFromError');
    expect(MAIN).toContain('<FailureView');
    expect(MAIN).toContain('detail={this.state.error.message || this.state.error.name}');
    expect(MAIN).toContain('onRetry={() => window.location.reload()}');
    expect(readSources('app/FailurePage.tsx')).toContain('role="alert"');
    expect(MAIN).toContain('<AppErrorBoundary>');
  });

  it('test_studio_resolves_model_and_global_inference_settings_without_roles', () => {
    const inference = readSources('features/chat/hooks/useChatInference.ts');
    const domain = readSources('features/chat/ChatProvider.tsx');
    expect(APP).toContain('inheritModelDefaults: true');
    expect(inference).toContain('const effectiveSettings = useMemo');
    expect(inference).toContain('modeTemplateSettings(settings, mode, runtime, realtime)');
    expect(domain).toContain('sampling: inference.sampling');
    expect(domain).toContain('system_prompt: inference.effectiveSettings.systemPrompt.trim()');
    expect(domain).toContain('systemPrompt: value.systemPrompt.trim()');
    expect(inference).toContain('max_tokens: effectiveSettings.maxTokens');
    expect(APP).not.toContain('roleGenerationSettings');
    expect(APP).not.toContain('LANGUAGE_CONSISTENCY_PROMPT');
    expect(APP).not.toContain('Before answering, identify the language');
  });

  it('test_studio_defaults_global_settings_to_inherited_model_parameters', () => {
    expect(APP).toContain('className="settings-page-inherited" disabled={settingsDraft.inheritModelDefaults}');
    expect(APP).toContain('checked={settingsDraft.inheritModelDefaults}');
    expect(APP).toContain('inherit_global_settings: preset.inheritGlobalSettings');
    expect(APP).toContain('typeof raw.inheritGlobalSettings === "boolean" ? raw.inheritGlobalSettings : true');
    expect(APP).toContain('typeof preset.metadata?.inherit_global_settings === "boolean"');
    expect(STYLES).toContain('.settings-page-inherited:disabled');
    expect(APP).not.toContain('className="role-inherited-fields"');
  });

  it('test_studio_exposes_theme_selection_without_using_sidebar_status_space', () => {
    expect(APP).toContain('settingsDraft.theme');
    expect(APP).toContain('<option value="system">{tr("跟随系统", "System")}</option>');
    expect(APP).toContain('<option value="light">{tr("浅色", "Light")}</option>');
    expect(APP).toContain('<option value="dark">{tr("深色", "Dark")}</option>');
    expect(APP).not.toContain('className="theme-switcher"');
    expect(APP).not.toContain('connection-card');
    expect(STYLES).not.toContain('.connection-card');
  });

  it('test_studio_exposes_omlx_style_runtime_lifecycle_controls', () => {
    const models = readSources('features/models');
    expect(APP).toContain('className="runtime-hero"');
    expect(models).toContain('Pin in memory');
    expect(models).toContain('Idle unload');
    expect(models).toContain('pin: loadPinned');
    expect(models).toContain('idle_ttl_seconds: loadIdleTtl');
    expect(API).toContain('idle_ttl_seconds?: number | null');
    expect(API).toContain('pin?: boolean');
    expect(STYLES).toContain('.runtime-hero {');
  });

  it('test_studio_runtime_monogram_tracks_the_real_model_lifecycle', () => {
    const hero = readSources('features/runtime/RuntimeHero.tsx');
    expect(hero).toContain('runtime?.model || \'Empty\'');
    expect(hero).toContain('job.kind === \'model.load\'');
    expect(hero).toContain('instance.state === \'loading\'');
    expect(hero).toContain('instance.state === \'failed\'');
    expect(hero).toContain('name={modelHero.name} state={modelHero.state}');
    expect(hero).toContain('runtime-status-pill ${modelHero.state}');
    for (const state of ['loading', 'ready', 'failed']) {
      expect(STYLES).toContain(`.model-monogram.${state}`);
    }
  });

  it('test_studio_overview_lists_every_loaded_model', () => {
    const overview = readSources('features/runtime/OverviewPage.tsx');
    expect(overview).toContain('className="overview-models-panel"');
    expect(overview).toContain('availableModelNames.map((name)');
    expect(overview).toContain('candidate.model === name && candidate.state !== \'failed\'');
    expect(overview).toContain('onClick={() => selectModel(name)}');
    expect(STYLES).toContain('.overview-model-grid {');
  });

  it('test_studio_adapts_prefix_cache_panel_to_flash_next_hot_cache', () => {
    const cache = readSources('features/runtime/CachePage.tsx');
    for (const field of ['prefix_cache_mode', 'prefix_cache_pending_bytes', 'prefix_cache_pending_max_bytes']) {
      expect(API).toContain(field);
    }
    expect(cache).toContain('single_device_hot_prefix');
    expect(cache).toContain('Device-hot prefix');
    expect(cache).toContain('Process lifetime');
    expect(cache).toContain('runtimeApi.clearRuntimeCache');
  });

  it('test_studio_uses_theme_aware_model_actions_and_readable_errors', () => {
    expect(STYLES).toContain('.panel-heading-actions button {');
    expect(STYLES).toContain('border: 1px solid var(--accent-border)');
    expect(STYLES).toContain('.mcp-form button { min-width: 64px;');
    expect(STYLES).toContain('.job-actions .secondary { border: 1px solid var(--panel-line);');
    expect(STYLES).toContain('.runtime-log p { min-width: 0; overflow-wrap: anywhere;');
    expect(STYLES).toContain('.error-banner span { min-width: 0; overflow-wrap: anywhere;');
  });

  it('test_dashboard_uses_hivellm_style_static_backend_console_components', () => {
    const shell = readSources('app/StudioShell.tsx');
    for (const component of ['ScreenHeader', 'SectionLabel', 'TMPanel', 'MetricTile', 'SettingRow', 'UsageBar', 'EmptyPanel', 'PanelDeck']) {
      expect(APP).toContain(`function ${component}`);
    }
    expect(APP).toContain('className="overview-memory-panel"');
    expect(APP).toContain('className="overview-footer-grid"');
    expect(APP).toContain('PANEL_COLLAPSED_KEY');
    expect(APP).toContain('aria-expanded={!isCollapsed}');
    expect(APP).toContain('const storageKey = `${page}:${id}`');
    expect(APP).toContain('localStorage.setItem(PANEL_COLLAPSED_KEY, JSON.stringify(updated))');
    expect(shell).toContain('<Outlet />');
    expect(APP_ENTRY).toContain('path="settings"');
    expect(APP_ENTRY).toContain('path="quantization"');
    const jobs = readSources('features/jobs/useJobRecordActions.ts');
    expect(jobs).toContain('jobsApi.clearCompletedJobs()');
    expect(jobs).toContain('jobsApi.deleteJob(id)');
    expect(APP).not.toContain('className="drawer-scrim"');
  });

  it('test_studio_streams_active_job_updates_without_polling_the_runtime', () => {
    const runtime = readSources('app/RuntimeProvider.tsx');
    const jobs = readSources('stores/jobStore.ts');
    expect(runtime).toContain('.watchActiveJobs(');
    expect(jobs).toContain('get().streamJobEvents(id,');
    expect(jobs).toContain('.streamJobEvents(');
    expect(API).toContain('/api/v1/jobs/${id}/events/stream');
    expect(API).toContain('readEventStream(response, onEvent, signal)');
    expect(APP).not.toContain('window.setInterval(() => void refreshRuntime(true), 2500)');
  });

  it('test_server_settings_are_available_while_local_startup_is_pending', () => {
    const runtime = readSources('app/RuntimeProvider.tsx');
    const shell = readSources('app/StudioShell.tsx');
    const connection = readSources('features/connections/ConnectionsPage.tsx');
    expect(runtime.indexOf('setStudio(status)')).toBeLessThan(runtime.indexOf('await startLocalStudio()'));
    expect(shell).toContain('location.pathname === \'/runtime\'');
    expect(connection).toContain('await configureStudio(draft)');
    expect(connection).toContain('if (credentialWritable) await saveStudioCredential(token)');
    expect(connection).toContain('setCredentialWritable(true)');
    expect(connection).toContain('await reloadService()');
  });

  it('test_server_page_matches_hivellm_information_architecture', () => {
    const connection = readSources('features/connections/ConnectionsPage.tsx', 'features/connections/MemorySettingsPanel.tsx', 'features/connections/InferenceDefaultsPanel.tsx');
    for (const label of ['Runtime', 'Memory plan', 'Persistent Prefix cache', 'Chat', 'Automation', 'Model ID', 'Bind address', 'Maximum output']) {
      expect(connection).toContain(label);
    }
    expect(connection).toContain('<ToolsRoutingPanel />');
    expect(connection).toContain('className="server-active-notice"');
    expect(APP_ENTRY).toContain('path="runtime"');
    expect(STYLES).toContain('.server-active-notice');
  });

});
