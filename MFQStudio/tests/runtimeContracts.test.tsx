/** 验证实例推理能力、附件类型和聊天工具栏的真实交互。 */
import { act, render, renderHook, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { useRuntime } from '../src/app/RuntimeProvider';
import { useSettings } from '../src/features/settings/SettingsProvider';
import { DEFAULT_SETTINGS } from '../src/features/settings/configuration';
import { useChat } from '../src/features/chat/ChatProvider';
import { useChatInference } from '../src/features/chat/hooks/useChatInference';
import { ChatToolbar } from '../src/features/chat/components/ChatToolbar';
import { useConversationSelector } from '../src/features/chat/state/conversationStore';
import { DOCUMENT_ACCEPT } from '../src/features/chat/attachments';
import type { RuntimeCapabilities, RuntimeInstance } from '../src/shared/api/types';

vi.mock('../src/app/RuntimeProvider', () => ({ useRuntime: vi.fn() }));
vi.mock('../src/features/settings/SettingsProvider', () => ({ useSettings: vi.fn() }));
vi.mock('../src/features/chat/ChatProvider', () => ({ useChat: vi.fn() }));
vi.mock('../src/features/chat/state/conversationStore', () => ({ useConversationSelector: vi.fn() }));
vi.mock('../src/features/voice/voiceLevelStore', () => ({ useVoiceLevel: () => 0 }));
vi.mock('../src/stores/jobStore', () => ({
  useJobStore: (selector: (state: { jobs: never[] }) => unknown) => selector({ jobs: [] }),
}));

let runtime: ReturnType<typeof useRuntime>;
let settings: ReturnType<typeof useSettings>['settings'];
const updateSettings = vi.fn();
const selectInteractionMode = vi.fn();

/** 创建默认仅支持文本的能力响应，按场景覆盖能力位。 */
function capabilities(features: Partial<RuntimeCapabilities['model_capabilities']['features']> = {}): RuntimeCapabilities {
  return {
    model: 'selected', model_type: 'test', vision_available: false,
    mtp_available: false, duplex_available: false,
    model_capabilities: {
      architecture_family: 'test', source: 'test',
      features: {
        text: true, image_input: false, video_input: false, audio_input: false,
        audio_output: false, full_duplex: false, mtp: false, ...features,
      },
    },
  };
}

/** 创建实例记录，区分选中模型、失败实例和其他模型。 */
function instance(overrides: Partial<RuntimeInstance> = {}): RuntimeInstance {
  return {
    id: 'selected-instance', model: 'selected', state: 'ready', devices: [],
    active_sessions: 0, queued_requests: 0, ...overrides,
  };
}

/** 将真实推理 hook 接入工具栏，验证更新设置后的界面行为。 */
function ToolbarHarness() {
  const inference = useChatInference('text');
  vi.mocked(useChat).mockReturnValue({
    inference, conversation: { conversationReady: true },
    voice: { voiceState: 'idle' }, busy: false, selectInteractionMode, toggleVoice: vi.fn(),
  } as unknown as ReturnType<typeof useChat>);
  return <ChatToolbar />;
}

beforeEach(() => {
  vi.clearAllMocks();
  settings = { ...DEFAULT_SETTINGS, inheritModelDefaults: false };
  runtime = {
    runtime: { instance_id: 'selected-instance' }, realtime: null,
    selectedModel: 'selected', instances: [], models: [], capabilities: capabilities(),
  } as unknown as ReturnType<typeof useRuntime>;
  vi.mocked(useRuntime).mockImplementation(() => runtime);
  updateSettings.mockImplementation((patch) => { settings = { ...settings, ...patch }; });
  vi.mocked(useSettings).mockImplementation(() => ({
    settings, updateSettings, english: true, tr: (_zh: string, en: string) => en,
    replaceSettings: vi.fn(), contextSize: 32768, setContextSize: vi.fn(),
  }));
  vi.mocked(useConversationSelector).mockImplementation((selector) => selector({
    activeId: 'session', sessions: [{ id: 'session', mode: 'text' }],
  } as never));
});

describe('useChatInference', () => {
  it('思考能力随运行时变化，只有支持且启用时发送 enable_thinking', () => {
    const { result, rerender } = renderHook(() => useChatInference('text'));
    expect(result.current.thinkingSupported).toBe(false);
    expect(result.current.sampling.enable_thinking).toBe(false);
    runtime.runtime = { chat_template_capabilities: { thinking: { supported: true } } };
    rerender();
    expect(result.current.thinkingSupported).toBe(true);
    expect(result.current.sampling.enable_thinking).toBe(true);
    settings = { ...settings, enableThinking: false };
    rerender();
    expect(result.current.sampling.enable_thinking).toBe(false);
  });

  it('返回 reasoning 档位并将所选档位或空值写入采样参数', () => {
    runtime.runtime = { chat_template_capabilities: {
      thinking: { supported: true }, reasoning_effort: { values: ['low', 'high'] },
    } };
    settings = { ...settings, reasoningEffort: 'high' };
    const { result, rerender } = renderHook(() => useChatInference('text'));
    expect(result.current.reasoningValues).toEqual(['low', 'high']);
    expect(result.current.sampling.reasoning_effort).toBe('high');
    settings = { ...settings, reasoningEffort: '' };
    runtime.runtime = {};
    rerender();
    expect(result.current.reasoningValues).toEqual([]);
    expect(result.current.sampling.reasoning_effort).toBeNull();
  });

  it.each([
    [true, true, true, true], [false, true, true, false],
    [true, false, true, false], [true, true, false, false],
  ])('选中实例 MTP supported=%s available=%s enabled=%s 时请求为 %s', (supported, available, enabled, expected) => {
    runtime.capabilities = { ...capabilities({ mtp: !supported }), mtp_available: !available };
    runtime.instances = [
      instance({ model: 'other', mtp_supported: true, mtp_available: true }),
      instance({ state: 'failed', mtp_supported: true, mtp_available: true }),
      instance({ mtp_supported: supported, mtp_available: available }),
    ];
    settings = { ...settings, enableMtp: enabled };
    const { result } = renderHook(() => useChatInference('text'));
    expect(result.current.mtpSupported).toBe(supported);
    expect(result.current.mtpAvailable).toBe(available);
    expect(result.current.sampling.enable_mtp).toBe(expected);
  });

  it('MTP 字段缺失时回退到选中模型能力，切换后不借用其他模型能力', () => {
    runtime.instances = [instance()];
    runtime.capabilities = { ...capabilities({ mtp: true }), mtp_available: true };
    const { result, rerender } = renderHook(() => useChatInference('text'));
    expect(result.current.sampling.enable_mtp).toBe(true);
    runtime.selectedModel = 'other';
    rerender();
    expect(result.current.mtpSupported).toBe(false);
    expect(result.current.mtpAvailable).toBe(false);
    expect(result.current.sampling.enable_mtp).toBe(false);
  });

  it.each([
    [true, true, true, true, ['image/*', 'video/*', 'audio/*']],
    [true, false, false, true, ['image/*']],
    [false, true, false, true, ['video/*']],
    [true, true, true, false, ['audio/*']],
    [false, false, false, true, []],
  ] as const)('附件过滤 image=%s video=%s audio=%s visionEnabled=%s', (image, video, audio, enabled, media) => {
    runtime.capabilities = capabilities({ image_input: image, video_input: video, audio_input: audio });
    settings = { ...settings, enableVision: enabled };
    const { result } = renderHook(() => useChatInference('text'));
    expect(result.current.attachmentAccept.split(',')).toEqual([...media, ...DOCUMENT_ACCEPT.split(',')]);
  });

  it('更新推理设置保留有效参数并退出模型默认值继承', () => {
    settings = { ...settings, inheritModelDefaults: true };
    runtime.runtime = { sampling_defaults: { temperature: 0.25 } };
    const { result } = renderHook(() => useChatInference('text'));
    act(() => result.current.updateGlobalInference({ enableVision: false }));
    expect(updateSettings).toHaveBeenCalledWith(expect.objectContaining({
      temperature: 0.25, enableVision: false, inheritModelDefaults: false, preset: 'custom',
    }));
  });
});

describe('ChatToolbar', () => {
  it('无音频能力时不显示交互模式', () => {
    render(<ToolbarHarness />);
    expect(screen.queryByRole('combobox', { name: 'Interaction mode' })).not.toBeInTheDocument();
  });

  it('有 audio 无 duplex 时禁用 full_duplex，选择 voice 更新模式', async () => {
    runtime.capabilities = capabilities({ audio_input: true });
    const user = userEvent.setup();
    render(<ToolbarHarness />);
    expect(screen.getByRole('option', { name: 'Full duplex' })).toBeDisabled();
    expect(screen.getByRole('option', { name: 'Voice' })).toBeEnabled();
    await user.selectOptions(screen.getByRole('combobox', { name: 'Interaction mode' }), 'voice');
    expect(selectInteractionMode).toHaveBeenCalledWith('voice');
  });

  it('点击视觉和 MTP 更新设置并反映开关状态', async () => {
    runtime.capabilities = capabilities({ image_input: true });
    runtime.instances = [instance({ mtp_supported: true, mtp_available: true })];
    const user = userEvent.setup();
    const view = render(<ToolbarHarness />);
    for (const [name, key] of [['Vision input', 'enableVision'], ['MTP', 'enableMtp']] as const) {
      expect(screen.getByRole('button', { name })).toHaveAttribute('aria-pressed', 'true');
      await user.click(screen.getByRole('button', { name }));
      expect(updateSettings).toHaveBeenLastCalledWith(expect.objectContaining({ [key]: false }));
      view.rerender(<ToolbarHarness />);
      expect(screen.getByRole('button', { name })).toHaveAttribute('aria-pressed', 'false');
      await user.click(screen.getByRole('button', { name }));
      expect(updateSettings).toHaveBeenLastCalledWith(expect.objectContaining({ [key]: true }));
      view.rerender(<ToolbarHarness />);
      expect(screen.getByRole('button', { name })).toHaveAttribute('aria-pressed', 'true');
    }
  });

  it('MTP 缺少可用权重时禁用按钮且点击不更新设置', async () => {
    runtime.instances = [instance({ mtp_supported: true, mtp_available: false })];
    const user = userEvent.setup();
    render(<ToolbarHarness />);
    expect(screen.getByRole('button', { name: 'MTP' })).toBeDisabled();
    await user.click(screen.getByRole('button', { name: 'MTP' }));
    expect(updateSettings).not.toHaveBeenCalled();
  });

  it.each([
    { supported: false, enabled: true, values: ['low'], visible: false },
    { supported: true, enabled: false, values: ['low'], visible: false },
    { supported: true, enabled: true, values: [], visible: false },
    { supported: true, enabled: true, values: ['low', 'high'], visible: true },
  ])('reasoning 档位支持=$supported 启用=$enabled values=$values 时可见=$visible', ({ supported, enabled, values, visible }) => {
    runtime.runtime = { chat_template_capabilities: {
      thinking: { supported }, reasoning_effort: { values },
    } };
    settings = { ...settings, enableThinking: enabled };
    render(<ToolbarHarness />);
    const select = screen.queryByRole('combobox', { name: 'Reasoning effort' });
    if (visible) expect(select).toBeInTheDocument();
    else expect(select).not.toBeInTheDocument();
  });

  it('选择 reasoning 档位更新设置，关闭思考后隐藏档位', async () => {
    runtime.runtime = { chat_template_capabilities: {
      thinking: { supported: true }, reasoning_effort: { values: ['low', 'high'] },
    } };
    const user = userEvent.setup();
    const view = render(<ToolbarHarness />);
    await user.selectOptions(screen.getByRole('combobox', { name: 'Reasoning effort' }), 'high');
    expect(updateSettings).toHaveBeenLastCalledWith(expect.objectContaining({ reasoningEffort: 'high' }));
    view.rerender(<ToolbarHarness />);
    expect(screen.getByRole('combobox', { name: 'Reasoning effort' })).toHaveValue('high');
    await user.click(screen.getByRole('button', { name: 'Thinking' }));
    view.rerender(<ToolbarHarness />);
    expect(screen.queryByRole('combobox', { name: 'Reasoning effort' })).not.toBeInTheDocument();
  });
});
