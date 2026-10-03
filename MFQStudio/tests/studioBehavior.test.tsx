/** 验证旧 Studio 契约中的 Markdown 修复、模型链接下载及面板持久化行为；少量接线检查单独标为过渡源码。 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { renderMarkdown } from '../src/features/chat/MessageMarkdown';
import { parseHubReference } from '../src/features/models/hubReference';
import { ModelBrowser } from '../src/features/models/ModelBrowser';
import { ModelHubPage } from '../src/features/models/ModelHubPage';
import { modelsApi } from '../src/shared/api/resources/models';
import { jobsApi } from '../src/shared/api/resources/jobs';
import { PanelDeck, PANEL_COLLAPSED_KEY } from '../src/app/PanelDeck';

const { addJob } = vi.hoisted(() => ({ addJob: vi.fn() }));
vi.mock('../src/app/RuntimeProvider', () => ({ useRuntime: () => ({ addJob }) }));
vi.mock('../src/features/settings/SettingsProvider', () => ({
  useSettings: () => ({ tr: (_zh: string, en: string) => en }),
}));

beforeEach(() => {
  vi.restoreAllMocks();
  addJob.mockReset();
});

describe('test_assistant_markdown_recovers_fully_escaped_structural_line_breaks', () => {
  it('行为：通过真实消息渲染入口恢复助手列表，关闭修复则保留字面换行', async () => {
    const text = String.raw`Heading\n\n- first\n- second`;
    const view = render(renderMarkdown(text, false, true));
    expect(await screen.findAllByRole('listitem')).toHaveLength(2);
    view.rerender(renderMarkdown(text, false, false));
    await waitFor(() => expect(screen.queryByRole('list')).not.toBeInTheDocument());
    expect(view.container).toHaveTextContent(text);
  });

  it.each([
    JSON.stringify({ text: 'first\n\n- second' }),
    '```js\nconst text = "\\n";\n```',
    String.raw`plain\ntext`,
  ])('行为：修复开关不破坏 JSON、代码或普通文本 %s', async (text) => {
    const view = render(renderMarkdown(text, false, true));
    await waitFor(() => expect(view.container.querySelector('div.rich-text')).not.toBeNull());
    expect(view.container.querySelector('li')).toBeNull();
    expect(view.container.textContent).toContain(text.startsWith('```') ? 'const text = "\\n";' : text);
  });

  it('过渡源码：持久消息仅为助手角色启用修复（非行为）', () => {
    const saved = readFileSync(resolve('src/features/chat/SavedMessageList.tsx'), 'utf8');
    expect(saved).toContain('renderMarkdown(parts.text, false, message.role === "assistant")');
    expect(saved).toContain('renderMarkdown(parts.reasoning, false, message.role === "assistant")');
  });
});

describe('test_model_hub_accepts_repository_links_and_downloads_into_the_model_catalog（行为）', () => {
  it.each([
    ['https://huggingface.co/team/model/tree/main', 'huggingface', 'team/model', 'main'],
    ['https://modelscope.cn/models/team/model/tree/release/v1', 'modelscope', 'team/model', 'release/v1'],
    [' team/model ', 'modelscope', 'team/model', undefined],
  ] as const)('解析 %s 的提供方、仓库和版本', (input, provider, repoId, revision) => {
    expect(parseHubReference(input, 'modelscope')).toEqual({ provider, repoId, ...(revision ? { revision } : {}) });
  });

  it.each(['', 'https://example.org/team/model', 'https://huggingface.co/team', 'https://huggingface.co/%ZZ/model'])('拒绝无效仓库 %s', (input) => {
    expect(parseHubReference(input, 'modelscope')).toBeNull();
  });

  it.each(['huggingface', 'modelscope'] as const)('%s 链接经页面提交下载至模型目录并登记任务', async (provider) => {
    const host = provider === 'huggingface' ? 'huggingface.co' : 'modelscope.cn/models';
    const configuration = {
      status: 'recommended' as const,
      recommendation: 'three_stars' as const,
      required_memory_bytes: 4096,
      available_memory_bytes: 16384,
      reasons: ['fits'],
    };
    const info = {
      provider, repo_id: 'team/model', revision: 'main', files: [], tags: [], downloads: 10, likes: 1, total_bytes: 4096,
      source_url: `https://${host}/team/model`, architectures: [], modalities: ['text'],
      gated: false, runtime_compatible: true,
      variants: [{
        id: 'mfq:model', label: 'model', format: 'mfq' as const,
        files: ['model.mfq'], byte_size: 4096, configuration,
      }],
    };
    const job = {
      id: 'download-job', kind: `download.${provider}`, status: 'queued' as const,
      payload: {}, progress: 0, cancel_requested: false, created_at: '', updated_at: '',
    };
    vi.spyOn(modelsApi, 'officialHubModels').mockResolvedValue({
      system: { platform: 'test', machine: 'test', backend: 'unknown' },
      data: [],
    });
    vi.spyOn(jobsApi, 'jobKinds').mockResolvedValue([{ kind: `download.${provider}`, payload_schema: {} }]);
    const inspect = vi.spyOn(modelsApi, 'resolveHubModel').mockResolvedValue(info);
    const createJob = vi.spyOn(jobsApi, 'createJob').mockResolvedValue(job);
    render(<MemoryRouter><ModelHubPage /></MemoryRouter>);
    fireEvent.click(screen.getByRole('tab', { name: 'Community' }));
    fireEvent.change(screen.getByPlaceholderText('Model name, owner/repo, or repository URL'), { target: { value: `https://${host}/team/model/tree/main` } });
    fireEvent.click(screen.getByRole('button', { name: 'Search' }));
    const download = await screen.findByRole('button', { name: 'Download' });
    await waitFor(() => expect(download).toBeEnabled());
    expect(inspect).toHaveBeenCalledExactlyOnceWith(
      `https://${host}/team/model/tree/main`,
      'huggingface',
    );
    fireEvent.click(download);
    await waitFor(() => expect(createJob).toHaveBeenCalledExactlyOnceWith(`download.${provider}`, {
      repo_id: 'team/model',
      destination: `models/${provider}/team/model/model`,
      revision: 'main',
      include: [
        'model.mfq', '*.json', '*.txt', '*.model', '*.tiktoken', '*.jinja',
        'tokenizer*', 'processor*', 'preprocessor*',
      ],
      expected_bytes: 4096,
    }));
    expect(addJob).toHaveBeenCalledWith(job);
  });
});

describe('test_model_hub_renders_device_recommendation_grades（行为）', () => {
  it('只用图标显示内存压力档位，并明确不代表模型能力或质量', async () => {
    const grades = [
      ['three_stars', 'recommended', '★★★', 'Low memory pressure: all precision tiers fit fully in memory', 'three-stars'],
      ['two_stars', 'recommended', '★★', 'Moderate memory pressure: most precision tiers fit fully in memory', 'two-stars'],
      ['one_star', 'recommended', '★', 'High memory pressure: only some precision tiers fit fully in memory', 'one-star'],
      ['caution', 'warning', '▲', 'Memory near limit: the smallest tier is close to fitting fully', 'caution'],
      ['not_recommended', 'warning', '✕', 'Insufficient memory: the smallest tier does not fit fully', 'not-recommended'],
      ['unknown', 'unknown', '?', 'Memory pressure unknown', 'unknown'],
    ] as const;
    const source = {
      provider: 'huggingface' as const,
      repo_id: 'team/model',
      revision: 'main',
      url: 'https://huggingface.co/team/model',
      available: true,
    };
    vi.spyOn(modelsApi, 'officialHubModels').mockResolvedValue({
      system: { platform: 'test', machine: 'test', backend: 'unknown' },
      data: grades.map(([recommendation, status], index) => ({
        id: `model-${index}`,
        name: `Model ${index}`,
        family: 'MFQ',
        architecture: 'test',
        description: 'test model',
        description_zh: '测试模型',
        modalities: ['text'],
        capabilities: [],
        precision_options: [],
        supports_ssd_streaming: false,
        sources: [source],
        selected_source: source,
        revision: 'main',
        downloads: 0,
        likes: 0,
        variants: index === 0 ? [{
          id: 'mfq:tier',
          label: 'tier',
          format: 'mfq' as const,
          precision: 'S4',
          files: ['tier.mfq'],
          byte_size: 70,
          configuration: {
            status: 'recommended' as const,
            recommendation: 'three_stars' as const,
            required_memory_bytes: 75,
            available_memory_bytes: 100,
            reasons: [],
          },
        }] : [],
        configuration: {
          status,
          recommendation,
          required_memory_bytes: 100,
          available_memory_bytes: 100,
          reasons: [],
        },
      })),
    });
    const view = render(
      <MemoryRouter>
        <ModelBrowser
          tab="official"
          onTabChange={vi.fn()}
          jobKinds={[]}
          onError={vi.fn()}
          onJobCreated={vi.fn()}
          tr={(_zh, en) => en}
        />
      </MemoryRouter>,
    );

    await screen.findAllByLabelText('Low memory pressure: all precision tiers fit fully in memory');
    expect(screen.getByText('Memory pressure')).toBeInTheDocument();
    expect(screen.getByText("Reflects only how many precision tiers fit fully in this device's memory, not model capability or quality.")).toBeInTheDocument();
    expect(screen.queryByText('3-star recommendation')).not.toBeInTheDocument();
    expect(screen.queryByText('2-star recommendation')).not.toBeInTheDocument();
    const tierPressure = screen.getByRole('progressbar', { name: 'Estimated share of runtime budget: 75.0%' });
    expect(tierPressure).toHaveAttribute('aria-valuetext', '75.0%');
    expect(tierPressure.querySelector('span')).toHaveStyle({ width: '75%' });
    expect(screen.getByText('75.0%')).toBeInTheDocument();
    expect(view.container.querySelector('.model-variant .configuration-badge')).toBeNull();
    for (const [, , symbol, label, className] of grades) {
      expect(screen.getAllByLabelText(label).length).toBeGreaterThan(0);
      expect(view.container.querySelector(`.configuration-badge.${className} b`)).toHaveTextContent(symbol);
    }
  });
});

describe('test_dashboard_uses_hivellm_style_static_backend_console_components（面板行为补充）', () => {
  it('点击折叠持久化页面和面板组合键，重新挂载恢复且不同页面互不影响', () => {
    const labels = { collapse: 'Collapse panel', expand: 'Expand panel' };
    const view = render(<PanelDeck page="overview" labels={labels}><section key="memory">Memory</section></PanelDeck>);
    fireEvent.click(screen.getByRole('button', { name: 'Collapse panel' }));
    expect(screen.getByRole('button', { name: 'Expand panel' })).toHaveAttribute('aria-expanded', 'false');
    expect(JSON.parse(localStorage.getItem(PANEL_COLLAPSED_KEY)!)).toEqual({ 'overview:memory': true });
    view.unmount();
    const restored = render(<PanelDeck page="overview" labels={labels}><section key="memory">Memory</section></PanelDeck>);
    expect(screen.getByRole('button', { name: 'Expand panel' })).toHaveAttribute('aria-expanded', 'false');
    restored.unmount();
    render(<PanelDeck page="models" labels={labels}><section key="memory">Memory</section></PanelDeck>);
    expect(screen.getByRole('button', { name: 'Collapse panel' })).toHaveAttribute('aria-expanded', 'true');
  });
});
