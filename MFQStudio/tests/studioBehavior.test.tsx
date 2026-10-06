/** Verify Markdown fixes, model-link downloads, and panel persistence from legacy Studio contracts; limited wiring checks are separately marked as transitional source checks. */
import { i18n } from '../src/i18n';
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
  useSettings: () => ({ t: i18n.getFixedT('en') }),
}));

beforeEach(() => {
  vi.restoreAllMocks();
  addJob.mockReset();
});

describe('test_assistant_markdown_recovers_fully_escaped_structural_line_breaks', () => {
  it('verifies studioBehavior test behavior 1', async () => {
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
  ])('verifies parameterized behavior %s', async (text) => {
    const view = render(renderMarkdown(text, false, true));
    await waitFor(() => expect(view.container.querySelector('div.rich-text')).not.toBeNull());
    expect(view.container.querySelector('li')).toBeNull();
    expect(view.container.textContent).toContain(text.startsWith('```') ? 'const text = "\\n";' : text);
  });

  it('verifies studioBehavior test behavior 2', () => {
    const saved = readFileSync(resolve('src/features/chat/SavedMessageList.tsx'), 'utf8');
    expect(saved).toContain('renderMarkdown(parts.text, false, message.role === "assistant")');
    expect(saved).toContain('renderMarkdown(parts.reasoning, false, message.role === "assistant")');
  });
});

describe('describes studioBehavior test behavior 3', () => {
  it.each([
    ['https://huggingface.co/team/model/tree/main', 'huggingface', 'team/model', 'main'],
    ['https://modelscope.cn/models/team/model/tree/release/v1', 'modelscope', 'team/model', 'release/v1'],
    [' team/model ', 'modelscope', 'team/model', undefined],
  ] as const)('verifies parameterized behavior %s', (input, provider, repoId, revision) => {
    expect(parseHubReference(input, 'modelscope')).toEqual({ provider, repoId, ...(revision ? { revision } : {}) });
  });

  it.each(['', 'https://example.org/team/model', 'https://huggingface.co/team', 'https://huggingface.co/%ZZ/model'])('verifies studioBehavior test behavior 4', (input) => {
    expect(parseHubReference(input, 'modelscope')).toBeNull();
  });

  it.each(['huggingface', 'modelscope'] as const)('verifies studioBehavior test behavior 5', async (provider) => {
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
      max_workers: 8,
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

describe('describes studioBehavior test behavior 6', () => {
  it('verifies studioBehavior test behavior 7', async () => {
    const grades = [
      ['three_stars', 'recommended', '★★★', 'Low weight pressure: all tier baselines fit the budget', 'three-stars'],
      ['two_stars', 'recommended', '★★', 'Moderate weight pressure: most tier baselines fit the budget', 'two-stars'],
      ['one_star', 'recommended', '★', 'High weight pressure: only some tier baselines fit the budget', 'one-star'],
      ['caution', 'warning', '▲', 'Weight budget near limit: the smallest tier baseline nearly fits', 'caution'],
      ['not_recommended', 'warning', '✕', 'Insufficient weight budget: the smallest tier baseline exceeds it', 'not-recommended'],
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
          t={i18n.getFixedT('en')}
        />
      </MemoryRouter>,
    );

    await screen.findAllByLabelText('Low weight pressure: all tier baselines fit the budget');
    expect(screen.getByText('Memory pressure')).toBeInTheDocument();
    expect(screen.getByText('Based on resident weight baselines; KV cache and runtime overhead need additional memory.')).toBeInTheDocument();
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

describe('describes studioBehavior test behavior 8', () => {
  it('verifies studioBehavior test behavior 9', () => {
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
