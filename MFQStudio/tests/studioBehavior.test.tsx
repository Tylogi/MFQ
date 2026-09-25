/** 验证旧 Studio 契约中的 Markdown 修复、模型链接下载及面板持久化行为；少量接线检查单独标为过渡源码。 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { renderMarkdown } from '../src/features/chat/MessageMarkdown';
import { parseHubReference } from '../src/features/models/hubReference';
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
    const info = {
      provider, repo_id: 'team/model', revision: 'main', files: [], tags: [], downloads: 10, likes: 1, total_bytes: 4096,
    };
    const job = {
      id: 'download-job', kind: `download.${provider}`, status: 'queued' as const,
      payload: {}, progress: 0, cancel_requested: false, created_at: '', updated_at: '',
    };
    vi.spyOn(jobsApi, 'jobKinds').mockResolvedValue([{ kind: `download.${provider}`, payload_schema: {} }]);
    const inspect = vi.spyOn(modelsApi, 'hubModelInfo').mockResolvedValue(info);
    const createJob = vi.spyOn(jobsApi, 'createJob').mockResolvedValue(job);
    render(<MemoryRouter><ModelHubPage /></MemoryRouter>);
    fireEvent.change(screen.getByPlaceholderText('Model, repository, or URL'), { target: { value: `https://${host}/team/model/tree/main` } });
    fireEvent.click(screen.getByRole('button', { name: 'Find' }));
    const download = await screen.findByRole('button', { name: 'Download' });
    await waitFor(() => expect(download).toBeEnabled());
    expect(inspect).toHaveBeenCalledExactlyOnceWith(provider, 'team/model', 'main');
    fireEvent.click(download);
    await waitFor(() => expect(createJob).toHaveBeenCalledExactlyOnceWith(`download.${provider}`, {
      repo_id: 'team/model', destination: `models/${provider}/team/model`, revision: 'main', expected_bytes: 4096,
    }));
    expect(addJob).toHaveBeenCalledWith(job);
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
