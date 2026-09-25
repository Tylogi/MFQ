/** 验证外壳在已就绪和可直达页面展示持续故障与正确重试入口。 */
import { fireEvent, render, screen, within } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { useRuntime } from './RuntimeProvider';
import { StudioShell } from './StudioShell';
import { NotFoundPage } from './NotFoundPage';

vi.mock('./RuntimeProvider', () => ({ useRuntime: vi.fn() }));
vi.mock('../features/settings/SettingsProvider', () => ({
  useSettings: () => ({ tr: (chinese: string) => chinese }),
}));

/** 挂载可直达路由，模拟各类共享运行时状态。 */
function renderShell(overrides: Partial<ReturnType<typeof useRuntime>>, path = '/settings') {
  const runtime = {
    runtime: null,
    selectedModel: '',
    models: [],
    instances: [],
    loading: false,
    connectionError: null,
    refreshError: null,
    jobStreamErrors: {},
    ready: true,
    reloadService: vi.fn(),
    refreshRuntime: vi.fn(),
    retryJobStreams: vi.fn(),
    ...overrides,
  } as unknown as ReturnType<typeof useRuntime>;
  vi.mocked(useRuntime).mockReturnValue(runtime);
  render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route element={<StudioShell />}>
          <Route path="settings" element={<p>Settings content</p>} />
          <Route path="runtime" element={<p>Connection settings content</p>} />
          <Route path="chat" element={<p>Chat content</p>} />
          <Route index element={<p>Overview content</p>} />
          <Route path="*" element={<NotFoundPage />} />
        </Route>
      </Routes>
    </MemoryRouter>,
  );
  return runtime;
}

describe('StudioShell 持久运行时告警', () => {
  beforeEach(() => vi.clearAllMocks());

  it('未就绪时设置页仍可打开，显示连接故障和重试', () => {
    const runtime = renderShell({ ready: false, connectionError: 'server offline' });
    expect(screen.getByText('Settings content')).toBeTruthy();
    expect(within(screen.getByRole('alert')).getByText('server offline')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: '重试连接' }));
    expect(runtime.reloadService).toHaveBeenCalledOnce();
  });

  it('运行时刷新与任务流错误分开展示，并分别重试', () => {
    const runtime = renderShell({
      refreshError: 'refresh offline',
      jobStreamErrors: { 'job-1': 'stream offline' },
    });
    expect(screen.getAllByRole('alert')).toHaveLength(2);
    expect(screen.getByText('refresh offline')).toBeTruthy();
    expect(screen.getByText('job-1: stream offline')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: '重新刷新' }));
    expect(runtime.refreshRuntime).toHaveBeenCalledWith(false);
    fireEvent.click(screen.getByRole('button', { name: '重新连接任务流' }));
    expect(runtime.retryJobStreams).toHaveBeenCalledOnce();
  });

  it('未就绪的业务页面显示独立断连页，允许重试或打开连接设置', () => {
    const runtime = renderShell({ ready: false, connectionError: 'server offline' }, '/chat');
    expect(screen.getByRole('heading', { name: '服务暂时无法连接' })).toBeTruthy();
    expect(screen.queryByText('Chat content')).toBeNull();
    fireEvent.click(screen.getByText('查看错误详情'));
    expect(screen.getByText('server offline')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: '重新连接' }));
    expect(runtime.reloadService).toHaveBeenCalledOnce();
    fireEvent.click(screen.getByRole('button', { name: '连接设置' }));
    expect(screen.getByText('Connection settings content')).toBeTruthy();
  });

  it('首次连接中只显示等待状态，不误报失败', () => {
    renderShell({ ready: false }, '/chat');
    expect(screen.getByRole('status')).toHaveTextContent('正在加载中');
    expect(screen.getByRole('heading', { name: '正在加载中' })).toBeTruthy();
    expect(screen.queryByRole('heading', { name: '服务暂时无法连接' })).toBeNull();
  });

  it('未知地址即使服务未就绪也显示 404，返回概览后显示连接状态', () => {
    renderShell({ ready: false, connectionError: 'server offline' }, '/missing');
    expect(screen.getByRole('heading', { name: '页面不存在' })).toBeTruthy();
    expect(screen.queryByRole('heading', { name: '服务暂时无法连接' })).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: '返回概览' }));
    expect(screen.getByRole('heading', { name: '服务暂时无法连接' })).toBeTruthy();
  });

  it('页面渲染异常显示独立故障页并允许返回概览', () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    function CrashingPage(): never {
      throw new Error('page render failed');
    }
    vi.mocked(useRuntime).mockReturnValue({
      runtime: null, selectedModel: '', models: [], instances: [], loading: false,
      connectionError: null, refreshError: null, jobStreamErrors: {}, ready: true,
      reloadService: vi.fn(), refreshRuntime: vi.fn(), retryJobStreams: vi.fn(),
    } as unknown as ReturnType<typeof useRuntime>);
    render(
      <MemoryRouter initialEntries={['/chat']}>
        <Routes>
          <Route element={<StudioShell />}>
            <Route path="chat" element={<CrashingPage />} />
            <Route index element={<p>Overview content</p>} />
          </Route>
        </Routes>
      </MemoryRouter>,
    );
    expect(screen.getByRole('heading', { name: '页面暂时无法显示' })).toBeTruthy();
    fireEvent.click(screen.getByText('查看错误详情'));
    expect(screen.getByText('page render failed')).toBeTruthy();
    expect(screen.getByRole('button', { name: '重试页面' })).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: '返回概览' }));
    expect(screen.getByText('Overview content')).toBeTruthy();
    vi.restoreAllMocks();
  });
});
