import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { openStudioExternal } from '../../shared/platform/studio';
import { ApplicationsPage } from './ApplicationsPage';
import { applicationConfiguration } from './configuration';

const tr = (_zh: string, en: string) => en;
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => ({ instances: [{ model: 'checkpoint', state: 'ready' }], selectedModel: 'checkpoint', connectionRevision: 1, studio: null }) }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr }) }));
vi.mock('../../shared/platform/studio', () => ({ openStudioExternal: vi.fn() }));

beforeEach(() => {
  vi.restoreAllMocks();
  vi.spyOn(runtimeApi, 'modelAliases').mockResolvedValue({ aliases: { checkpoint: 'alias' } });
  vi.spyOn(runtimeApi, 'runtimeListener').mockResolvedValue({ host: '127.0.0.1', port: 8090, anthropic_port: 8991, configurable: true });
  Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: vi.fn().mockResolvedValue(undefined) } });
});

it('shows five configuration snippets without application management actions', async () => {
  render(<ApplicationsPage />);
  await waitFor(() => expect(screen.getByRole('combobox')).toHaveValue('alias'));
  expect(screen.getAllByRole('button', { name: 'Copy configuration' })).toHaveLength(5);
  expect(screen.getByText('http://127.0.0.1:8991/v1/messages')).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: /install|configure and connect|download|launch/i })).not.toBeInTheDocument();
});

it('copies a snippet but neither launches an app nor calls a write API', async () => {
  render(<ApplicationsPage />);
  await waitFor(() => expect(screen.getByRole('combobox')).toHaveValue('alias'));
  fireEvent.click(screen.getAllByRole('button', { name: 'Copy configuration' })[1]);
  await waitFor(() => expect(navigator.clipboard.writeText).toHaveBeenCalledOnce());
  const config = JSON.parse(vi.mocked(navigator.clipboard.writeText).mock.calls[0][0]);
  expect(config.model).toBe('mfq/alias');
  expect(config.provider.mfq.options.apiKey).toBe('YOUR_MFQ_API_KEY');
  expect(openStudioExternal).not.toHaveBeenCalled();
});

it('keeps SDK root inside Claude configuration and full request URL in the page', async () => {
  render(<ApplicationsPage />);
  await waitFor(() => expect(screen.getByRole('combobox')).toHaveValue('alias'));
  fireEvent.click(screen.getAllByRole('button', { name: 'Copy configuration' })[2]);
  await waitFor(() => expect(navigator.clipboard.writeText).toHaveBeenCalledOnce());
  expect(JSON.parse(vi.mocked(navigator.clipboard.writeText).mock.calls[0][0]).env.ANTHROPIC_BASE_URL).toBe('http://127.0.0.1:8991');
  expect(screen.getByText('http://127.0.0.1:8991/v1/messages')).toBeInTheDocument();
});

it('escapes model IDs as data in JSON and YAML', () => {
  const model = 'name"\nnot: a new property';
  const opencode = JSON.parse(applicationConfiguration('opencode', model, 'http://127.0.0.1:8090/v1', null));
  expect(opencode.model).toBe(`mfq/${model}`);
  expect(applicationConfiguration('hermes', model, 'http://127.0.0.1:8090/v1', null)).toContain(JSON.stringify(model));
  expect(applicationConfiguration('claude', model, '', null)).toBe('');
});
