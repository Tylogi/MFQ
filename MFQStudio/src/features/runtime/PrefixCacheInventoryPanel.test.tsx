import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { PrefixCacheInventoryPanel } from './PrefixCacheInventoryPanel';
import { ApiError } from '../../shared/api/client';
import { toast } from '../../stores/toastStore';

const api = vi.hoisted(() => ({ entries: vi.fn(), text: vi.fn(), purge: vi.fn(), confirm: vi.fn(), refresh: vi.fn() }));
const state = vi.hoisted(() => ({ connectionRevision: 0 }));
vi.mock('../../shared/api/resources/runtime', () => ({ runtimeApi: { prefixCacheEntries: api.entries, prefixCacheText: api.text, purgePrefixCache: api.purge } }));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => ({ ...state, refreshRuntime: api.refresh }) }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));
vi.mock('../../studio', () => ({ studioConfirm: api.confirm }));
vi.mock('../../stores/toastStore', () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

const inventory = { directory: '/cache/prefix', total_bytes: 4096, total_blocks: 1, can_clear: true,
  offset: 0, limit: 100, data: [{ id: 'a', model_name: 'Qwen-Test', blocks: 1, bytes: 4096,
    context_size: 8192, text_blocks: 1, last_used_at: '2026-10-06T00:00:00Z', max_prefix_tokens: 256 }],
  blocks: [{ id: 'b', bytes: 4096, prefix_tokens: 256, complete_chain: true, text_available: true }] };

beforeEach(() => {
  vi.clearAllMocks();
  state.connectionRevision = 0;
  api.entries.mockResolvedValue(inventory);
  api.text.mockResolvedValue({ available: true, text: 'Original prompt and generated reply', total_tokens: 256, offset: 0, next_offset: null });
  api.confirm.mockResolvedValue(true);
  api.purge.mockResolvedValue({ removed_blocks: 1, released_bytes: 4096, failed_blocks: 0 });
});

it('shows the actual directory and previews saved original tokens as text', async () => {
  render(<PrefixCacheInventoryPanel />);
  await screen.findByText('/cache/prefix');
  fireEvent.click(screen.getByRole('button', { name: 'Inspect' }));
  await screen.findByRole('button', { name: 'View text' });
  fireEvent.click(screen.getByRole('button', { name: 'View text' }));
  await screen.findByText('Original prompt and generated reply');
  expect(api.text).toHaveBeenCalledWith('a', 'b', 0);
});

it.each([
  ['tokenizer_unavailable', 'The model or tokenizer is unavailable'],
  ['text_invalid', 'Text record integrity check failed'],
  ['incomplete_chain', 'Earlier prefix blocks have been reclaimed'],
  ['cache_not_found', 'This cache block has been reclaimed or cleared'],
  ['invalid_cache_id', 'Invalid cache identifier'],
  ['prefix_too_large', 'This prefix is too large to display'],
  ['text_not_saved', 'This older cache has no saved text record'],
  ['new_reason', 'Cache text is unavailable'],
])('explains unavailable text reason %s without implying the cache never saved text', async (reason, message) => {
  api.text.mockResolvedValue({ available: false, reason });
  render(<PrefixCacheInventoryPanel />);
  await screen.findByText('Qwen-Test');
  fireEvent.click(screen.getByRole('button', { name: 'Inspect' }));
  await screen.findByRole('button', { name: 'View text' });
  fireEvent.click(screen.getByRole('button', { name: 'View text' }));
  await screen.findByText(message);
});

it('clears an individual cache group only after confirmation', async () => {
  render(<PrefixCacheInventoryPanel />);
  await screen.findByText('Qwen-Test');
  fireEvent.click(screen.getByRole('button', { name: 'Clear' }));
  await waitFor(() => expect(api.purge).toHaveBeenCalledWith('a'));
  expect(api.confirm).toHaveBeenCalled();
  await waitFor(() => expect(api.refresh).toHaveBeenCalledWith(false));
});

it('cancelled clear-all leaves cache unchanged', async () => {
  api.confirm.mockResolvedValue(false);
  render(<PrefixCacheInventoryPanel />);
  await screen.findByText('Qwen-Test');
  fireEvent.click(screen.getByRole('button', { name: 'Clear all' }));
  await waitFor(() => expect(api.confirm).toHaveBeenCalled());
  expect(api.purge).not.toHaveBeenCalled();
});

it('inference in progress disables both clearing actions', async () => {
  api.entries.mockResolvedValue({ ...inventory, can_clear: false });
  render(<PrefixCacheInventoryPanel />);
  await screen.findByText('Qwen-Test');
  expect(screen.getByRole('button', { name: 'Clear all' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Clear' })).toBeDisabled();
});

it('does not restore purged cache groups when an older refresh resolves last', async () => {
  let resolve!: (value: typeof inventory) => void;
  api.entries.mockResolvedValueOnce(inventory)
    .mockReturnValueOnce(new Promise((done) => { resolve = done; }))
    .mockResolvedValue({ ...inventory, total_bytes: 0, total_blocks: 0, data: [], blocks: [] });
  render(<PrefixCacheInventoryPanel />);
  await screen.findByText('Qwen-Test');
  fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
  fireEvent.click(screen.getByRole('button', { name: 'Clear all' }));
  await screen.findByText('No SSD caches');
  await act(async () => { resolve(inventory); });
  expect(screen.queryByText('Qwen-Test')).not.toBeInTheDocument();
  expect(screen.getByText('No SSD caches')).toBeInTheDocument();
});

it('does not report an older refresh failure after a newer successful refresh', async () => {
  let reject!: (cause: Error) => void;
  api.entries.mockResolvedValueOnce(inventory)
    .mockReturnValueOnce(new Promise((_resolve, fail) => { reject = fail; }))
    .mockResolvedValue({ ...inventory, total_bytes: 0, total_blocks: 0, data: [], blocks: [] });
  render(<PrefixCacheInventoryPanel />);
  await screen.findByText('Qwen-Test');
  fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
  fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
  await screen.findByText('No SSD caches');
  await act(async () => { reject(new Error('old refresh failed')); });
  expect(screen.queryByText('old refresh failed')).not.toBeInTheDocument();
});

it('keeps inspected cache text and allows retry when maintenance races a clear request', async () => {
  api.purge.mockRejectedValueOnce(new ApiError(409, { error: { code: 'runtime_busy',
    message: 'wait for inference and cache writes to finish before clearing cache', retryable: true, details: {} } }));
  render(<PrefixCacheInventoryPanel />);
  await screen.findByText('Qwen-Test');
  fireEvent.click(screen.getByRole('button', { name: 'Inspect' }));
  await screen.findByRole('button', { name: 'View text' });
  fireEvent.click(screen.getByRole('button', { name: 'View text' }));
  await screen.findByText('Original prompt and generated reply');
  fireEvent.click(screen.getByRole('button', { name: 'Clear all' }));
  await waitFor(() => expect(toast.error).toHaveBeenCalledWith('runtime_busy: wait for inference and cache writes to finish before clearing cache'));
  expect(screen.getByRole('button', { name: 'Clear all' })).toBeEnabled();
  expect(screen.getByText('Original prompt and generated reply')).toBeInTheDocument();
  expect(api.refresh).not.toHaveBeenCalled();
  api.entries.mockResolvedValue({ ...inventory, total_bytes: 0, total_blocks: 0, data: [], blocks: [] });
  fireEvent.click(screen.getByRole('button', { name: 'Clear all' }));
  await screen.findByText('No SSD caches');
  expect(api.purge).toHaveBeenCalledTimes(2);
  expect(screen.queryByText('Original prompt and generated reply')).not.toBeInTheDocument();
});

it('cannot purge caches after a pending confirmation outlives the page', async () => {
  let confirm!: (accepted: boolean) => void;
  api.confirm.mockReturnValue(new Promise((resolve) => { confirm = resolve; }));
  const view = render(<PrefixCacheInventoryPanel />);
  await screen.findByText('Qwen-Test');
  fireEvent.click(screen.getByRole('button', { name: 'Clear all' }));
  view.unmount();
  await act(async () => { confirm(true); });
  expect(api.purge).not.toHaveBeenCalled();
});

it('keeps other cache groups inspectable when clearing cancels an in-flight preview', async () => {
  let resolve!: (value: typeof inventory) => void;
  const other = { ...inventory, data: [{ ...inventory.data[0], id: 'other', model_name: 'Other model' }] };
  api.entries.mockResolvedValue(other)
    .mockResolvedValueOnce({ ...inventory, data: [...inventory.data, ...other.data] })
    .mockReturnValueOnce(new Promise((done) => { resolve = done; }));
  render(<PrefixCacheInventoryPanel />);
  await screen.findByText('Qwen-Test');
  fireEvent.click(screen.getAllByRole('button', { name: 'Inspect' })[0]);
  fireEvent.click(screen.getAllByRole('button', { name: 'Clear' })[0]);
  await waitFor(() => expect(api.refresh).toHaveBeenCalledWith(false));
  await waitFor(() => expect(screen.queryByText('Qwen-Test')).not.toBeInTheDocument());
  expect(screen.getByRole('button', { name: 'Inspect' })).toBeEnabled();
  await act(async () => { resolve(inventory); });
  expect(screen.queryByRole('button', { name: 'View text' })).not.toBeInTheDocument();
});

it('replaces the inventory and aborts old polling when changing servers', async () => {
  const view = render(<PrefixCacheInventoryPanel />);
  await screen.findByText('Qwen-Test');
  const signal = api.entries.mock.calls[0][2] as AbortSignal;
  api.entries.mockResolvedValue({ ...inventory, directory: '/new/prefix',
    data: [{ ...inventory.data[0], id: 'new', model_name: 'New model' }] });
  state.connectionRevision = 1;
  view.rerender(<PrefixCacheInventoryPanel />);
  await screen.findByText('New model');
  expect(signal.aborted).toBe(true);
  expect(screen.queryByText('Qwen-Test')).not.toBeInTheDocument();
  expect(screen.getByText('/new/prefix')).toBeInTheDocument();
});

it('ignores an old inspection result after switching to a different server', async () => {
  let resolve!: (value: typeof inventory) => void;
  api.entries.mockResolvedValueOnce(inventory)
    .mockReturnValueOnce(new Promise((done) => { resolve = done; }));
  const view = render(<PrefixCacheInventoryPanel />);
  await screen.findByText('Qwen-Test');
  fireEvent.click(screen.getByRole('button', { name: 'Inspect' }));
  state.connectionRevision = 1;
  api.entries.mockResolvedValue({ ...inventory, data: [] });
  view.rerender(<PrefixCacheInventoryPanel />);
  await act(async () => { resolve(inventory); });
  expect(screen.queryByRole('button', { name: 'View text' })).not.toBeInTheDocument();
  expect(screen.queryByText('Qwen-Test')).not.toBeInTheDocument();
});

it('never displays cached text retrieved from a previous server', async () => {
  let resolve!: (value: { available: boolean; text: string }) => void;
  api.text.mockReturnValueOnce(new Promise((done) => { resolve = done; }));
  const view = render(<PrefixCacheInventoryPanel />);
  await screen.findByText('Qwen-Test');
  fireEvent.click(screen.getByRole('button', { name: 'Inspect' }));
  await screen.findByRole('button', { name: 'View text' });
  fireEvent.click(screen.getByRole('button', { name: 'View text' }));
  state.connectionRevision = 1;
  api.entries.mockResolvedValue({ ...inventory, data: [] });
  view.rerender(<PrefixCacheInventoryPanel />);
  await act(async () => { resolve({ available: true, text: 'Old server text' }); });
  expect(screen.queryByText('Old server text')).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'View text' })).not.toBeInTheDocument();
});

it('keeps the new server usable while discarding an old clear confirmation', async () => {
  let confirm!: (accepted: boolean) => void;
  api.confirm.mockReturnValueOnce(new Promise((done) => { confirm = done; }));
  const view = render(<PrefixCacheInventoryPanel />);
  await screen.findByText('Qwen-Test');
  fireEvent.click(screen.getByRole('button', { name: 'Clear' }));
  state.connectionRevision = 1;
  api.entries.mockResolvedValue({ ...inventory, data: [{ ...inventory.data[0], id: 'new', model_name: 'New model' }] });
  view.rerender(<PrefixCacheInventoryPanel />);
  await screen.findByText('New model');
  expect(screen.getByRole('button', { name: 'Clear' })).toBeEnabled();
  fireEvent.click(screen.getByRole('button', { name: 'Clear' }));
  await waitFor(() => expect(api.purge).toHaveBeenCalledExactlyOnceWith('new'));
  await act(async () => { confirm(true); });
  expect(api.purge).toHaveBeenCalledExactlyOnceWith('new');
});
