import { useState } from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import type { ModelCacheProfile, RuntimeInstance } from '../../shared/api/types';
import { QsaKvOffloadPanel } from './QsaKvOffloadPanel';

const state = vi.hoisted(() => ({ language: 'en' }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (zh: string, en: string) => state.language === 'zh-CN' ? zh : en }) }));
const model = { id: 'qsa', model: 'Qwen3.8-Flash-Next', state: 'ready', qsa_kv_offload_supported: true } as RuntimeInstance;
const profile: ModelCacheProfile = { max_context: 262144, fixed_bytes: 29978,
  fixed_components: [{ group: 'QSA', name: 'indexer_tail', bytes: 29978 }], components: [
    { group: 'QSA', name: 'raw_kv', layers: 13, bytes_per_row: 26624, tokens_per_row: 1,
      allocation: 'power_of_two', minimum_rows: 16, max_read_rows_per_token: 2048 },
    { group: 'QSA', name: 'indexer_pooled', subgroup: 'indexer', bytes_per_row: 6656, tokens_per_row: 4,
      allocation: 'power_of_two', minimum_rows: 16, row_rounding: 'floor' },
  ] };

function Controls({ context = 678920, supported = true, metadata = profile as ModelCacheProfile | null, disabled = false }) {
  const [enabled, setEnabled] = useState(false);
  const [budget, setBudget] = useState('2');
  return <QsaKvOffloadPanel model={{ ...model, qsa_kv_offload_supported: supported }} context={context} enabled={enabled}
    budget={budget} profile={metadata} disabled={disabled} onEnabledChange={setEnabled} onBudgetChange={setBudget} />;
}

beforeEach(() => { state.language = 'en'; });

it('is opt-in and collapsed, with no separate model selector or save action', () => {
  render(<Controls />);
  expect(screen.getByRole('checkbox', { name: 'Enable streaming sparse attention' })).not.toBeChecked();
  expect(screen.getByRole('button', { name: 'Streaming Sparse Attention' })).toHaveAttribute('aria-expanded', 'false');
  expect(screen.queryByRole('spinbutton')).not.toBeInTheDocument();
  expect(screen.queryByRole('combobox')).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Apply settings' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Save to this model' })).not.toBeInTheDocument();
});

it('expands controlled drafts without submitting a server request', () => {
  const fetch = vi.spyOn(globalThis, 'fetch');
  render(<Controls />);
  fireEvent.click(screen.getByRole('checkbox'));
  expect(screen.getByRole('checkbox')).toBeChecked();
  expect(screen.getByRole('button')).toHaveAttribute('aria-expanded', 'true');
  expect(screen.getByRole('status', { name: 'Maximum context' })).toHaveTextContent('678,920');
  expect(screen.getAllByRole('spinbutton')).toHaveLength(1);
  expect(fetch).not.toHaveBeenCalled();
  fetch.mockRestore();
});

it('disables streaming controls for unsupported models', () => {
  render(<Controls supported={false} />);
  expect(screen.getByRole('checkbox')).toBeDisabled();
  fireEvent.click(screen.getByRole('button'));
  expect(screen.getByRole('spinbutton')).toBeDisabled();
  expect(screen.getByText(/Requires a QSA model/)).toBeInTheDocument();
  expect(screen.queryByText('Maximum raw KV reads/token')).not.toBeInTheDocument();
});

it('updates all-layer estimates from budget drafts without changing the fixed raw read maximum', () => {
  render(<Controls />);
  fireEvent.click(screen.getByRole('button'));
  const values = () => [...screen.getByText('Indexer limit estimate').parentElement!.parentElement!.querySelectorAll('b')].map(item => item.textContent);
  expect(values()).toEqual(['1.05 GiB', '906.58 MiB', '0 MiB', '52 MiB']);
  fireEvent.change(screen.getByRole('spinbutton'), { target: { value: '1' } });
  expect(values()).toEqual(['1 GiB', '0 MiB', '53.42 MiB', '52 MiB']);
  fireEvent.change(screen.getByRole('spinbutton'), { target: { value: '' } });
  expect(values()).toEqual(['—', '—', '—', '—']);
});

it('follows the parent model context draft rather than a saved offload snapshot', () => {
  const view = render(<Controls />);
  fireEvent.click(screen.getByRole('button'));
  view.rerender(<Controls context={524288} />);
  expect(screen.getByRole('status')).toHaveTextContent('524,288');
  expect(screen.getByText('Indexer limit estimate').parentElement).toHaveTextContent('832.03 MiB');
});

it('does not fabricate read estimates when cache metadata is missing', () => {
  render(<Controls metadata={null} />);
  fireEvent.click(screen.getByRole('button'));
  const values = screen.getByText('Indexer limit estimate').parentElement!.parentElement!.querySelectorAll('b');
  expect([...values].map(item => item.textContent)).toEqual(['—', '—', '—', '—']);
});

it('has ordered Chinese metric labels and remains read-only while the model is saving', () => {
  state.language = 'zh-CN';
  render(<Controls disabled />);
  fireEvent.click(screen.getByRole('button'));
  const estimate = screen.getByText('Indexer 上限估算').parentElement!.parentElement!;
  expect([...estimate.querySelectorAll('span')].map(item => item.textContent)).toEqual([
    'Indexer 上限估算', '原始 KV 常驻下界', '最大 Indexer 缓存读取量/token', '最大原始 KV 读取量/token',
  ]);
  expect(screen.getByRole('checkbox')).toBeDisabled();
  expect(screen.getByRole('spinbutton')).toBeDisabled();
});
