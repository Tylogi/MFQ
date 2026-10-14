import { useState } from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { ModelLoadContext } from './ModelLoadContext';
import { DEFAULT_LOAD_CONTEXT, readLoadContext, saveLoadContext } from './loadContextDefaults';
import { setApiBaseUrl } from '../../shared/api/client';

vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (zh: string) => zh }) }));

beforeEach(() => { localStorage.clear(); setApiBaseUrl(''); });
function Form() {
  const [value, setValue] = useState(readLoadContext);
  return <ModelLoadContext value={value} onChange={next => { setValue(next); saveLoadContext(next); }} />;
}

it('defaults to 32768 with collapsed automatic settings and remembers explicit context', () => {
  const view = render(<Form />);
  const context = screen.getByRole('spinbutton', { name: '默认上下文' });
  expect(context).toHaveAttribute('placeholder', '32768');
  expect(context).toHaveValue(null);
  expect(screen.getByRole('button', { name: '高级' })).toHaveAttribute('aria-expanded', 'false');
  expect(screen.getByRole('checkbox', { name: '自动' })).toBeChecked();
  fireEvent.change(context, { target: { value: '65536' } });
  view.unmount(); render(<Form />);
  expect(screen.getByRole('spinbutton')).toHaveValue(65536);
  fireEvent.change(screen.getByRole('spinbutton'), { target: { value: '' } });
  expect(readLoadContext().context_size).toBeNull();
});

it('enforces the sparse RAM/SSD dependency and keeps non-sparse KV in VRAM', () => {
  render(<Form />);
  fireEvent.click(screen.getByRole('button', { name: '高级' }));
  expect(screen.getByRole('checkbox', { name: 'SSD KV' })).toBeDisabled();
  expect(screen.getByRole('combobox', { name: '默认 KV 量化档位' })).toBeDisabled();
  fireEvent.click(screen.getByRole('checkbox', { name: '自动' }));
  fireEvent.click(screen.getByRole('checkbox', { name: 'RAM 预分配' }));
  expect(screen.getByRole('checkbox', { name: 'SSD KV' })).toBeChecked();
  fireEvent.click(screen.getByRole('checkbox', { name: 'SSD KV' }));
  expect(screen.getByRole('checkbox', { name: 'RAM 预分配' })).toBeChecked();
  fireEvent.click(screen.getByRole('button', { name: '非稀疏注意力模型' }));
  expect(screen.queryByRole('checkbox', { name: 'SSD KV' })).not.toBeInTheDocument();
  expect(screen.queryByText(/SSD KV/)).not.toBeInTheDocument();
  expect(screen.queryByRole('checkbox', { name: 'RAM 预分配' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('checkbox', { name: 'VRAM 预分配' }));
  expect(readLoadContext().vram_preallocated).toBe(false);
  fireEvent.change(screen.getByRole('combobox'), { target: { value: '2.5' } });
  expect(readLoadContext().kv_quantization).toEqual({ enabled: true, bits: 2.5, algorithm: 'turboquant' });
  fireEvent.click(screen.getByRole('checkbox', { name: '自动' }));
  expect(screen.getByRole('combobox')).toHaveValue('off');
  expect(screen.getByRole('checkbox', { name: 'VRAM 预分配' })).toBeChecked();
  fireEvent.click(screen.getByRole('checkbox', { name: '自动' }));
  expect(screen.getByRole('combobox')).toHaveValue('2.5');
});

it('isolates stored defaults by server and tolerates blocked storage', () => {
  setApiBaseUrl('http://127.0.0.1:8090');
  saveLoadContext({ ...DEFAULT_LOAD_CONTEXT, context_size: 8192 });
  setApiBaseUrl('http://127.0.0.1:8092');
  expect(readLoadContext()).toEqual(DEFAULT_LOAD_CONTEXT);
  const write = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('blocked'); });
  expect(() => saveLoadContext(DEFAULT_LOAD_CONTEXT)).not.toThrow();
  write.mockRestore();
  setApiBaseUrl('http://127.0.0.1:8090');
  expect(readLoadContext().context_size).toBe(8192);
});
