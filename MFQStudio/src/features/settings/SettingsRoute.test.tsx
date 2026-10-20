import { fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { beforeEach, expect, it, vi } from 'vitest';
import { presetsApi } from '../../shared/api/resources/presets';
import { DEFAULT_SETTINGS, SETTINGS_KEY } from './configuration';
import { SettingsProvider } from './SettingsProvider';
import { SettingsRoute } from './SettingsRoute';

vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => ({ connectionRevision: 0, selectedModel: '', studio: null }) }));
vi.mock('../chat/hooks/useActiveSessionMode', () => ({ useActiveSessionMode: () => 'text' }));
vi.mock('../../stores/toastStore', () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

beforeEach(() => {
  vi.restoreAllMocks();
  localStorage.clear();
  localStorage.setItem(SETTINGS_KEY, JSON.stringify({ ...DEFAULT_SETTINGS, language: 'en', theme: 'dark', enableVision: false,
    inheritModelDefaults: false, temperature: 0.37, systemPrompt: 'Existing client preference' }));
  vi.spyOn(presetsApi, 'generationPresets').mockResolvedValue([]);
});

function mount() {
  render(<MemoryRouter><SettingsProvider><SettingsRoute /></SettingsProvider></MemoryRouter>);
}

it('shows appearance only, without data, connection or generation entry points', () => {
  mount();
  expect(screen.getByText('Appearance', { exact: true })).toBeInTheDocument();
  for (const text of ['Data and connection', 'Local settings data', 'Server connection', 'Generation defaults', 'Presets and prompt', 'Generation preset', 'Sampling', 'Temperature', 'Chat MTP', 'Vision input']) {
    expect(screen.queryByText(text, { exact: true })).not.toBeInTheDocument();
  }
  expect(screen.queryByRole('switch')).not.toBeInTheDocument();
  expect(screen.queryByRole('slider')).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Export' })).not.toBeInTheDocument();
  expect(screen.queryByText('Import')).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Open server settings' })).not.toBeInTheDocument();
  expect(presetsApi.generationPresets).not.toHaveBeenCalled();
});

it('applies appearance without overwriting inference preferences', () => {
  mount();
  fireEvent.change(screen.getByLabelText('Theme'), { target: { value: 'light' } });
  fireEvent.click(screen.getAllByRole('button', { name: 'Apply settings' })[0]);
  const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY)!);
  expect(saved.theme).toBe('light');
  expect(saved.enableVision).toBe(false);
  expect(saved.inheritModelDefaults).toBe(false);
  expect(saved.temperature).toBe(0.37);
  expect(saved.systemPrompt).toBe('Existing client preference');
});

it('resets appearance without resetting hidden generation values', () => {
  mount();
  fireEvent.click(screen.getByRole('button', { name: 'Reset' }));
  fireEvent.click(screen.getAllByRole('button', { name: 'Apply settings' })[0]);
  const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY)!);
  expect(saved.theme).toBe('system');
  expect(saved.language).toBe('system');
  expect(saved.enableVision).toBe(false);
  expect(saved.temperature).toBe(0.37);
});
