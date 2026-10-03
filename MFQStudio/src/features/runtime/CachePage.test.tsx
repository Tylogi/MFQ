import { render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { CachePage } from './CachePage';

vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => ({ runtime: null }) }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));
vi.mock('./ResourceMonitorPanel', () => ({ ResourceMonitorPanel: () => <h2>Resource monitoring</h2> }));

it('resources contains monitoring and cache but no service configuration', () => {
  render(<CachePage />);
  expect(screen.getByRole('heading', { name: 'Resource monitoring' })).toBeInTheDocument();
  expect(screen.queryByText('Runtime profiles')).not.toBeInTheDocument();
  expect(screen.queryByText('Tool servers and model-visible tools')).not.toBeInTheDocument();
});
