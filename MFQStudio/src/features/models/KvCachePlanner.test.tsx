import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import type { ModelCacheProfile } from '../../shared/api/types';
import { cacheUsageBytes, estimateCacheBytes, KvCachePlanner } from './KvCachePlanner';

const profile: ModelCacheProfile = {
  max_context: 10000, fixed_bytes: 1024,
  components: [{ bytes_per_row: 16, tokens_per_row: 1, minimum_rows: 16, allocation: 'power_of_two' },
    { bytes_per_row: 32, tokens_per_row: 4, minimum_rows: 0, allocation: 'exact' }],
};

it('accounts for geometric growth, compressed rows, fixed state and the model limit', () => {
  expect(estimateCacheBytes(profile, 16)).toBe(1024 + 16 * 16 + 4 * 32);
  expect(estimateCacheBytes(profile, 17)).toBe(1024 + 32 * 16 + 5 * 32);
  expect(estimateCacheBytes(profile, 10000)).toBe(1024 + 10000 * 16 + 2500 * 32);
  expect(estimateCacheBytes(profile, 20000)).toBe(estimateCacheBytes(profile, 10000));
  expect(estimateCacheBytes({ max_context: 2048, fixed_bytes: 100, components: [] }, 2048)).toBe(100);
});

it('plots continuous theoretical usage without changing the preallocation budget', () => {
  expect(cacheUsageBytes(profile, 16)).toBe(1024 + 16 * 24);
  expect(cacheUsageBytes(profile, 17)).toBe(1024 + 17 * 24);
  expect(cacheUsageBytes(profile, 18) - cacheUsageBytes(profile, 17)).toBe(24);
  expect(estimateCacheBytes(profile, 17)).toBeGreaterThan(cacheUsageBytes(profile, 17));
  expect(cacheUsageBytes(profile, 20000)).toBe(cacheUsageBytes(profile, 10000));
});

it('shows exact ctx and GiB on hover using the logarithmic x axis without applying a plan', () => {
  const onApply = vi.fn();
  const graphProfile: ModelCacheProfile = { max_context: 16384, fixed_bytes: 2 ** 30,
    components: [{ bytes_per_row: 2 ** 18, tokens_per_row: 1, minimum_rows: 16, allocation: 'power_of_two' }] };
  render(<KvCachePlanner profile={graphProfile} onApply={onApply} tr={(_, en) => en} />);
  fireEvent.click(screen.getByText('KV Cache curve & calculator'));
  const graph = screen.getByRole('img', { name: 'KV Cache memory by context length' });
  vi.spyOn(graph, 'getBoundingClientRect').mockReturnValue({ left: 0, top: 0, width: 460, height: 174 } as DOMRect);
  fireEvent.mouseMove(graph, { clientX: 249, clientY: 90 });
  expect(screen.getByRole('tooltip')).toHaveTextContent('4,096 ctx');
  expect(screen.getByRole('tooltip')).toHaveTextContent('2.000 GiB');
  fireEvent.mouseMove(graph, { clientX: 151.5, clientY: 90 });
  expect(screen.getByRole('tooltip')).toHaveTextContent('2,048 ctx');
  expect(screen.getByRole('tooltip')).toHaveTextContent('1.500 GiB');
  expect(screen.getByRole('spinbutton')).toHaveValue(4096);
  expect(onApply).not.toHaveBeenCalled();
  fireEvent.click(graph, { clientX: 151.5, clientY: 90 });
  expect(screen.getByRole('spinbutton')).toHaveValue(2048);
  fireEvent.mouseLeave(graph);
  expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
  fireEvent.mouseMove(graph, { clientX: 20, clientY: 90 });
  expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
});

it('selects clickable power-of-two ticks and applies only when explicitly confirmed', () => {
  const onApply = vi.fn();
  render(<KvCachePlanner profile={profile} onApply={onApply} tr={(_, en) => en} />);
  fireEvent.click(screen.getByText('KV Cache curve & calculator'));
  for (const tokens of [1024, 2048, 4096, 8192]) expect(screen.getByRole('button', { name: `ctx ${tokens}` })).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'ctx 16384' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'ctx 8192' }));
  expect(screen.getByRole('spinbutton')).toHaveValue(8192);
  expect(screen.getByRole('button', { name: 'ctx 8192' })).toHaveAttribute('aria-pressed', 'true');
  expect(onApply).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Apply' }));
  expect(onApply).toHaveBeenCalledWith(8192);
});

it('starts collapsed, previews without applying and rejects invalid ctx', () => {
  const onApply = vi.fn();
  render(<KvCachePlanner profile={profile} onApply={onApply} tr={(_, en) => en} />);
  const summary = screen.getByText('KV Cache curve & calculator');
  expect(summary.closest('details')!.open).toBe(false);
  fireEvent.click(summary);
  expect(summary.closest('details')!.open).toBe(true);
  expect(screen.getByRole('img', { name: 'KV Cache memory by context length' })).toBeInTheDocument();
  fireEvent.change(screen.getByRole('spinbutton', { name: 'Expected ctx' }), { target: { value: '8192' } });
  expect(onApply).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Apply' }));
  expect(onApply).toHaveBeenCalledWith(8192);
  for (const value of ['', '0', '10001', '1.5']) {
    fireEvent.change(screen.getByRole('spinbutton'), { target: { value } });
    expect(screen.getByRole('button', { name: 'Apply' })).toBeDisabled();
    expect(screen.getByRole('alert')).toBeInTheDocument();
  }
  expect(onApply).toHaveBeenCalledTimes(1);
});

it('allows clearing an applied plan and explains unavailable model metadata', () => {
  const onApply = vi.fn();
  const view = render(<KvCachePlanner profile={profile} appliedContext={4096} onApply={onApply} tr={(_, en) => en} />);
  fireEvent.click(screen.getByText('KV Cache curve & calculator'));
  fireEvent.click(screen.getByRole('button', { name: 'Clear' }));
  expect(onApply).toHaveBeenCalledWith(undefined);
  view.rerender(<KvCachePlanner onApply={onApply} tr={(_, en) => en} />);
  expect(screen.getByText('Insufficient cache structure metadata for a reliable estimate.')).toBeInTheDocument();
  expect(screen.queryByRole('spinbutton')).not.toBeInTheDocument();
});
