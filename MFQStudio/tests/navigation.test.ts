/** Verify deep links, route generation, and fallback behavior remain consistent after splitting pages. */
import { describe, expect, it } from 'vitest';
import {
  dashboardPath,
  isStudioPath,
  labPath,
  normalizeStudioPath,
  resolveStudioLocation,
} from '../src/navigation';

describe('describes navigation test behavior 1', () => {
  it.each([
    ['/', 'dashboard', 'overview'],
    ['/models', 'dashboard', 'models'],
    ['/runtime', 'dashboard', 'connections'],
    ['/resources', 'dashboard', 'cache'],
    ['/logs', 'dashboard', 'logs'],
    ['/settings', 'dashboard', 'settings'],
    ['/model-hub', 'lab', 'models'],
    ['/evaluations', 'lab', 'evaluations'],
    ['/quantization', 'lab', 'quantization'],
    ['/chat', 'chat', 'overview'],
  ])('verifies parameterized behavior %s', (path, view, page) => {
    const location = resolveStudioLocation(path);
    expect(location.view).toBe(view);
    expect(view === 'lab' ? location.labPage : location.dashboardPage).toBe(page);
  });

  it('verifies navigation test behavior 2', () => {
    expect(resolveStudioLocation(`${dashboardPath('connections')}/`).dashboardPage).toBe(
      'connections',
    );
    expect(resolveStudioLocation(`${labPath('evaluations')}///`).labPage).toBe('evaluations');
    expect(normalizeStudioPath('/')).toBe('/');
  });

  it('verifies navigation test behavior 3', () => {
    expect(isStudioPath('/chat')).toBe(true);
    expect(isStudioPath('/chat/')).toBe(true);
    expect(isStudioPath('/missing')).toBe(false);
    expect(isStudioPath('/chat/extra')).toBe(false);
  });
});
