/** 验证深链接、路由生成与回退行为在页面拆分后保持一致。 */
import { describe, expect, it } from 'vitest';
import {
  dashboardPath,
  isStudioPath,
  labPath,
  normalizeStudioPath,
  resolveStudioLocation,
} from '../src/navigation';

describe('Studio 页面路由', () => {
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
  ])('解析深链接 %s', (path, view, page) => {
    const location = resolveStudioLocation(path);
    expect(location.view).toBe(view);
    expect(view === 'lab' ? location.labPage : location.dashboardPage).toBe(page);
  });

  it('生成的业务路径可回读，兼容尾部斜杠', () => {
    expect(resolveStudioLocation(`${dashboardPath('connections')}/`).dashboardPage).toBe(
      'connections',
    );
    expect(resolveStudioLocation(`${labPath('evaluations')}///`).labPage).toBe('evaluations');
    expect(normalizeStudioPath('/')).toBe('/');
  });

  it('未知地址不会被视为业务页面', () => {
    expect(isStudioPath('/chat')).toBe(true);
    expect(isStudioPath('/chat/')).toBe(true);
    expect(isStudioPath('/missing')).toBe(false);
    expect(isStudioPath('/chat/extra')).toBe(false);
  });
});
