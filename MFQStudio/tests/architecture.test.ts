/** 验证应用入口、页面与共享层的依赖边界，防止重新出现根组件业务堆积。 */
import { readFileSync, readdirSync } from 'node:fs';
import { join, resolve, sep } from 'node:path';
import { describe, expect, it } from 'vitest';

const root = resolve('src');
function source(path: string) { return readFileSync(join(root, path), 'utf8'); }
function files(path: string): string[] {
  return readdirSync(path, { withFileTypes: true }).flatMap((entry) => entry.isDirectory() ? files(join(path, entry.name)) : /\.tsx?$/.test(entry.name) && !entry.name.includes('.test.') ? [join(path, entry.name)] : []);
}

describe('业务职责边界', () => {
  it('App仅组织Provider和路由，不拥有请求、业务状态或业务表单', () => {
    const app = source('App.tsx');
    expect(app).not.toMatch(/\buseState\b|\buseEffect\b|\bapi\.|\bfetch\(|<form\b|<input\b/);
    expect(app.split('\n').length).toBeLessThan(100);
    for (const route of ['chat', 'models', 'runtime', 'resources', 'logs', 'settings', 'model-hub', 'evaluations', 'quantization']) {
      expect(app).toContain(`path="${route}"`);
    }
  });
  it('共享传输层不反向引用React、页面或业务状态', () => {
    for (const file of files(join(root, 'shared/api'))) {
      expect(readFileSync(file, 'utf8')).not.toMatch(/from ['"][^'"]*(?:features\/|App|react|zustand)/);
    }
  });
  it('业务模块不能依赖App，运行时启动不请求页面专属资源', () => {
    for (const file of files(join(root, 'features'))) expect(readFileSync(file, 'utf8')).not.toMatch(/from ['"][^'"]*\/App['"]/);
    const runtime = source('app/RuntimeProvider.tsx');
    for (const request of ['datasets', 'evaluations', 'modelArtifacts', 'runtimeProfiles', 'runtimeLogs', 'generationPresets', 'mcpServers', 'artifactLineage']) {
      expect(runtime).not.toContain(`.${request}(`);
    }
  });
  it('业务模块按领域依赖资源 API，不依赖统一聚合入口', () => {
    for (const file of files(root)) {
      const relative = file.replace(`${root}${sep}`, '').replaceAll(sep, '/');
      expect(relative).not.toBe('api.ts');
      expect(readFileSync(file, 'utf8')).not.toMatch(/from ['"][^'"]*\/api['"]/);
      expect(readFileSync(file, 'utf8')).not.toMatch(/\bapi\.[A-Za-z]/);
    }
  });
  it('全局样式入口只保留有序导入，各业务规则在独立样式中', () => {
    const styles = source('styles.css');
    expect(styles).not.toContain('{');
    expect(styles).toContain('features/chat/chat.css');
    expect(styles).toContain('shared/styles/overrides.css');
  });
});
