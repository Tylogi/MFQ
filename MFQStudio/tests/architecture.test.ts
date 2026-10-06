/** Verify dependency boundaries between the app entry, pages, and shared layer to prevent business logic from accumulating in the root component. */
import { readFileSync, readdirSync } from 'node:fs';
import { join, resolve, sep } from 'node:path';
import { describe, expect, it } from 'vitest';

const root = resolve('src');
function source(path: string) { return readFileSync(join(root, path), 'utf8'); }
function files(path: string): string[] {
  return readdirSync(path, { withFileTypes: true }).flatMap((entry) => entry.isDirectory() ? files(join(path, entry.name)) : /\.tsx?$/.test(entry.name) && !entry.name.includes('.test.') ? [join(path, entry.name)] : []);
}

describe('describes architecture test behavior 1', () => {
  it('verifies architecture test behavior 2', () => {
    const app = source('App.tsx');
    expect(app).not.toMatch(/\buseState\b|\buseEffect\b|\bapi\.|\bfetch\(|<form\b|<input\b/);
    expect(app.split('\n').length).toBeLessThan(100);
    for (const route of ['chat', 'models', 'runtime', 'resources', 'logs', 'settings', 'model-hub', 'evaluations', 'quantization']) {
      expect(app).toContain(`path="${route}"`);
    }
  });
  it('verifies architecture test behavior 3', () => {
    for (const file of files(join(root, 'shared/api'))) {
      expect(readFileSync(file, 'utf8')).not.toMatch(/from ['"][^'"]*(?:features\/|App|react|zustand)/);
    }
  });
  it('verifies architecture test behavior 4', () => {
    for (const file of files(join(root, 'features'))) expect(readFileSync(file, 'utf8')).not.toMatch(/from ['"][^'"]*\/App['"]/);
    const runtime = source('app/RuntimeProvider.tsx');
    for (const request of ['datasets', 'evaluations', 'modelArtifacts', 'runtimeProfiles', 'runtimeLogs', 'generationPresets', 'mcpServers', 'artifactLineage']) {
      expect(runtime).not.toContain(`.${request}(`);
    }
  });
  it('verifies architecture test behavior 5', () => {
    for (const file of files(root)) {
      const relative = file.replace(`${root}${sep}`, '').replaceAll(sep, '/');
      expect(relative).not.toBe('api.ts');
      expect(readFileSync(file, 'utf8')).not.toMatch(/from ['"][^'"]*\/api['"]/);
      expect(readFileSync(file, 'utf8')).not.toMatch(/\bapi\.[A-Za-z]/);
    }
  });
  it('verifies architecture test behavior 6', () => {
    const styles = source('styles.css');
    expect(styles).not.toContain('{');
    expect(styles).toContain('features/chat/chat.css');
    expect(styles).toContain('shared/styles/overrides.css');
  });
});
