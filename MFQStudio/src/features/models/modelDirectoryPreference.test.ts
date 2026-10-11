import { afterEach, expect, it, vi } from 'vitest';
import { setApiBaseUrl } from '../../shared/api/client';
import { readModelDirectory, saveModelDirectory } from './modelDirectoryPreference';

afterEach(() => { setApiBaseUrl(''); vi.restoreAllMocks(); });

it('defaults to the root and restores an explicitly selected folder', () => {
  expect(readModelDirectory()).toBe('/');
  saveModelDirectory('/models/chosen');
  expect(readModelDirectory()).toBe('/models/chosen');
});

it('does not reuse a different server’s folder', () => {
  setApiBaseUrl('https://first.example/');
  saveModelDirectory('/first/models');
  setApiBaseUrl('https://second.example');
  expect(readModelDirectory()).toBe('/');
  saveModelDirectory('D:\\Models');
  setApiBaseUrl('https://first.example');
  expect(readModelDirectory()).toBe('/first/models');
  setApiBaseUrl('https://second.example/');
  expect(readModelDirectory()).toBe('D:\\Models');
});

it('still permits selecting a folder when browser storage is unavailable', () => {
  vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('blocked'); });
  vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('blocked'); });
  expect(readModelDirectory()).toBe('/');
  expect(() => saveModelDirectory('/models')).not.toThrow();
});
