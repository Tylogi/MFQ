/** Register DOM assertions and clean up components mounted by each test to prevent cross-test contamination. */
import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/react';
import { afterEach, beforeEach } from 'vitest';
import { applyLanguage } from '../src/i18n';

beforeEach(() => {
  applyLanguage('en');
});

afterEach(() => {
  cleanup();
  localStorage.clear();
  sessionStorage.clear();
});
