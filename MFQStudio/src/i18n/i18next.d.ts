/** Check translation keys and namespaces against the bundled English source catalog. */
import 'i18next';
import type { namespaces, resources } from './resources';

declare module 'i18next' {
  interface CustomTypeOptions {
    defaultNS: typeof namespaces;
    resources: typeof resources.en;
    returnNull: false;
    strictKeyChecks: true;
  }
}
