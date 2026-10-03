import { Icon, type IconName } from '../../app/display';
import type { HubSystemProfile } from '../../shared/api/types';

const vendors: Partial<Record<HubSystemProfile['backend'], { name: string; icon: IconName }>> = {
  metal: { name: 'Apple', icon: 'apple' },
  cuda: { name: 'NVIDIA', icon: 'nvidia' },
  rocm: { name: 'AMD', icon: 'amd' },
};

export function BackendBadge({ backend }: { backend: HubSystemProfile['backend'] }) {
  if (backend === 'unknown') return null;
  const vendor = vendors[backend];
  return (
    <span className="hardware-backend-badge" role="img" aria-label={vendor ? `${vendor.name} · ${backend.toUpperCase()}` : 'CPU'}>
      {vendor && <Icon name={vendor.icon} size={22} />}
      {backend.toUpperCase()}
    </span>
  );
}
