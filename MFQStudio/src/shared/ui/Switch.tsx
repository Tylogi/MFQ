/** Provide an accessible controlled switch for boolean settings that take effect immediately. */
import * as SwitchPrimitive from '@radix-ui/react-switch';
import './primitives.css';

interface SwitchProps {
  checked: boolean;
  /** Submit the new value when toggled; the parent is responsible for persisting the setting. */
  onCheckedChange: (checked: boolean) => void;
  label: string;
  disabled?: boolean;
  id?: string;
}

/** Render a settings switch that supports Space and screen-reader state without implicit form submission. */
export function Switch({ checked, onCheckedChange, label, disabled, id }: SwitchProps) {
  return (
    <SwitchPrimitive.Root
      id={id}
      className="studio-switch"
      checked={checked}
      onCheckedChange={onCheckedChange}
      disabled={disabled}
      aria-label={label}
      type="button"
    >
      <SwitchPrimitive.Thumb className="studio-switch-thumb" />
    </SwitchPrimitive.Root>
  );
}
