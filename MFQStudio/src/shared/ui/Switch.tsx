/** 为即时生效的布尔设置提供带无障碍名称的受控开关。 */
import * as SwitchPrimitive from '@radix-ui/react-switch';
import './primitives.css';

interface SwitchProps {
  checked: boolean;
  /** 用户切换开关时提交新值，由父级负责持久化设置。 */
  onCheckedChange: (checked: boolean) => void;
  label: string;
  disabled?: boolean;
  id?: string;
}

/** 渲染支持空格键和读屏状态的设置开关，避免表单内隐式提交。 */
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
