import { CaretDownIcon } from '@phosphor-icons/react';
import { useLayoutEffect, useRef, useState, type SelectHTMLAttributes } from 'react';

export function CompactSelect({ children, className = '', onChange, title, ...props }: SelectHTMLAttributes<HTMLSelectElement>) {
  const select = useRef<HTMLSelectElement>(null);
  const [label, setLabel] = useState('');
  useLayoutEffect(() => {
    setLabel(select.current?.selectedOptions[0]?.textContent ?? '');
  }, [children, props.value, props.defaultValue]);
  return <span className={`compact-select ${className}`.trim()}>
    <span className="compact-select-value" aria-hidden="true">{label}</span>
    <CaretDownIcon size={14} aria-hidden="true" />
    <select {...props} ref={select} title={title ?? label} onChange={event => {
      setLabel(event.currentTarget.selectedOptions[0]?.textContent ?? '');
      onChange?.(event);
    }}>{children}</select>
  </span>;
}
