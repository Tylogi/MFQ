import { render, screen } from '@testing-library/react';
import { expect, it } from 'vitest';
import { ModelMemoryPressure } from './ModelMemoryPressure';

it.each([
  [25, 100, '25.0%', 25, '25%'],
  [150, 100, '150.0%', 100, '100%'],
  [10, 0, 'No available memory', 100, '100%'],
  [0, 0, '0.0%', 0, '0%'],
  [null, 100, '—', null, '0%'],
])('keeps pressure values without inline color at %s / %s', (required, available, text, value, width) => {
  const { container } = render(<ModelMemoryPressure required={required} available={available}
    label="Memory pressure" emptyLabel="No available memory" />);
  const bar = screen.getByRole('progressbar', { name: `Memory pressure: ${text}` });
  expect(bar).toHaveAttribute('aria-valuetext', text);
  if (value == null) expect(bar).not.toHaveAttribute('aria-valuenow');
  else expect(bar).toHaveAttribute('aria-valuenow', String(value));
  expect(bar.firstElementChild).toHaveStyle({ width });
  expect((bar.firstElementChild as HTMLElement).style.background).toBe('');
  expect((bar.firstElementChild as HTMLElement).style.backgroundColor).toBe('');
  expect(container.querySelector('strong')).toHaveTextContent(text);
  expect((container.querySelector('strong') as HTMLElement).style.color).toBe('');
});
