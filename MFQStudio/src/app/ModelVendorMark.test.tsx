import { render, screen } from '@testing-library/react';
import { expect, it } from 'vitest';
import { ModelVendorMark, modelVendor } from './ModelVendorMark';

it.each([
  ['qwen4_exp_text', 'qwen'], ['Qwen3NextForCausalLM', 'qwen'], ['qwen3_vl', 'qwen'],
  ['deepseek_v4', 'deepseek'], ['DeepseekV3ForCausalLM', 'deepseek'],
  ['glm5_next', 'zai'], ['ChatGLMForConditionalGeneration', 'zai'],
])('uses architecture %s even for a renamed checkpoint', (architecture, vendor) => {
  expect(modelVendor('renamed-checkpoint', architecture)).toBe(vendor);
});

it.each([
  ['Tylogi/Qwen3.8-Flash-Next-S4-L', 'qwen'], ['Qwen', 'qwen'],
  ['community/DeepSeek-V4.1-MFQ', 'deepseek'], ['GLM-5.3', 'zai'], ['ChatGLM3-6B', 'zai'],
])('uses the model basename when architecture is absent: %s', (name, vendor) => {
  expect(modelVendor(name)).toBe(vendor);
});

it('prefers declared architecture to model name and does not label unsupported architectures', () => {
  expect(modelVendor('DeepSeek-R1-Distill-Qwen-32B', ['Qwen2ForCausalLM'])).toBe('qwen');
  expect(modelVendor('DeepSeek-R1-Distill-Llama-70B', 'LlamaForCausalLM')).toBeNull();
  expect(modelVendor('Qwen-custom', 'minicpm')).toBeNull();
  expect(modelVendor('Qwen/unknown-model')).toBeNull();
  expect(modelVendor('Qwenish-model')).toBeNull();
  expect(modelVendor('MiniCPM-o-4_5')).toBeNull();
  expect(modelVendor('GLM-5.3', ['', 'GlmMoeForCausalLM'])).toBe('zai');
});

it.each([['qwen3', 'Qwen'], ['deepseek4', 'DeepSeek'], ['glm5', 'Z.ai']])('renders %s as an unfilled borderless vector', (architecture, name) => {
  render(<ModelVendorMark architecture={architecture} />);
  const mark = screen.getByRole('img', { name });
  expect(mark).toHaveAttribute('fill', 'none');
  expect(mark).toHaveAttribute('stroke', 'currentColor');
  expect(mark.querySelectorAll('path')).toHaveLength(1);
  expect(mark.querySelector('rect')).toBeNull();
  expect(mark.querySelector('text')).toBeNull();
});

it('does not create a placeholder or logo for unknown models', () => {
  const { container } = render(<ModelVendorMark name="unknown" />);
  expect(container).toBeEmptyDOMElement();
});
