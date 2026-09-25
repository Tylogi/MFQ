/** 验证概览与历史消息实际展示相同的原生预填充指标。 */
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { expect, it, vi } from 'vitest';
import { OverviewPage } from '../src/features/runtime/OverviewPage';
import { SavedMessageList } from '../src/features/chat/SavedMessageList';
import { TooltipProvider } from '../src/shared/ui/Tooltip';
import type { Message, ResponseResource, ResponsePerformance } from '../src/shared/api/types';

const performance: ResponsePerformance = {
  ttft_ms: 200, prefill_tokens: 40,
  prefill_ms: 50, model_prefill_ms: 100, complete_prefill_ms: 2000,
  prefill_tps: 999, decode_tps: 10, processor_ms: 1800, multimodal_ms: 50,
  complete_prefill_tps: 20, decode_ms: 1000, generation_ms: 1200,
  complete_generation_ms: 3000, generation_tps: 10,
  sampling: {
    max_tokens: 4096, temperature: 0.7, top_p: 0.8, top_k: 20,
    repetition_penalty: 1, presence_penalty: 0, frequency_penalty: 0,
    enable_thinking: true, enable_vision: true, enable_mtp: false,
  },
};

vi.mock('../src/app/RuntimeProvider', () => ({
  useRuntime: () => ({
    runtime: { last_request: performance }, models: [], instances: [], studio: null,
    selectedModel: '', setSelectedModel: vi.fn(), refreshRuntime: vi.fn(), loading: false,
  }),
}));
vi.mock('../src/features/settings/SettingsProvider', () => ({
  useSettings: () => ({ tr: (_chinese: string, english: string) => english }),
}));

it('概览预填充卡片展示原生速度与耗时，而非媒体准备总时间或预报速度', () => {
  render(<MemoryRouter><OverviewPage /></MemoryRouter>);
  expect(screen.getByText('200 tok/s')).toBeInTheDocument();
  expect(screen.getByText('200 ms · Prompt processing')).toBeInTheDocument();
  expect(screen.queryByText('999 tok/s')).not.toBeInTheDocument();
});

it('历史响应使用同一预填充速度，并单独显示媒体准备耗时', () => {
  const message: Message = { id: 'answer', role: 'assistant', parts: [], parent_id: null, created_at: '2026-09-24T00:00:00Z' };
  const response: ResponseResource = {
    id: 'response-a', request_id: 'request-a', session_id: 'session-a', status: 'completed',
    output: [], created_at: message.created_at, performance,
    usage: { prompt_tokens: 40, completion_tokens: 10, total_tokens: 50 },
  };
  render(<TooltipProvider><SavedMessageList
    messages={[message]} responses={{ answer: response }} mcpTools={[]} busy={false}
    tr={(_chinese, english) => english} editDraft={null} setEditDraft={vi.fn()}
    actions={{ saveEdit: vi.fn(), copyMessage: vi.fn(), regenerate: vi.fn(), executeToolCalls: vi.fn() }}
  /></TooltipProvider>);
  expect(screen.getByText('200 pp')).toBeInTheDocument();
  expect(screen.getByText('200 ms TTFT')).toBeInTheDocument();
  expect(screen.getByText('Media preparation 1,800 ms')).toBeInTheDocument();
});
