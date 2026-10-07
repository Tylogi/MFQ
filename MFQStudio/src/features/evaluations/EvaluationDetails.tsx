import { useState } from 'react';
import type { EvaluationResult } from '../../shared/api/types';

type Translate = (zh: string, en: string) => string;
export const resultObjects = (value: unknown): Record<string, unknown>[] => Array.isArray(value) ? value.filter((item): item is Record<string, unknown> => !!item && typeof item === 'object' && !Array.isArray(item)) : [];

export function ThroughputTable({ series, tr }: { series: unknown; tr: Translate }) {
  const rows = resultObjects(series);
  const number = (value: unknown, percent = false) => typeof value === 'number' && Number.isFinite(value) ? percent ? `${(value * 100).toFixed(2)}%` : value.toFixed(1) : '—';
  if (!rows.length) return null;
  return <div className="evaluation-comparison"><table><thead><tr>
    <th>{tr('输入 token', 'Input tokens')}</th><th>{tr('模式', 'Mode')}</th><th>prefill tok/s</th><th>decode tok/s</th><th>TTFT ms</th><th>{tr('实际输出 token', 'Actual output tokens')}</th><th>{tr('MTP 接受率', 'MTP acceptance')}</th>
  </tr></thead><tbody>{rows.flatMap((row, index) => (['decode', 'mtp'] as const).filter((mode) => `${mode}_prefill_tps` in row).map((mode) => <tr key={`${index}-${mode}`}>
    <th>{String(row.prompt_tokens)}</th><td>{mode === 'mtp' ? 'MTP' : tr('普通', 'Ordinary')}</td><td>{number(row[`${mode}_prefill_tps`])}</td><td>{number(row[`${mode}_decode_tps`])}</td><td>{number(row[`${mode}_ttft_ms`])}</td><td>{String(row[`${mode}_completion_tokens`] ?? '—')}</td><td>{mode === 'mtp' ? number(row.mtp_acceptance_rate, true) : '—'}</td>
  </tr>))}</tbody></table></div>;
}

export function AnswerDetails({ metrics, tr }: { metrics: EvaluationResult['metrics']; tr: Translate }) {
  const [filter, setFilter] = useState('all');
  const questions = resultObjects(metrics.questions);
  if (!questions.length) return null;
  const statuses: Record<string, string> = { correct: tr('正确', 'Correct'), incorrect: tr('答错', 'Incorrect'), truncated: tr('输出截断', 'Truncated'), no_answer: tr('无最终回答', 'No final answer'), parse_error: tr('解析失败', 'Parse error') };
  const selected = questions.filter((row) => filter === 'all' || row.status === filter);
  const categories = metrics.categories && typeof metrics.categories === 'object' ? Object.entries(metrics.categories) : [];
  return <details className="evaluation-answers"><summary>{tr('分类得分与逐题结果', 'Category scores & question results')} · {questions.length}</summary>
    <div className="evaluation-category-scores">{categories.map(([category, value]) => {
      const group = value as Record<string, unknown>;
      return <span key={category}>{category}<b>{typeof group.accuracy === 'number' ? `${(group.accuracy * 100).toFixed(1)}%` : '—'}</b><small>{String(group.correct)} / {String(group.total)}</small></span>;
    })}</div>
    <label className="evaluation-answer-filter">{tr('结果筛选', 'Filter answers')}<select value={filter} onChange={(event) => setFilter(event.target.value)}><option value="all">{tr('全部', 'All')}</option>{Object.entries(statuses).map(([status, label]) => <option key={status} value={status}>{label}</option>)}</select></label>
    {selected.slice(0, 100).map((row, index) => <details className="evaluation-question" key={`${String(row.id)}-${index}`}><summary><span>{String(row.id)} · {String(row.category)}{typeof row.generation_index === 'number' ? ` · ${tr('样本', 'Sample')} ${row.generation_index + 1}` : ''}</span><b>{statuses[String(row.status)] || String(row.status)}</b></summary>
      <p>{String(row.question)}</p>{Array.isArray(row.choices) && <ol type="A">{row.choices.map((choice, i) => <li key={i}>{String(choice)}</li>)}</ol>}
      <p>{tr('参考答案', 'Expected')}: {Array.isArray(row.expected) ? row.expected.join(' / ') : String(row.expected)} · {tr('模型答案', 'Predicted')}: {String(row.predicted ?? '—')}</p>
      <pre>{String(row.response || '')}</pre>{row.reasoning ? <details><summary>{tr('思考内容', 'Reasoning')}</summary><pre>{String(row.reasoning)}</pre></details> : null}
      <small>{String(row.finish_reason)} · {typeof row.latency_s === 'number' ? row.latency_s.toFixed(2) : '—'} s</small>
    </details>)}
    {selected.length > 100 && <p className="evaluation-note">{tr('页面展示前 100 条筛选结果；导出的 JSON 包含全部题目。', 'The first 100 matching results are shown; the JSON export contains all questions.')}</p>}
  </details>;
}

export function exportEvaluation(item: EvaluationResult) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(item, null, 2)], { type: 'application/json' }));
  const anchor = document.createElement('a');
  anchor.href = url; anchor.download = `mfq-evaluation-${item.id}.json`; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
