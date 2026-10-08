import type { TaskBenchmark } from './benchmarkTasks';

const marks: Record<TaskBenchmark['id'], [string, string?]> = {
  mmlu: ['M'], accuracy: ['M', 'P'], livecodebench: ['L', '6'],
  aime2025: ['A', '2025'], 'gpqa-diamond': ['G', 'D'], truthfulqa: ['T'],
};

export function BenchmarkMark({ id }: { id: TaskBenchmark['id'] }) {
  const [letter, suffix] = marks[id];
  return <span className="evaluation-benchmark-mark" aria-hidden="true"><b>{letter}</b>{suffix && <sup>{suffix}</sup>}</span>;
}

export function BenchmarkGroupMark({ group }: { group: TaskBenchmark['group'] }) {
  return <svg className="evaluation-group-mark" data-group={group} aria-hidden="true" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
    {group === 'math' && <path d="M18 4H6l7 8-7 8h12" />}
    {group === 'reasoning' && <><circle cx="10.5" cy="10.5" r="6.5" /><path d="m15.5 15.5 5 5" /></>}
    {group === 'code' && <path d="m4 6 6 6-6 6M13 18h7" />}
    {group === 'facts' && <><path d="M12 6c-3-2-6-2-9-1v14c3-1 6-1 9 1 3-2 6-2 9-1V5c-3-1-6-1-9 1v14" /><path d="M6 9h3M6 12h3M15 9h3M15 12h3" /></>}
  </svg>;
}
