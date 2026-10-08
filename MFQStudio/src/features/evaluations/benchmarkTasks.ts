export const taskBenchmarks = [
  { id: 'mmlu', name: 'MMLU', group: 'facts', dataset: 'mmlu-test', protocol: 'likelihood', tokens: 1, samples: 100, repository: 'hendrycks/test', zh: '57 学科 · 官方概率评分', en: '57 subjects · official likelihood scoring' },
  { id: 'accuracy', name: 'MMLU-Pro', group: 'reasoning', dataset: 'mmlu-pro-test', protocol: 'mcq', tokens: 4000, samples: 100, repository: 'TIGER-AI-Lab/MMLU-Pro', zh: '学科推理 · 官方评分', en: 'Academic reasoning · official scoring' },
  { id: 'livecodebench', name: 'LiveCodeBench', group: 'code', dataset: 'livecodebench-v6', protocol: 'code', tokens: 4096, samples: 20, repository: 'LiveCodeBench/LiveCodeBench', zh: 'v6 · Python pass@1', en: 'v6 · Python pass@1' },
  { id: 'aime2025', name: 'AIME2025', group: 'math', dataset: 'aime-2025', protocol: 'math', tokens: 4096, samples: 0, repository: 'eth-sri/matharena', zh: 'I + II · MathArena 评分', en: 'I + II · MathArena scoring' },
  { id: 'gpqa-diamond', name: 'GPQA-Diamond', group: 'reasoning', dataset: 'gpqa-diamond', protocol: 'mcq', tokens: 2048, samples: 100, repository: 'idavidrein/gpqa', zh: 'Diamond · 官方评分', en: 'Diamond · official scoring' },
  { id: 'truthfulqa', name: 'TruthfulQA', group: 'facts', dataset: 'truthfulqa-mc1', protocol: 'likelihood', tokens: 1, samples: 100, repository: 'sylinrl/TruthfulQA', zh: 'MC1 / MC2 · 官方概率评分', en: 'MC1 / MC2 · official likelihood scoring' },
] as const;

export type TaskBenchmark = typeof taskBenchmarks[number];

export const taskBenchmarkGroups = [
  { id: 'math', zh: '数学', en: 'Mathematics' },
  { id: 'reasoning', zh: '推理', en: 'Reasoning' },
  { id: 'code', zh: '代码', en: 'Code' },
  { id: 'facts', zh: '事实', en: 'Facts' },
] as const;
