"""Measure streamed prose and its continuation with ordinary stochastic sampling.

Run against a loaded FP16-KV API service. A cold first turn requires a fresh
model load before invoking this command. The synthetic repeated input gives
approximately 8K tokens; output questions require Chinese prose, not counting.
"""

import argparse
import json
import time
from pathlib import Path
from uuid import uuid4

import requests

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
parser.add_argument('--url', default='http://127.0.0.1:8090')
parser.add_argument('--model', default='Qwen3.8-Flash-Next-EWQ-MFQ-S2-L')
parser.add_argument('--background', type=Path, help='Use a document as the input background')
parser.add_argument('--rounds', type=int, default=2)
parser.add_argument('--repeats', type=int, default=740)
parser.add_argument('--max-tokens', type=int, default=128)
parser.add_argument('--mtp', choices=['on', 'off'], default='on')
parser.add_argument('--temperature', type=float, default=0.7)
parser.add_argument('--top-p', type=float, default=0.8)
parser.add_argument('--top-k', type=int, default=20)
parser.add_argument('--topic', choices=['technical', 'everyday', 'story'], default='technical')
args = parser.parse_args()
if args.temperature <= 0:
    raise ValueError('This benchmark requires sampling, with temperature > 0')
questions = {
    'technical': (
        '请用自然流畅的中文解释：为什么大语言模型推理服务需要同时管理显存、内存和磁盘？'
        '这些资源之间怎样配合，才能兼顾响应速度和长上下文支持？用自然段回答。',
        '继续解释：当用户不断追加新的对话内容时，这套资源管理可能遇到什么问题，应该怎样处理？用自然段回答。'),
    'everyday': (
        '我最近工作很忙，睡前总是忍不住继续看手机。请用自然流畅的中文分析这种习惯为什么难以改变，'
        '并给出一段可以在日常生活中实行的建议，用自然段回答。',
        '如果我尝试几天后又恢复了原来的习惯，应该怎样调整？请给出实际、温和的建议，用自然段回答。'),
    'story': (
        '请用自然流畅的中文写一个短篇故事的开头：一位修理收音机的老人，在雨夜听到了已经停播多年的电台。'
        '注重场景、人物动作和心理描写。',
        '继续这个故事，让老人发现这段广播与他年轻时的一件往事有关，保持自然的叙述。'),
}
if not 0 < args.top_p <= 1 or args.top_k <= 0:
    raise ValueError('This benchmark requires top_p in (0, 1] and top_k > 0')
out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
s = requests.Session()
s.trust_env = False
base = args.url.rstrip('/')
background = args.background.read_text(encoding='utf-8') if args.background else (
    'The quick brown fox jumps over the lazy dog.\n' * args.repeats)
result = {'topic': args.topic, 'mtp': args.mtp,
          'sampling': {'temperature': args.temperature, 'top_p': args.top_p, 'top_k': args.top_k, 'seed': 1234},
          'input_note': 'Document background' if args.background else
              'Synthetic repeated English background (~8K tokens), followed by an open-ended Chinese question.',
          'max_output_tokens': args.max_tokens, 'records': [],
          'initial_status': s.get(base+'/api/v1/runtime/status', timeout=20).json()}
assert result['initial_status'].get('kv_quantization_bits') == 0
messages = [{'role': 'system', 'content': '请直接回答用户最后的问题，不要复述英文背景或会话标识。会话标识：' + uuid4().hex},
            {'role': 'user', 'content':
             background +
             '\n最后的问题：' + questions[args.topic][0]}]
conversation = str(uuid4())
for turn in range(args.rounds):
    payload = {'model': args.model, 'messages': messages,
               'mfq_session_id': conversation, 'temperature': args.temperature, 'seed': 1234,
               'top_p': args.top_p, 'top_k': args.top_k, 'repetition_penalty': 1,
               'max_tokens': args.max_tokens, 'enable_thinking': False,
               'enable_mtp': args.mtp == 'on', 'mtp_max_draft_tokens': 3, 'stream': True,
               'stream_options': {'include_usage': True}}
    print(json.dumps({'start_turn': turn+1, 'topic': args.topic, 'mtp': args.mtp}), flush=True)
    started = time.perf_counter()
    row = {'turn': turn+1, 'deltas': [], 'metrics': {}, 'usage': {}}
    with s.post(base+'/v1/chat/completions', json=payload, stream=True, timeout=(15, 180)) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1):
            if not line.startswith(b'data:'):
                continue
            data = line[5:].strip()
            if data == b'[DONE]':
                break
            event = json.loads(data)
            if 'error' in event:
                raise RuntimeError(event['error'])
            if event.get('usage'):
                row['usage'] = event['usage']
            if event.get('mfq_metrics'):
                row['metrics'] = event['mfq_metrics']
            for choice in event.get('choices', []):
                piece = choice.get('delta', {}).get('content', '')
                if piece:
                    row['deltas'].append({'ms': (time.perf_counter()-started)*1000, 'text': piece})
                if choice.get('finish_reason'):
                    row['finish_reason'] = choice['finish_reason']
    row['wall_seconds'] = time.perf_counter()-started
    row['text'] = ''.join(item['text'] for item in row['deltas'])
    gaps = [b['ms']-a['ms'] for a,b in zip(row['deltas'], row['deltas'][1:])]
    row['largest_gaps_ms'] = sorted(gaps, reverse=True)[:5]
    row['status_after'] = s.get(base+'/api/v1/runtime/status', timeout=20).json()
    result['records'].append(row)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    metric = row['metrics']
    print(json.dumps({'turn': turn+1, 'prompt_tokens': row['usage'].get('prompt_tokens'),
        'output_tokens': row['usage'].get('completion_tokens'), 'decode_tps': metric.get('decode_tps'),
        'prefill_tps': metric.get('prefill_tps'), 'ttft_ms': metric.get('ttft_ms'),
        'mtp_acceptance_rate': metric.get('mtp_acceptance_rate'),
        'mtp_selected_depth': metric.get('mtp_selected_depth'), 'mtp_cycles': metric.get('mtp_cycles'),
        'mtp_standard_tokens': metric.get('mtp_standard_tokens'), 'mtp_park_count': metric.get('mtp_park_count'),
        'largest_gaps_ms': row['largest_gaps_ms'], 'text_sample': row['text'][:200]}, ensure_ascii=True), flush=True)
    messages += [{'role': 'assistant', 'content': row['text']},
                 {'role': 'user', 'content': questions[args.topic][1]}]
