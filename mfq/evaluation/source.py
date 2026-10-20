from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import psutil

from mfq.formats.hf_source import _source_records, _weight_map


class ReferenceSource:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_dir():
            raise ValueError('reference generation requires an original HF model directory')
        self.config = json.loads((self.path / 'config.json').read_text(encoding='utf-8'))
        self.text = self.config.get('text_config') or self.config
        self.architecture = self.text.get('model_type', self.config.get('model_type', 'unknown'))
        if 'nvfp4' in json.dumps(self.config).lower():
            raise ValueError('NVFP4 reference sources are not supported')
        self.records = _source_records(self.path, _weight_map(self.path))
        self.files = sorted({record.path for record in self.records.values()} | {self.path / 'config.json'} |
            {path for path in (self.path / 'tokenizer.json', self.path / 'tokenizer_config.json') if path.is_file()})

    def identity(self):
        return [(str(path.relative_to(self.path)), path.stat().st_size, path.stat().st_mtime_ns) for path in self.files]

    def summary(self):
        labels = {'BF16': 'BF16', 'F16': 'FP16', 'F32': 'FP32', 'F64': 'FP64',
            'F8_E4M3': 'FP8', 'F8_E4M3FN': 'FP8', 'F8_E5M2': 'FP8'}
        weights = [record for record in self.records.values() if not any(part in record.name for part in
            ('weight_scale', 'weight_inverse_scale', 'weight_scale_inv', 'input_scale', 'activation_scale'))]
        for record in weights:
            if record.name.endswith('.weight') and record.dtype not in labels:
                scale = self.records.get(record.name.removesuffix('.weight') + '.weight_scale')
                if record.dtype not in {'I8', 'U8'} or scale is None or scale.dtype != 'F8_E8M0':
                    raise ValueError(f'unsupported reference source weight: {record.name}={record.dtype}')
        precisions = {labels[record.dtype] for record in weights if record.dtype in labels}
        for record in weights:
            scale_name = record.name.removesuffix('.weight') + '.weight_scale'
            scale = self.records.get(scale_name)
            if scale is not None and scale.dtype == 'F8_E8M0':
                precisions.add('MXFP4' if record.dtype in {'I8', 'U8'} else 'MXFP8')
        return {'path': str(self.path), 'architecture': self.architecture, 'tensors': len(weights),
            'parameters': sum(math.prod(record.shape) * (2 if record.dtype in {'I8', 'U8'} else 1) for record in weights),
            'source_precisions': sorted(precisions), 'eligible_candidates': [], 'format': 'hf', 'full_precision': True,
            'imatrix_supported': self.architecture.removesuffix('_text') in {'qwen3_5', 'qwen3_5_moe', 'gemma4'}}


def source_summary(path):
    return ReferenceSource(path).summary()


def loading_plan(model, *, backend='auto', purpose='wt2', context_size=512, batch_size=1, device=None):
    source = ReferenceSource(model)
    text = source.text
    embedding = next((record for name, record in source.records.items() if name.endswith('embed_tokens.weight')), None)
    weights = source.summary()['parameters'] * (4 if embedding is not None and embedding.dtype == 'F32' else 2)
    hidden = int(text['hidden_size'])
    intermediate = max(hidden, int(text.get('intermediate_size', 0)), int(text.get('moe_intermediate_size', 0)))
    workspace = max(512 << 20, batch_size * context_size * (hidden * 32 + intermediate * 16))
    workspace += batch_size * int(text.get('num_attention_heads', 1)) * context_size**2 * 4
    if purpose == 'imatrix':
        workspace += source.summary()['parameters'] // 32
    required = math.ceil(weights * 1.15) + workspace
    host = psutil.virtual_memory()
    available = max(0, host.available - max(1 << 30, host.total // 32))
    backend = ('metal' if sys.platform == 'darwin' else 'cuda') if backend == 'auto' else backend
    if backend == 'metal':
        try:
            import mlx.core as mx
            if not mx.metal.is_available():
                available = 0
            else:
                limit = mx.device_info()['max_recommended_working_set_size']
                available = min(available, max(0, limit - mx.get_active_memory()))
        except ImportError:
            available = 0
    elif backend == 'cuda':
        from mfq.server.services.hardware import gpu_memory_status
        ordinal = str(device or 'cuda:0').split(':')[-1]
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible is not None:
            devices = visible.split(',')
            index = int(ordinal)
            if not 0 <= index < len(devices) or not devices[index].strip():
                raise ValueError('CUDA reference device is not visible')
            ordinal = devices[index].strip()
        gpu = next((item for item in gpu_memory_status() if item['id'] == ordinal), {})
        free, total = gpu.get('memory_available_bytes') or 0, gpu.get('memory_total_bytes') or 0
        available = min(available, max(0, free - max(512 << 20, total // 32)))
    else:
        raise ValueError('reference generation requires Metal or CUDA')
    return {'backend': backend, 'resident_required_bytes': required, 'available_bytes': available,
        'resident_allowed': required <= available, 'weights_bytes': weights, 'workspace_bytes': workspace}
