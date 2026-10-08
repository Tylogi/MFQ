from __future__ import annotations

import gc
import json
import math
import sys
from contextlib import ExitStack, contextmanager

import psutil
import torch


def loading_plan(model, *, backend='auto', purpose='imatrix', context_size=16384, batch_size=1, device=None):
    from mfq.quantize.workbench import ModelSource

    source = ModelSource(model)
    try:
        if not source.path.is_dir():
            raise ValueError('generation requires an original HF model directory')
        config = source.config.get('text_config', source.config)
        weights = source.parameters() * 2
        hidden = int(config['hidden_size'])
        intermediate = max(hidden, int(config.get('intermediate_size', 0)), int(config.get('moe_intermediate_size', 0)))
        heads = int(config.get('num_attention_heads', 1))
        workspace = max(512 << 20, batch_size * context_size * (hidden * 32 + intermediate * 16))
        workspace += batch_size * heads * context_size * context_size * 4
        if purpose == 'imatrix':
            workspace += source.parameters() // 32
        required = math.ceil(weights * 1.15) + workspace
    finally:
        source.close()
    host = psutil.virtual_memory()
    reserve = max(1 << 30, host.total // 32)
    host_available = max(0, host.available - reserve)
    backend = ('metal' if sys.platform == 'darwin' else 'cuda') if backend == 'auto' else backend
    if backend == 'metal':
        available = 0
        if torch.backends.mps.is_available():
            available = min(host_available, max(0, torch.mps.recommended_max_memory() - torch.mps.driver_allocated_memory()))
    elif backend == 'cuda' and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(device)
        available = min(host_available, max(0, free - max(512 << 20, total // 32)))
    else:
        available = 0
    return {'backend': backend, 'resident_required_bytes': required, 'available_bytes': available,
        'resident_allowed': required <= available, 'weights_bytes': weights, 'workspace_bytes': workspace}


def release(device):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.synchronize()
        torch.mps.empty_cache()


def embedding_hidden(backend, input_ids, embedding=None):
    ids = input_ids.detach().to(device='cpu', dtype=torch.int64)
    if embedding is not None:
        hidden = embedding.index_select(0, ids.reshape(-1).to(embedding.device))
    else:
        unique, inverse = torch.unique(ids.reshape(-1), sorted=True, return_inverse=True)
        rows = []
        for block in unique.split(256):
            start, end = int(block[0]), int(block[-1]) + 1
            if end - start <= 8192:
                weight = backend.index.tensor('model.language_model.embed_tokens.weight', row_start=start, row_end=end, device='cpu')
                rows.append(weight.index_select(0, block - start))
            else:
                rows.extend(backend.index.tensor('model.language_model.embed_tokens.weight', row_start=int(row), row_end=int(row) + 1, device='cpu') for row in block)
        hidden = torch.cat(rows).index_select(0, inverse)
    if str(backend.config.model_type).removesuffix('_text') == 'gemma4':
        hidden = hidden * torch.tensor(backend.hidden_size**0.5, dtype=hidden.dtype, device=hidden.device)
    return hidden.reshape(*ids.shape, backend.hidden_size).to(device=backend.device, dtype=backend.teacher_dtype)


class SourceLayers:
    def __init__(self, backend, *, layerwise=True, purpose='imatrix', context_size=16384, batch_size=1):
        self.backend = backend
        self.requested_layerwise = layerwise
        self.layerwise = layerwise
        self.purpose = purpose
        self.context_size = context_size
        self.batch_size = batch_size
        self.stack = ExitStack()
        self.layers = []
        self.embedding = None
        self.plan = None
        self.fallback_reason = None

    def __enter__(self):
        if not self.layerwise:
            self.plan = loading_plan(self.backend.root, backend='metal' if self.backend.device.type == 'mps' else 'cuda',
                purpose=self.purpose, context_size=self.context_size, batch_size=self.batch_size, device=self.backend.device)
            if not self.plan['resident_allowed']:
                self.layerwise = True
                self.fallback_reason = 'insufficient_memory'
            else:
                try:
                    for index in range(self.backend.num_layers):
                        self.layers.append(self.stack.enter_context(self.backend.layer(index, quantized=False)))
                    self.embedding = self.backend.index.tensor('model.language_model.embed_tokens.weight',
                        device=self.backend.device, dtype=self.backend.teacher_dtype)
                except RuntimeError as error:
                    if not isinstance(error, torch.OutOfMemoryError) and 'MPS backend out of memory' not in str(error):
                        self.__exit__(*sys.exc_info())
                        raise
                    self.layers.clear()
                    self.embedding = None
                    self.stack.close()
                    release(self.backend.device)
                    self.layerwise = True
                    self.fallback_reason = 'insufficient_memory'
                except BaseException:
                    self.__exit__(*sys.exc_info())
                    raise
        print(json.dumps({'event': 'source_loading', 'layerwise': self.layerwise,
            'requested_layerwise': self.requested_layerwise, 'fallback_reason': self.fallback_reason,
            'memory': self.plan}), flush=True)
        return self

    @contextmanager
    def layer(self, index):
        if self.layerwise:
            with self.backend.layer(index, quantized=False) as layer:
                yield layer
        else:
            yield self.layers[index]

    def initial_hidden(self, ids):
        return embedding_hidden(self.backend, ids, self.embedding)

    def __exit__(self, *args):
        self.layers.clear()
        self.embedding = None
        self.stack.__exit__(*args)
        release(self.backend.device)
