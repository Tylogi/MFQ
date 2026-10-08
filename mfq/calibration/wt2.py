from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

from mfq.calibration.collector import HiddenStateStore
from mfq.calibration.imatrix import _backend
from mfq.calibration.loading import SourceLayers, release
from mfq.calibration.qwen35 import qwen35_head_weight_name


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class ReferenceWriter:
    def __init__(self, path, tokens, vocab):
        self.path = Path(path)
        self.partial = self.path.with_name(self.path.name + '.partial')
        self.tokens = np.asarray(tokens, dtype='<i4')
        self.chunks, self.context = self.tokens.shape
        self.start = self.context // 2 + 1
        self.count = self.context - self.start
        self.vocab = vocab
        self.elements = 2 * ((vocab + 1) // 2) + 4
        self.rows = None
        self.targets = None
        self.committed = False

    def __enter__(self):
        if self.path.exists():
            raise FileExistsError(f'reference already exists: {self.path}')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.partial.open('xb') as stream:
            stream.write(b'_logit3_' + struct.pack('<II', self.vocab, self.chunks))
            for _ in range(self.chunks):
                stream.write(struct.pack('<III', self.context, self.start, self.count))
            stream.write(self.tokens.tobytes())
            target_offset = stream.tell()
            row_offset = target_offset + self.chunks * self.count * 4
            stream.truncate(row_offset + self.chunks * self.count * self.elements * 2)
        self.targets = np.memmap(self.partial, mode='r+', dtype='<f4', offset=target_offset, shape=(self.chunks, self.count))
        self.rows = np.memmap(self.partial, mode='r+', dtype='<u2', offset=row_offset, shape=(self.chunks, self.count, self.elements))
        return self

    def row_header(self, chunk, start, maximum_logp, target_logp):
        count = len(maximum_logp)
        self.targets[chunk, start:start + count] = target_logp
        values = np.stack((np.full(count, 16 / 65535, dtype='<f4'), maximum_logp.astype('<f4') - 16), axis=-1)
        self.rows[chunk, start:start + count, :4] = values.view('<u2').reshape(count, 4)

    def row_block(self, chunk, start, vocabulary_start, codes):
        self.rows[chunk, start:start + len(codes), 4 + vocabulary_start:4 + vocabulary_start + codes.shape[1]] = codes

    def commit(self):
        self.targets.flush()
        self.rows.flush()
        self.targets._mmap.close()
        self.rows._mmap.close()
        self.targets = self.rows = None
        os.link(self.partial, self.path)
        self.partial.unlink()
        self.committed = True

    def __exit__(self, *args):
        for mapping in (self.targets, self.rows):
            if mapping is not None:
                mapping._mmap.close()
        self.targets = self.rows = None
        if not self.committed:
            self.partial.unlink(missing_ok=True)


def final_norm(backend):
    if str(backend.config.model_type).removesuffix('_text') == 'gemma4':
        from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm
        norm_type = Gemma4RMSNorm
    else:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
        norm_type = Qwen3_5RMSNorm
    with torch.device('meta'):
        norm = norm_type(backend.hidden_size, eps=float(backend.config.rms_norm_eps))
    norm.load_state_dict({'weight': backend.index.tensor('model.language_model.norm.weight', device='cpu')}, strict=True, assign=True)
    return norm.to(device=backend.device, dtype=backend.teacher_dtype).eval().requires_grad_(False)


@torch.inference_mode()
def write_head(backend, hidden, targets, writer, chunk, *, row_chunk=4096, token_chunk=32, resident_weight=None):
    head_name = qwen35_head_weight_name(backend.index)
    vocab = writer.vocab
    transform = getattr(backend, 'transform_logits', lambda value: value)
    total_nll = 0.
    for offset in range(0, hidden.shape[0], token_chunk):
        matrix = hidden[offset:offset + token_chunk]
        target = targets[offset:offset + token_chunk].to(backend.device)
        maximum = torch.full((len(matrix),), -torch.inf, device=backend.device)
        denominator = torch.zeros_like(maximum)
        target_logits = torch.full_like(maximum, torch.nan)
        for start in range(0, vocab, row_chunk):
            end = min(vocab, start + row_chunk)
            weight = resident_weight[start:end] if resident_weight is not None else backend.index.tensor(head_name,
                row_start=start, row_end=end, device=backend.device, dtype=backend.teacher_dtype)
            logits = transform(functional.linear(matrix, weight)).float()
            next_max = torch.maximum(maximum, logits.max(dim=1).values)
            denominator = denominator * torch.exp(maximum - next_max) + torch.exp(logits - next_max[:, None]).sum(dim=1)
            maximum = next_max
            selected = (target >= start) & (target < end)
            target_logits[selected] = logits[selected, target[selected] - start]
            del weight, logits
        normalizer = maximum + denominator.log()
        exact = target_logits - normalizer
        if not torch.isfinite(exact).all():
            raise ValueError('reference logits contain invalid values or target token IDs')
        total_nll -= float(exact.sum().item())
        writer.row_header(chunk, offset, (maximum - normalizer).cpu().numpy(), exact.cpu().numpy())
        for start in range(0, vocab, row_chunk):
            end = min(vocab, start + row_chunk)
            weight = resident_weight[start:end] if resident_weight is not None else backend.index.tensor(head_name,
                row_start=start, row_end=end, device=backend.device, dtype=backend.teacher_dtype)
            logits = transform(functional.linear(matrix, weight)).float()
            codes = torch.floor((logits - maximum[:, None] + 16) * (65535 / 16) + .5).clamp_(0, 65535)
            writer.row_block(chunk, offset, start, codes.cpu().numpy().astype('<u2'))
            del weight, logits, codes
    return total_nll


def tokenize_windows(root, text, context, chunks):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=True)
    config_path = Path(root) / 'tokenizer_config.json'
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    tokens = tokenizer.encode(text, add_special_tokens=True)
    if len(tokens) < context * chunks:
        raise ValueError(f'WT2 contains {len(tokens)} tokens; requested {context * chunks}')
    windows = np.array(tokens[:context * chunks], dtype='<i4').reshape(chunks, context)
    add_bos = bool(config.get('add_bos_token', False))
    if add_bos:
        if tokenizer.bos_token_id is None:
            raise ValueError('tokenizer requests BOS insertion but has no BOS token')
        windows[:, 0] = tokenizer.bos_token_id
    return windows, {'vocab_size': len(tokenizer), 'bos_token': tokenizer.bos_token_id,
        'eos_token': tokenizer.eos_token_id, 'pad_token': tokenizer.pad_token_id,
        'add_bos': add_bos, 'add_eos': bool(config.get('add_eos_token', False)),
        'add_special': True, 'parse_special': True, 'source': 'original_hf_tokenizer'}


def generate_reference(model, dataset, output, *, context_size=512, chunks=8, device='mps', layerwise=True, row_chunk=4096):
    from mfq.quantize.workbench import source_summary
    from mfq.server.services.evaluation_datasets import WT2_TEXT_SHA256
    root, dataset, output = Path(model).resolve(), Path(dataset).resolve(), Path(output).resolve()
    manifest_path = output.with_name(output.name + '.manifest.json')
    if output.exists() or manifest_path.exists():
        raise FileExistsError('reference output or manifest already exists')
    if sha256(dataset) != WT2_TEXT_SHA256:
        raise ValueError('WT2 generation requires the official mfq-wt2-text-v1 corpus')
    summary = source_summary(root)
    if summary['format'] != 'hf' or not summary['imatrix_supported']:
        raise ValueError('WT2 source generation currently supports original Qwen3.5 and Gemma4 HF architectures')
    if context_size < 32 or chunks < 1 or row_chunk < 1:
        raise ValueError('invalid WT2 context, window count or head row batch')
    target_device = torch.device(device)
    if target_device.type not in {'mps', 'cuda'}:
        raise ValueError('WT2 source generation requires Metal or CUDA')
    tokens, tokenizer = tokenize_windows(root, dataset.read_text(encoding='utf-8'), context_size, chunks)
    backend, model_type = _backend(root, target_device, 'sdpa')
    source_paths = sorted({root / shard for shard in backend.index.weight_map.values()} | {root / 'config.json'} |
        {path for path in (root / 'tokenizer.json', root / 'tokenizer_config.json') if path.is_file()})
    source_identity = [(path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in source_paths]
    config_hash = sha256(root / 'config.json')
    head_name = qwen35_head_weight_name(backend.index)
    vocab = backend.index.shape(head_name)[0]
    if tokens.min() < 0 or tokens.max() >= vocab:
        raise ValueError('tokenizer IDs exceed the source output vocabulary')
    tokenizer['vocab_size'] = vocab
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='mfq-wt2-', dir=output.parent) as work:
        store = HiddenStateStore(Path(work) / 'hidden.bf16', tokens.size, backend.hidden_size, backend.teacher_dtype)
        try:
            with SourceLayers(backend, layerwise=layerwise, purpose='wt2', context_size=context_size) as loading:
                for chunk, ids in enumerate(tokens):
                    hidden = loading.initial_hidden(torch.from_numpy(ids.astype(np.int64)).unsqueeze(0))
                    store.write(chunk * context_size, hidden)
                    del hidden
                store.flush()
                store.discard_cached_pages()
                for index in range(backend.num_layers):
                    with loading.layer(index) as layer:
                        for chunk in range(chunks):
                            start = chunk * context_size
                            hidden = store.read(start, start + context_size, (1, context_size), device=target_device)
                            store.write(start, backend.forward_layer(layer, index, hidden))
                            del hidden
                    del layer
                    store.flush()
                    store.discard_cached_pages()
                    if loading.layerwise:
                        release(target_device)
                    print(json.dumps({'event': 'wt2_layer', 'layer': index + 1, 'layers': backend.num_layers}), flush=True)
                norm = final_norm(backend)
                head = None if loading.layerwise else backend.index.tensor(head_name, device=target_device, dtype=backend.teacher_dtype)
                with ReferenceWriter(output, tokens, vocab) as writer:
                    nll = 0.
                    for chunk in range(chunks):
                        start = chunk * context_size
                        hidden = store.read(start, start + context_size, (1, context_size), device=target_device)
                        with torch.inference_mode():
                            value = norm(hidden[:, writer.start - 1:context_size - 1]).squeeze(0)
                        targets = torch.from_numpy(tokens[chunk, writer.start:].astype(np.int64))
                        nll += write_head(backend, value, targets, writer, chunk, row_chunk=row_chunk, resident_weight=head)
                        del hidden, value, targets
                        print(json.dumps({'event': 'wt2_head', 'chunk': chunk + 1, 'chunks': chunks}), flush=True)
                    scored = chunks * writer.count
                    if source_identity != [(path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in source_paths] or config_hash != sha256(root / 'config.json'):
                        raise ValueError('source model changed during reference generation')
                    writer.commit()
                del norm, head
                mode = {'layerwise': loading.layerwise, 'requested_layerwise': layerwise,
                    'fallback_reason': loading.fallback_reason, 'memory': loading.plan}
        finally:
            store.close()
            backend.release_initial_state()
            release(target_device)
    document = {
        'format': 'mfq.perplexity-logits-manifest.v1',
        'producer': {'tool': 'mfq.wt2-reference', 'runtime': 'torch', 'torch_version': torch.__version__, 'device': device},
        'reference': {'magic': '_logit3_', 'file': {'name': output.name, 'bytes': output.stat().st_size, 'sha256': sha256(output)},
            'distribution_encoding': {'type': 'linear_uint16_log_probability', 'logit_range': 16, 'code_max': 65535,
                'zero_code_is_clipped': True, 'target_log_probability': 'exact_float32'}},
        'model': {'label': root.name, 'architecture': model_type, 'precision': {'classification': '+'.join(summary['source_precisions']),
            'execution_dtype': 'bfloat16'}, 'config_sha256': config_hash,
            'files': [{'name': name, 'bytes': size, 'modified_ns': modified} for name, size, modified in source_identity]},
        'dataset': {'id': 'wikitext2', 'file_name': dataset.name, 'bytes': dataset.stat().st_size, 'sha256': WT2_TEXT_SHA256},
        'tokenizer': tokenizer,
        'evaluation': {'protocol': 'llama.cpp-non-strided-second-half-v1', 'n_ctx': context_size,
            'n_batch': context_size, 'n_ubatch': context_size, 'n_seq': 1, 'chunks': chunks,
            'input_tokens': tokens.size, 'input_token_ids_sha256': hashlib.sha256(tokens.tobytes()).hexdigest(),
            'per_chunk_bos_replacement': tokenizer['add_bos'], 'target_start': context_size // 2 + 1,
            'score_count_per_chunk': context_size - (context_size // 2 + 1), 'scored_tokens': scored,
            'logit_positions': [context_size // 2, context_size - 2], 'target_positions': [context_size // 2 + 1, context_size - 1],
            'attention': 'hf_sdpa', 'kv_cache_dtype': 'none'},
        'loading': mode, 'result': {'cross_entropy': nll / scored, 'perplexity': math.exp(nll / scored)}}
    manifest_partial = manifest_path.with_name(manifest_path.name + '.partial')
    try:
        with manifest_partial.open('x', encoding='utf-8') as stream:
            json.dump(document, stream, ensure_ascii=False, allow_nan=False, indent=2)
        os.link(manifest_partial, manifest_path)
        manifest_partial.unlink()
    except BaseException:
        output.unlink(missing_ok=True)
        manifest_partial.unlink(missing_ok=True)
        raise
    result = {'output': str(output), 'manifest': str(manifest_path), 'context_size': context_size, 'chunks': chunks,
        'parallel': 1, 'scored_tokens': scored, 'perplexity': document['result']['perplexity'], 'loading': mode}
    print(json.dumps({'event': 'wt2_reference_saved', **result}), flush=True)
    return result
