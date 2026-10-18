from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
from pathlib import Path

import numpy as np

from mfq.evaluation.source import ReferenceSource, loading_plan


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
        if self.count < 1 or vocab < 1 or self.chunks < 1:
            raise ValueError('invalid reference geometry')
        self.vocab = vocab
        self.elements = 2 * ((vocab + 1) // 2) + 4
        self.rows = self.targets = None
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

    def logits(self, chunk, start, logits):
        values = np.asarray(logits, dtype=np.float32)
        if (values.ndim != 2 or values.shape[1] != self.vocab or not np.isfinite(values).all()
                or not 0 <= chunk < self.chunks or start < 0 or start + len(values) > self.count):
            raise ValueError('invalid native reference logits')
        maximum = values.max(-1)
        shifted = values - maximum[:, None]
        log_denominator = np.log(np.exp(shifted).sum(-1))
        targets = self.tokens[chunk, self.start + start:self.start + start + len(values)]
        if np.any(targets < 0) or np.any(targets >= self.vocab):
            raise ValueError('reference target is outside the vocabulary')
        target_logp = shifted[np.arange(len(values)), targets] - log_denominator
        self.row_header(chunk, start, -log_denominator, target_logp)
        codes = np.floor((shifted + 16) * (65535 / 16) + .5).clip(0, 65535).astype('<u2')
        self.row_block(chunk, start, 0, codes)
        return -float(target_logp.sum(dtype=np.float64))

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


def tokenize_windows(root, text, context, chunks):
    from tokenizers import Tokenizer
    root = Path(root)
    tokenizer = Tokenizer.from_file(str(root / 'tokenizer.json'))
    config_path = root / 'tokenizer_config.json'
    config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.is_file() else {}
    def special(name):
        value = config.get(name)
        if isinstance(value, dict):
            value = value.get('content')
        return tokenizer.token_to_id(value) if isinstance(value, str) else None
    bos, eos, pad = (special(name) for name in ('bos_token', 'eos_token', 'pad_token'))
    tokens = tokenizer.encode(text, add_special_tokens=True).ids
    add_bos, add_eos = bool(config.get('add_bos_token', False)), bool(config.get('add_eos_token', False))
    if add_bos and bos is None or add_eos and eos is None:
        raise ValueError('tokenizer requests a special token absent from its vocabulary')
    if add_bos and (not tokens or tokens[0] != bos):
        tokens.insert(0, bos)
    if add_eos and (not tokens or tokens[-1] != eos):
        tokens.append(eos)
    if len(tokens) < context * chunks:
        raise ValueError(f'WT2 contains {len(tokens)} tokens; requested {context * chunks}')
    windows = np.array(tokens[:context * chunks], dtype='<i4').reshape(chunks, context)
    if add_bos:
        windows[:, 0] = bos
    return windows, {'vocab_size': tokenizer.get_vocab_size(), 'bos_token': bos, 'eos_token': eos,
        'pad_token': pad, 'add_bos': add_bos, 'add_eos': add_eos, 'add_special': True,
        'parse_special': True, 'source': 'original_hf_tokenizer'}


def generate_reference(model, dataset, output, *, context_size=512, chunks=8, device='mps', layerwise=True, runtime=None):
    from mfq.server.services.evaluation_datasets import WT2_TEXT_SHA256
    source = ReferenceSource(model)
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    manifest = output.with_name(output.name + '.manifest.json')
    if output.exists() or manifest.exists():
        raise FileExistsError('reference output or manifest already exists')
    if sha256(dataset) != WT2_TEXT_SHA256:
        raise ValueError('WT2 generation requires the official mfq-wt2-text-v1 corpus')
    architecture = source.architecture.removesuffix('_text')
    if architecture not in {'qwen3_5', 'qwen3_5_moe', 'gemma4', 'glm5_next'}:
        raise ValueError('WT2 reference generation supports original Qwen3.5, Gemma4 and GLM-5.3-Flash HF architectures')
    if context_size < 32 or chunks < 1:
        raise ValueError('invalid WT2 context or window count')
    if context_size > int(source.text['max_position_embeddings']):
        raise ValueError('WT2 context exceeds the original model context capacity')
    backend = 'metal' if device in {'mps', 'metal'} else 'cuda' if device.startswith('cuda') else None
    if backend is None:
        raise ValueError('WT2 source generation requires Metal or CUDA')
    if architecture == 'glm5_next' and backend != 'cuda':
        raise ValueError('GLM-5.3-Flash WT2 reference generation requires native CUDA')
    plan = loading_plan(source.path, backend=backend, purpose='wt2', context_size=context_size, device=device)
    mode = {'layerwise': layerwise or not plan['resident_allowed'], 'requested_layerwise': layerwise,
        'fallback_reason': 'insufficient_memory' if not layerwise and not plan['resident_allowed'] else None, 'memory': plan}
    print(json.dumps({'event': 'source_loading', **mode}), flush=True)
    tokens, tokenizer = tokenize_windows(source.path, dataset.read_text(encoding='utf-8'), context_size, chunks)
    vocab = int(source.text.get('vocab_size', source.config.get('vocab_size', 0)))
    if vocab < 1 or tokens.min() < 0 or tokens.max() >= vocab:
        raise ValueError('tokenizer IDs exceed the source output vocabulary')
    tokenizer['vocab_size'] = vocab
    identity, config_hash = source.identity(), sha256(source.path / 'config.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    with (tempfile.TemporaryDirectory(prefix='mfq-wt2-', dir=output.parent) as directory,
            ReferenceWriter(output, tokens, vocab) as writer):
        if backend == 'metal':
            from mfq.evaluation.reference_mlx import generate
            nll, execution = generate(source, tokens, writer, Path(directory), layerwise=mode['layerwise'])
        else:
            from mfq.evaluation.reference_cuda import generate
            nll, execution = generate(source, tokens, writer, Path(directory), layerwise=mode['layerwise'], runtime=runtime, device=device)
        if identity != source.identity() or config_hash != sha256(source.path / 'config.json'):
            raise ValueError('source model changed during reference generation')
        if execution.get('layerwise', mode['layerwise']) and not mode['layerwise']:
            mode.update(layerwise=True, fallback_reason='insufficient_memory')
        scored = chunks * (context_size - (context_size // 2 + 1))
        perplexity = math.exp(nll / scored)
        if not math.isfinite(perplexity):
            raise ValueError('non-finite native reference perplexity')
        writer.commit()
    try:
        return _save_manifest(source, dataset, output, manifest, tokens, tokenizer, identity,
            config_hash, execution, mode, device, context_size, chunks, scored, nll, perplexity)
    except BaseException:
        output.unlink(missing_ok=True)
        raise


def _save_manifest(source, dataset, output, manifest, tokens, tokenizer, identity,
        config_hash, execution, mode, device, context_size, chunks, scored, nll, perplexity):
    from mfq.server.services.evaluation_datasets import WT2_TEXT_SHA256
    document = {'format': 'mfq.perplexity-logits-manifest.v1',
        'producer': {'tool': 'mfq.wt2-reference', 'runtime': execution['runtime'], 'device': device},
        'reference': {'magic': '_logit3_', 'file': {'name': output.name, 'bytes': output.stat().st_size, 'sha256': sha256(output)},
            'distribution_encoding': {'type': 'linear_uint16_log_probability', 'logit_range': 16, 'code_max': 65535,
                'zero_code_is_clipped': True, 'target_log_probability': 'exact_float32'}},
        'model': {'label': source.path.name, 'architecture': source.architecture,
            'precision': {'classification': '+'.join(source.summary()['source_precisions']), 'execution_dtype': execution['dtype']},
            'config_sha256': config_hash, 'files': [{'name': name, 'bytes': size, 'modified_ns': modified} for name, size, modified in identity]},
        'dataset': {'id': 'wikitext2', 'file_name': dataset.name, 'bytes': dataset.stat().st_size, 'sha256': WT2_TEXT_SHA256},
        'tokenizer': tokenizer,
        'evaluation': {'protocol': 'llama.cpp-non-strided-second-half-v1', 'n_ctx': context_size,
            'n_batch': context_size, 'n_ubatch': context_size, 'n_seq': 1, 'chunks': chunks,
            'input_tokens': tokens.size, 'input_token_ids_sha256': hashlib.sha256(tokens.tobytes()).hexdigest(),
            'per_chunk_bos_replacement': tokenizer['add_bos'], 'target_start': context_size // 2 + 1,
            'score_count_per_chunk': context_size - (context_size // 2 + 1), 'scored_tokens': scored,
            'logit_positions': [context_size // 2, context_size - 2], 'target_positions': [context_size // 2 + 1, context_size - 1],
            'attention': execution['attention'], 'kv_cache_dtype': 'none'},
        'loading': mode, 'result': {'cross_entropy': nll / scored, 'perplexity': perplexity}}
    partial = manifest.with_name(manifest.name + '.partial')
    created = False
    try:
        with partial.open('x', encoding='utf-8') as stream:
            created = True
            json.dump(document, stream, ensure_ascii=False, allow_nan=False, indent=2)
        os.link(partial, manifest)
        partial.unlink()
    except BaseException:
        if created:
            partial.unlink(missing_ok=True)
        raise
    result = {'output': str(output), 'manifest': str(manifest), 'context_size': context_size, 'chunks': chunks,
        'parallel': 1, 'scored_tokens': scored, 'perplexity': document['result']['perplexity'], 'loading': mode}
    print(json.dumps({'event': 'wt2_reference_saved', **result}), flush=True)
    return result
