from __future__ import annotations

import gc
import json
import math
import mmap
import re

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mfq.formats.hf_source import _DTYPE_LAYOUTS


class SourceWeights:
    def __init__(self, source):
        self.source = source
        self.records = source.records
        self.prefix = next((prefix for prefix in ('model.language_model.', 'language_model.model.', 'model.')
            if prefix + 'embed_tokens.weight' in self.records), None)
        if self.prefix is None:
            raise ValueError('source token embedding was not found')
        self.embedding = self.prefix + 'embed_tokens.weight'
        self.head = next((name for name in ('lm_head.weight', 'language_model.lm_head.weight', 'model.lm_head.weight')
            if name in self.records), self.embedding)
        if source.text.get('tie_word_embeddings', False):
            self.head = self.embedding
        dtype = self.records[self.embedding].dtype
        self.dtype = mx.float32 if dtype == 'F32' else mx.float16 if dtype == 'F16' else mx.bfloat16

    def raw(self, name, start=None, end=None):
        record = self.records[name]
        row_size = math.prod(record.shape[1:]) if record.shape else 1
        begin, stop = (0, record.shape[0]) if start is None and record.shape else (start or 0, end)
        if not record.shape:
            begin, stop = 0, 1
        elif stop is None:
            stop = record.shape[0]
        if not 0 <= begin <= stop <= (record.shape[0] if record.shape else 1):
            raise ValueError('source row range is invalid')
        dtype = _DTYPE_LAYOUTS[record.dtype]
        with record.path.open('rb', buffering=0) as stream:
            stream.seek(record.offset + begin * row_size * dtype.itemsize)
            data = stream.read((stop - begin) * row_size * dtype.itemsize)
        if len(data) != (stop - begin) * row_size * dtype.itemsize:
            raise EOFError(f'truncated source tensor: {name}')
        shape = (stop - begin, *record.shape[1:]) if record.shape else ()
        return np.frombuffer(data, dtype=dtype).reshape(shape)

    def read(self, name, start=None, end=None):
        record = self.records[name]
        raw = self.raw(name, start, end)
        if record.dtype == 'BF16':
            return mx.array(raw).view(mx.bfloat16)
        if record.dtype not in {'F8_E4M3', 'F8_E4M3FN', 'F8_E5M2', 'I8', 'U8'}:
            return mx.array(raw)
        scale_name = next((candidate for candidate in (
            name.removesuffix('.weight') + '.weight_scale', name + '_scale_inv',
            name.removesuffix('.weight') + '.weight_scale_inv', name + '_scale') if candidate in self.records), None)
        if record.dtype in {'I8', 'U8'}:
            if scale_name is None or self.records[scale_name].dtype != 'F8_E8M0':
                raise ValueError(f'unsupported integer source weight: {name}')
            table = np.asarray((0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6), dtype=np.float32)
            values = np.stack((table[raw.view(np.uint8) & 15], table[raw.view(np.uint8) >> 4]), axis=-1)
            values = values.reshape(*raw.shape[:-1], raw.shape[-1] * 2)
        else:
            bits = raw.view(np.uint8)
            mantissa_bits = 2 if record.dtype == 'F8_E5M2' else 3
            bias = 15 if mantissa_bits == 2 else 7
            exponent = (bits & 127) >> mantissa_bits
            mantissa = (bits & ((1 << mantissa_bits) - 1)).astype(np.float32)
            values = np.where(exponent == 0, np.ldexp(mantissa, 1 - bias - mantissa_bits),
                np.ldexp(1 + mantissa / (1 << mantissa_bits), exponent.astype(np.int32) - bias))
            values = np.where(bits & 128, -values, values)
            if mantissa_bits == 3:
                values[(bits & 127) == 127] = np.nan
            else:
                values[exponent == 31] = np.nan
        if scale_name is not None:
            scale_record = self.records[scale_name]
            scales = self.raw(scale_name)
            if scale_record.dtype == 'F8_E8M0':
                invalid = scales == 255
                scales = np.ldexp(np.ones(scales.shape, dtype=np.float32), scales.astype(np.int32) - 127)
                scales[invalid] = np.nan
            elif scale_record.dtype == 'BF16':
                scales = (scales.astype(np.uint32) << 16).view(np.float32)
            scales = np.asarray(scales, dtype=np.float32)
            if scales.size != 1:
                if values.ndim not in (2, 3) or scales.ndim != values.ndim:
                    raise ValueError(f'unsupported source scale geometry: {name}')
                logical_columns = record.shape[-1] * (2 if record.dtype in {'I8', 'U8'} else 1)
                block = self.source.config.get('quantization_config', {}).get('weight_block_size')
                if block is None:
                    options = ((1, 32), (32, 32), (128, 128)) if scale_record.dtype == 'F8_E8M0' else ((128, 128),)
                    block = next((pair for pair in options if tuple(scales.shape[-2:]) ==
                        ((record.shape[-2] + pair[0] - 1) // pair[0], (logical_columns + pair[1] - 1) // pair[1])), None)
                if block is None:
                    raise ValueError(f'unsupported source scale geometry: {name}')
                row_start = (start or 0) if values.ndim == 2 else 0
                rows = np.arange(row_start, row_start + values.shape[-2]) // int(block[0])
                columns = np.arange(logical_columns) // int(block[1])
                if rows.max(initial=0) >= scales.shape[-2] or columns.max(initial=0) >= scales.shape[-1]:
                    raise ValueError(f'source scale shape does not match block size: {name}')
                if values.ndim == 3:
                    scales = scales[(start or 0):(start or 0) + values.shape[0]]
                scales = scales[..., rows[:, None], columns[None, :]]
            values = values * scales
        if not np.isfinite(values).all():
            raise ValueError(f'non-finite source weight: {name}')
        return mx.array(values).astype(self.dtype)

    def rows(self, name, ids):
        unique, inverse = np.unique(np.asarray(ids).reshape(-1), return_inverse=True)
        pieces = [self.read(name, int(index), int(index) + 1) for index in unique]
        result = mx.concatenate(pieces)[mx.array(inverse)]
        return result.reshape(*ids.shape, -1).astype(self.dtype)

    def layer_weights(self, index, gemma):
        prefix = self.prefix + f'layers.{index}.'
        result = {}
        for name in self.records:
            if not name.startswith(prefix) or any(marker in name for marker in ('weight_scale', 'input_scale', 'rotary_emb')):
                continue
            key = name[len(prefix):]
            value = self.read(name)
            if not gemma:
                if key.endswith('conv1d.weight'):
                    value = value.transpose(0, 2, 1)
                if key.endswith(('input_layernorm.weight', 'post_attention_layernorm.weight', 'q_norm.weight', 'k_norm.weight')):
                    value = value + 1
            if key.removesuffix('.weight').endswith('experts.gate_up_proj'):
                base = key.removesuffix('.weight').removesuffix('.gate_up_proj')
                base = base + '.switch_glu' if gemma else base.replace('experts', 'switch_mlp')
                gate, up = mx.split(value, 2, axis=-2)
                result[base + '.gate_proj.weight'] = mx.contiguous(gate)
                result[base + '.up_proj.weight'] = mx.contiguous(up)
            elif key.removesuffix('.weight').endswith('experts.down_proj'):
                base = key.removesuffix('.weight').removesuffix('.down_proj')
                base = base + '.switch_glu' if gemma else base.replace('experts', 'switch_mlp')
                result[base + '.down_proj.weight'] = value
            else:
                result[key] = value
        split = {}
        for key in list(result):
            match = re.fullmatch(r'(.*experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight', key)
            if match:
                prefix, expert, projection = match.groups()
                base = prefix + '.switch_glu' if gemma else prefix.replace('experts', 'switch_mlp')
                split.setdefault(base + '.' + projection + '.weight', {})[int(expert)] = result.pop(key)
        for key, experts in split.items():
            if sorted(experts) != list(range(int(self.source.text['num_experts']))):
                raise ValueError(f'incomplete reference expert bank: {key}')
            result[key] = mx.stack([experts[index] for index in sorted(experts)])
        return result


def generate(source, tokens, writer, directory, *, layerwise):
    from mlx_lm.models import gemma4_text, qwen3_5
    from mlx_lm.models.base import create_attention_mask
    weights = SourceWeights(source)
    gemma = source.architecture.removesuffix('_text') == 'gemma4'
    args = (gemma4_text.ModelArgs if gemma else qwen3_5.TextModelArgs).from_dict(dict(source.text))
    count, context = tokens.shape
    hidden_size, layers = int(args.hidden_size), int(args.num_hidden_layers)
    hidden = np.memmap(directory / 'hidden.f32', mode='w+', dtype='<f4', shape=(count, context, hidden_size))
    initial = np.memmap(directory / 'initial.f32', mode='w+', dtype='<f4', shape=hidden.shape) if gemma and args.hidden_size_per_layer_input else None
    resident = []
    norm_weight = None
    def release_mapping(mapping):
        mapping.flush()
        if hasattr(mapping._mmap, 'madvise'):
            mapping._mmap.madvise(mmap.MADV_DONTNEED)
    def make_layer(index):
        layer = (gemma4_text.DecoderLayer if gemma else qwen3_5.DecoderLayer)(args, index)
        if not gemma and source.text.get('layer_types'):
            expected_linear = source.text['layer_types'][index] == 'linear_attention'
            if expected_linear != layer.is_linear:
                raise ValueError('reference attention layout disagrees with full_attention_interval')
        state = weights.layer_weights(index, gemma)
        expected = {name for name, _ in tree_flatten(layer.parameters())}
        if gemma and not layer.self_attn.has_kv:
            state = {name: value for name, value in state.items() if not name.startswith(('self_attn.k_proj.', 'self_attn.v_proj.', 'self_attn.k_norm.'))}
        missing, extra = expected - state.keys(), state.keys() - expected
        if missing or extra:
            raise ValueError(f'reference layer {index} weight mismatch: missing={sorted(missing)}, unexpected={sorted(extra)}')
        layer.load_weights(list(state.items()), strict=True)
        layer.eval()
        mx.eval(layer.parameters())
        return layer
    try:
        for chunk, ids in enumerate(tokens):
            value = weights.rows(weights.embedding, ids[None])
            if gemma:
                value = value * mx.array(hidden_size**.5, dtype=value.dtype)
            mx.eval(value)
            hidden[chunk] = np.asarray(value[0].astype(mx.float32))
            release_mapping(hidden)
        release_mapping(hidden)
        if initial is not None:
            for chunk in range(count):
                initial[chunk] = hidden[chunk]
                release_mapping(initial)
                release_mapping(hidden)
            release_mapping(initial)
        if not layerwise:
            try:
                for index in range(layers):
                    resident.append(make_layer(index))
            except RuntimeError as error:
                if not any(marker in str(error).lower() for marker in ('out of memory', 'memory limit', 'allocation failed')):
                    raise
                resident.clear()
                gc.collect()
                mx.clear_cache()
                layerwise = True
                print(json.dumps({'event': 'source_loading', 'layerwise': True,
                    'requested_layerwise': False, 'fallback_reason': 'insufficient_memory'}), flush=True)
        shared_producers = {}
        if gemma and args.num_kv_shared_layers:
            boundary = layers - args.num_kv_shared_layers
            for index in range(boundary):
                shared_producers[args.layer_types[index]] = index
        for index in range(layers):
            layer = resident[index] if resident else make_layer(index)
            for chunk in range(count):
                value = mx.array(np.array(hidden[chunk:chunk + 1])).astype(weights.dtype)
                if gemma:
                    per_layer = None
                    if initial is not None:
                        width = args.hidden_size_per_layer_input
                        embedding = weights.rows(weights.prefix + 'embed_tokens_per_layer.weight', tokens[chunk:chunk + 1])
                        embedding = embedding[..., index * width:(index + 1) * width] * mx.array(width**.5, dtype=weights.dtype)
                        projection = weights.read(weights.prefix + 'per_layer_model_projection.weight', index * width, (index + 1) * width)
                        initial_value = mx.array(np.array(initial[chunk:chunk + 1])).astype(weights.dtype)
                        projected = (initial_value @ projection.T) * hidden_size**-.5
                        projected = mx.fast.rms_norm(projected, weights.read(weights.prefix + 'per_layer_projection_norm.weight'), args.rms_norm_eps)
                        per_layer = (projected + embedding) * 2**-.5
                    shared = None
                    producer = shared_producers.get(args.layer_types[index])
                    if not layer.self_attn.has_kv:
                        shared = tuple(mx.array(np.load(directory / f'kv-{producer}-{chunk}-{part}.npy')).astype(weights.dtype) for part in range(2))
                    mask = create_attention_mask(value, None, window_size=args.sliding_window if layer.layer_type == 'sliding_attention' else None)
                    value, kv, _offset = layer(value, mask=mask, per_layer_input=per_layer, shared_kv=shared, offset=0)
                    mx.eval(value, kv)
                    if producer == index:
                        for part, array in enumerate(kv):
                            np.save(directory / f'kv-{index}-{chunk}-{part}.npy', np.asarray(array.astype(mx.float32)))
                else:
                    value = layer(value, mask=None if layer.is_linear else create_attention_mask(value, None))
                    mx.eval(value)
                hidden[chunk] = np.asarray(value[0].astype(mx.float32))
                release_mapping(hidden)
                del value
            release_mapping(hidden)
            if initial is not None:
                release_mapping(initial)
            del layer
            if layerwise:
                gc.collect()
                mx.clear_cache()
            print(json.dumps({'event': 'wt2_layer', 'layer': index + 1, 'layers': layers}), flush=True)
        resident.clear()
        norm_weight = weights.read(weights.prefix + 'norm.weight')
        if not gemma:
            norm_weight = norm_weight + 1
        nll = 0.
        for chunk in range(count):
            for offset in range(0, writer.count, 32):
                stop = min(writer.count, offset + 32)
                value = mx.array(np.array(hidden[chunk, writer.start - 1 + offset:writer.start - 1 + stop])).astype(weights.dtype)
                value = mx.fast.rms_norm(value, norm_weight, args.rms_norm_eps)
                logits = np.empty((stop - offset, writer.vocab), dtype=np.float32)
                for row in range(0, writer.vocab, 4096):
                    matrix = weights.read(weights.head, row, min(writer.vocab, row + 4096))
                    result = value @ matrix.T
                    cap = source.text.get('final_logit_softcapping') or 0
                    if cap > 0:
                        result = mx.tanh(result / cap) * cap
                    logits[:, row:row + result.shape[-1]] = np.asarray(result.astype(mx.float32))
                    del matrix, result
                nll += writer.logits(chunk, offset, logits)
            print(json.dumps({'event': 'wt2_head', 'chunk': chunk + 1, 'chunks': count}), flush=True)
        return nll, {'runtime': 'mlx', 'dtype': str(weights.dtype).removeprefix('mlx.core.'),
            'attention': 'native_sdpa', 'layerwise': layerwise}
    finally:
        resident.clear()
        hidden._mmap.close()
        if initial is not None:
            initial._mmap.close()
        norm_weight = None
        gc.collect()
        mx.clear_cache()
