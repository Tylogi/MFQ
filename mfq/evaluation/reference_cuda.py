from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np


def generate(source, tokens, writer, directory, *, layerwise, runtime, device):
    if source.architecture.removesuffix('_text') == 'gemma4' and (
            source.text.get('num_kv_shared_layers', 0) or source.text.get('hidden_size_per_layer_input', 0)):
        raise ValueError('native CUDA reference generation does not yet support Gemma4 shared KV or per-layer embeddings')
    executable = runtime or shutil.which('mfq-runtime')
    if not executable or not Path(executable).is_file():
        raise ValueError('native CUDA reference generation requires --runtime pointing to mfq-runtime')
    input_path, output = directory / 'tokens.bin', directory / 'logits.f32'
    with input_path.open('xb') as stream:
        stream.write(struct.pack('<II', *tokens.shape))
        stream.write(tokens.astype('<i4').tobytes())
    command = [str(executable), '--wt2-reference', '--model', str(source.path), '--ctx-size', str(tokens.shape[1]),
        '--reference-input', str(input_path), '--reference-output', str(output)]
    if not layerwise:
        command.append('--resident')
    environment = os.environ.copy()
    ordinal = int(device.partition(':')[2] or 0)
    visible = environment.get('CUDA_VISIBLE_DEVICES')
    if visible:
        devices = visible.split(',')
        if not 0 <= ordinal < len(devices):
            raise ValueError('CUDA reference device is not visible')
        environment['CUDA_VISIBLE_DEVICES'] = devices[ordinal]
    else:
        environment['CUDA_VISIBLE_DEVICES'] = str(ordinal)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True, env=environment)
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get('event') == 'source_loading' and event.get('layerwise'):
                layerwise = True
        code = process.wait()
        if code:
            raise RuntimeError(f'native reference generation failed with exit code {code}')
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        process.stdout.close()
    expected = writer.chunks * writer.count * writer.vocab * 4
    if not output.is_file() or output.stat().st_size != expected:
        raise ValueError('native reference output has incomplete logits')
    nll = 0.
    with output.open('rb') as stream:
        for chunk in range(writer.chunks):
            for offset in range(0, writer.count, 32):
                count = min(32, writer.count - offset)
                values = np.frombuffer(stream.read(count * writer.vocab * 4), dtype='<f4').reshape(count, writer.vocab)
                nll += writer.logits(chunk, offset, values)
    return nll, {'runtime': 'mfq-native-cuda', 'dtype': 'float16', 'attention': 'native_sdpa', 'layerwise': layerwise}
