"""Resolve browser load defaults against the model/backend's capabilities."""
from mfq.server.protocol.models import (
    KvMemorySettings, KvQuantizationSettings, ModelLoadRequest, QsaKvOffloadPolicy,
)


def apply_load_context(request: ModelLoadRequest, architecture: str, backend: str) -> ModelLoadRequest:
    defaults = request.context_management
    if defaults is None:
        return request
    kind = architecture.casefold()
    qsa = kind.startswith('qwen4_exp') and backend in {'cuda', 'metal'}
    quantized = backend in {'cuda', 'metal'} and kind.startswith(('qwen4_exp', 'qwen3.5', 'qwen3_5', 'qwen35'))
    manual = not defaults.automatic
    changes = {}
    # A load can still specify exact per-model settings. Browser defaults only
    # fill missing fields, and become explicit before saved settings are read.
    if 'kv_quantization' not in request.model_fields_set:
        changes['kv_quantization'] = defaults.kv_quantization if manual and quantized else KvQuantizationSettings()
    if 'kv_memory' not in request.model_fields_set:
        changes['kv_memory'] = KvMemorySettings(
            placement='auto_ssd' if manual and qsa and backend == 'cuda' and defaults.ssd_kv else 'auto',
            ram_preallocated=defaults.ram_preallocated if manual and qsa and backend == 'cuda' else True,
        )
    if 'qsa_kv_offload' not in request.model_fields_set:
        changes['qsa_kv_offload'] = QsaKvOffloadPolicy(enabled=manual and qsa and backend == 'metal' and defaults.ssd_kv)
    return request.model_copy(update=changes)
