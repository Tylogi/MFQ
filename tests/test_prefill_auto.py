"""Auto is the public default; explicit old chunk limits remain real overrides."""
from pathlib import Path

import pytest
from pydantic import ValidationError

from mfq.cli import _build_parser
from mfq.server.protocol.models import ModelLoadRequest
from mfq.server.runtime.native import NativeRuntime


def test_prefill_auto_defaults_and_saved_profile_roundtrip():
    assert _build_parser().parse_args(["serve"]).prefill_chunk_size == "auto"
    assert _build_parser().parse_args(["serve", "--prefill-chunk-size", "auto"]).prefill_chunk_size == "auto"
    request = ModelLoadRequest(model="test")
    assert request.prefill_chunk_size == "auto"
    assert ModelLoadRequest.model_validate_json(request.model_dump_json()) == request


@pytest.mark.parametrize("value", [1, 333, 2048, 4096, 8192])
@pytest.mark.parametrize("backend", ["cuda", "metal", "hip"])
def test_explicit_prefill_limit_is_forwarded(tmp_path: Path, value: int, backend: str):
    request = ModelLoadRequest(model="test", prefill_chunk_size=value)
    runtime = NativeRuntime(tmp_path / "runtime", tmp_path / "model", "test", backend,
                            prefill_chunk_size=request.prefill_chunk_size)
    command = runtime.command(8090)
    assert command[command.index("--prefill-chunk-size") + 1] == str(value)
    assert _build_parser().parse_args(["serve", "--prefill-chunk-size", str(value)]).prefill_chunk_size == value


@pytest.mark.parametrize("value", [0, -1, 8193, "automatic"])
def test_invalid_prefill_limit_is_rejected(value):
    with pytest.raises(ValidationError):
        ModelLoadRequest(model="test", prefill_chunk_size=value)


@pytest.mark.parametrize("backend", ["cuda", "metal", "hip"])
def test_auto_uses_native_backend_policy(tmp_path: Path, backend: str):
    runtime = NativeRuntime(tmp_path / "runtime", tmp_path / "model", "test", backend)
    assert "--prefill-chunk-size" not in runtime.command(8090)
