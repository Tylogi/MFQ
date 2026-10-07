from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from mfq.server.protocol.models import ProtocolModel


class QuantizationDirectories(ProtocolModel):
    import_directory: str = Field(min_length=1, max_length=4096)
    export_directory: str = Field(min_length=1, max_length=4096)


class QuantizationWorkspace(QuantizationDirectories):
    candidates: list[str]
    candidate_groups: dict[str, list[str]] = Field(default_factory=dict)
    official_imatrix_url: str | None = None


class QuantizationFile(ProtocolModel):
    name: str
    path: str
    directory: bool
    byte_size: int | None = None


class QuantizationFiles(ProtocolModel):
    path: str
    parent: str | None
    data: list[QuantizationFile]


class QuantizationSourceRequest(ProtocolModel):
    path: str = Field(min_length=1, max_length=4096)


class SourceLoadingRequest(QuantizationSourceRequest):
    purpose: Literal['imatrix', 'wt2'] = 'imatrix'
    backend: Literal['auto', 'metal', 'cuda'] = 'auto'
    context_size: int = Field(default=16384, ge=2, le=1048576)
    batch_size: int = Field(default=1, ge=1, le=64)


class SourceLoadingPlan(ProtocolModel):
    backend: str
    resident_required_bytes: int
    available_bytes: int
    resident_allowed: bool
    weights_bytes: int
    workspace_bytes: int


class Wt2ReferencePayload(ProtocolModel):
    model: str = Field(min_length=1, max_length=4096)
    dataset_id: UUID
    output: str = Field(min_length=1, max_length=4096)
    backend: Literal['auto', 'metal', 'cuda'] = 'auto'
    context_size: int = Field(default=512, ge=32, le=1048576)
    chunks: int = Field(default=8, ge=1, le=10000)
    layerwise: bool = True


class QuantizationSource(ProtocolModel):
    path: str
    architecture: str
    tensors: int
    parameters: int
    format: Literal['hf', 'mfq']
    full_precision: bool
    source_precisions: list[str] = Field(default_factory=list)
    eligible_candidates: list[str] = Field(default_factory=list)
    imatrix_supported: bool = False


class RecipePayload(ProtocolModel):
    input: str = Field(min_length=1, max_length=4096)
    output: str = Field(min_length=1, max_length=4096)
    target_bpw: float = Field(gt=0, le=32, allow_inf_nan=False)
    method: Literal['DF-V1-AlphaQ'] = 'DF-V1-AlphaQ'
    candidates: list[str] = Field(min_length=1, max_length=32)
    service_policy: Literal['preserve', 'allow'] = 'preserve'
    backend: Literal['auto', 'cpu', 'metal', 'cuda'] = 'auto'


class FitPayload(ProtocolModel):
    input: str = Field(min_length=1, max_length=4096)
    recipe: str = Field(min_length=1, max_length=4096)
    output: str = Field(min_length=1, max_length=4096)
    imatrix: str | None = Field(default=None, max_length=4096)
    service_policy: Literal['preserve', 'allow'] = 'preserve'
    backend: Literal['auto', 'cpu', 'metal', 'cuda'] = 'auto'
    row_chunk: int = Field(default=1024, ge=8, le=8192)

    @model_validator(mode='after')
    def aligned(self):
        if self.row_chunk % 8:
            raise ValueError('row batch must be a multiple of eight')
        return self


class WorkbenchImportPayload(ProtocolModel):
    kind: Literal['recipe', 'imatrix']
    media_id: UUID | None = None
    source: str | None = Field(default=None, max_length=4096)
    output: str | None = Field(default=None, max_length=4096)

    @model_validator(mode='after')
    def source_required(self):
        if (self.media_id is None) == (self.source is None):
            raise ValueError('provide a local source or an uploaded file')
        if self.media_id is not None and not self.output:
            raise ValueError('uploaded artifacts require an output path')
        return self


class WorkbenchImatrixPayload(ProtocolModel):
    model: str = Field(min_length=1, max_length=4096)
    corpus: str = Field(min_length=1, max_length=4096)
    output: str = Field(min_length=1, max_length=4096)
    backend: Literal['auto', 'metal', 'cuda'] = 'auto'
    window_length: int = Field(default=16384, ge=2, le=1048576)
    train_tokens: int = Field(default=1572864, ge=2)
    apply_chat_template: bool = True
    layerwise: bool = True
