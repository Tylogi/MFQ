"""Model-hub discovery and the curated MFQ model catalog."""

from __future__ import annotations

import asyncio
import os
import platform
import re
import time
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import unquote, urlparse

import httpx

from mfq.architectures.tensor_schema import tensor_schema_for_config
from mfq.server.protocol.models import (
    HubModelFile,
    HubModelInfo,
    HubModelSearchResult,
    HubModelSummary,
    HubModelVariant,
    HubSystemProfile,
    ModelConfigurationStatus,
    OfficialModelInfo,
    OfficialModelList,
    OfficialModelSource,
)
from mfq.server.runtime.host_memory import (
    host_memory_snapshot,
    metal_recommended_working_set_size,
    total_physical_memory,
)

HubProvider = Literal["huggingface", "modelscope"]

_REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_MFQ_SHARD_PATTERN = re.compile(
    r"^(?P<name>.+?)(?:-\d{5}-of-\d{5})?\.mfq$", re.IGNORECASE
)
_PRECISION_PATTERN = re.compile(
    r"(?:^|[-_.])((?:S|V|Q|NINT|NVQ)\d+(?:[-_][A-Za-z0-9]+)?)",
    re.IGNORECASE,
)
_GIB = 1 << 30


class HubError(RuntimeError):
    pass


@dataclass(frozen=True)
class _OfficialSourceSpec:
    provider: HubProvider
    repo_id: str

    @property
    def url(self) -> str:
        if self.provider == "huggingface":
            return f"https://huggingface.co/{self.repo_id}"
        return f"https://modelscope.cn/models/{self.repo_id}"


@dataclass(frozen=True)
class _OfficialModelSpec:
    id: str
    name: str
    family: str
    architecture: str
    description: str
    description_zh: str
    parameter_label: str | None
    active_parameter_label: str | None
    modalities: tuple[str, ...]
    capabilities: tuple[str, ...]
    precision_options: tuple[str, ...]
    license: str | None
    sources: tuple[_OfficialSourceSpec, ...]
    minimum_memory_gib: int
    recommended_memory_gib: int
    supports_ssd_streaming: bool = False


_OFFICIAL_MODELS = (
    _OfficialModelSpec(
        id="deepseek-v4-flash-0731",
        name="DeepSeek-V4-Flash-0731",
        family="DeepSeek-V4-Flash Series",
        architecture="deepseek_v4",
        description=(
            "Official native-QAT release with expert-wise MFQ variants, MTP, "
            "and SSD-streamed expert execution for memory-constrained systems."
        ),
        description_zh=(
            "官方原生 QAT 版本，提供逐专家 MFQ 精度版本、MTP，以及面向内存受限设备的 "
            "SSD 专家流式推理。"
        ),
        parameter_label="~160B total",
        active_parameter_label="~6B active per token",
        modalities=("text",),
        capabilities=("MTP", "MoE", "SSD streaming"),
        precision_options=("Expert-Wise V2",),
        license=None,
        sources=(
            _OfficialSourceSpec(
                "huggingface", "Tylogi/DeepSeek-V4-Flash-0731-EW-MFQ"
            ),
            _OfficialSourceSpec(
                "modelscope", "Tylogi/DeepSeek-V4-Flash-0731-EW-MFQ"
            ),
        ),
        minimum_memory_gib=48,
        recommended_memory_gib=96,
        supports_ssd_streaming=True,
    ),
    _OfficialModelSpec(
        id="qwen3-8-27b",
        name="Qwen3.8-27B",
        family="Qwen3.5–3.8",
        architecture="qwen3_5",
        description=(
            "Dense 27B MFQ release with scalar and vector-quantized quality tiers "
            "for high-throughput local inference."
        ),
        description_zh=(
            "27B 稠密 MFQ 版本，提供标量与向量量化的多档质量选择，面向高吞吐本地推理。"
        ),
        parameter_label="27B",
        active_parameter_label="27B active per token",
        modalities=("text", "image", "video"),
        capabilities=("Vision", "MTP"),
        precision_options=("V1–V4", "S4–S6"),
        license=None,
        sources=(_OfficialSourceSpec("modelscope", "Tylogi/Qwen3.8-27B-MFQ"),),
        minimum_memory_gib=24,
        recommended_memory_gib=32,
    ),
    _OfficialModelSpec(
        id="qwen3-6-27b",
        name="Qwen3.6-27B",
        family="Qwen3.5–3.8",
        architecture="qwen3_5",
        description=(
            "Dense 27B MFQ release spanning compact VQ and higher-fidelity SQ "
            "tiers for Apple and CUDA runtimes."
        ),
        description_zh=(
            "27B 稠密 MFQ 版本，覆盖紧凑 VQ 与更高保真 SQ 精度档，支持 Apple 与 CUDA 运行时。"
        ),
        parameter_label="27B",
        active_parameter_label="27B active per token",
        modalities=("text", "image", "video"),
        capabilities=("Vision", "MTP"),
        precision_options=("V2–V3", "S2–S6"),
        license=None,
        sources=(_OfficialSourceSpec("huggingface", "Tylogi/Qwen3.6-27B-MFQ"),),
        minimum_memory_gib=16,
        recommended_memory_gib=32,
    ),
    _OfficialModelSpec(
        id="minicpm-o-4-5",
        name="MiniCPM-o 4.5",
        family="MiniCPM-o",
        architecture="minicpmo",
        description=(
            "Omnimodal MFQ release for text, image, video, audio, speech output, "
            "and full-duplex interaction."
        ),
        description_zh=(
            "全模态 MFQ 版本，支持文本、图像、视频、音频、语音输出与全双工交互。"
        ),
        parameter_label=None,
        active_parameter_label=None,
        modalities=("text", "image", "video", "audio"),
        capabilities=("Vision", "Audio", "Speech output", "Full duplex"),
        precision_options=("S4–S8",),
        license=None,
        sources=(_OfficialSourceSpec("modelscope", "Tylogi/MiniCPM-o-4_5-MFQ"),),
        minimum_memory_gib=16,
        recommended_memory_gib=32,
    ),
)


def resolve_hub_reference(
    reference: str, fallback_provider: HubProvider = "huggingface"
) -> tuple[HubProvider, str, str | None]:
    """Normalize a repository id or trusted hub URL without following it."""

    value = reference.strip()
    if _REPOSITORY_PATTERN.fullmatch(value):
        return fallback_provider, value, None
    try:
        parsed = urlparse(value)
    except ValueError as error:
        raise HubError(f"invalid model repository reference: {error}") from error
    host = (parsed.hostname or "").lower()
    if parsed.username or parsed.password:
        raise HubError("model repository links must not contain credentials")
    if parsed.scheme not in {"http", "https"}:
        raise HubError("model repository links must use HTTP or HTTPS")
    if host in {"huggingface.co", "www.huggingface.co"}:
        provider: HubProvider = "huggingface"
    elif host in {"modelscope.cn", "www.modelscope.cn"}:
        provider = "modelscope"
    else:
        raise HubError("only Hugging Face and ModelScope repository links are supported")
    try:
        parts = [unquote(part) for part in parsed.path.split("/") if part]
    except UnicodeDecodeError as error:
        raise HubError("model repository link contains invalid path encoding") from error
    if provider == "modelscope" and parts[:1] == ["models"]:
        parts = parts[1:]
    if len(parts) < 2:
        raise HubError("model repository link is missing an owner or model name")
    repo_id = "/".join(parts[:2])
    if not _REPOSITORY_PATTERN.fullmatch(repo_id):
        raise HubError("model repository id contains unsupported characters")
    revision = None
    if len(parts) > 3 and parts[2] in {"tree", "blob"}:
        revision = "/".join(parts[3:])
    return provider, repo_id, revision


def system_profile(
    *, backend: str = "unknown", runtime_memory_budget_bytes: int | None = None
) -> HubSystemProfile:
    snapshot = host_memory_snapshot()
    physical = snapshot.total if snapshot is not None else total_physical_memory()
    available = snapshot.reclaimable(active_ratio=0.35) if snapshot is not None else None
    if backend == "metal":
        recommended = metal_recommended_working_set_size()
        if runtime_memory_budget_bytes is None:
            runtime_memory_budget_bytes = recommended
    normalized_backend = backend if backend in {"metal", "cuda", "cpu"} else "unknown"
    return HubSystemProfile(
        platform=platform.system() or "unknown",
        machine=platform.machine() or "unknown",
        backend=normalized_backend,
        physical_memory_bytes=physical,
        available_memory_bytes=available,
        runtime_memory_budget_bytes=runtime_memory_budget_bytes,
    )


def _configuration_status(
    byte_size: int,
    profile: HubSystemProfile,
    *,
    minimum_memory_bytes: int | None = None,
    recommended_memory_bytes: int | None = None,
    supported: bool | None = True,
    unsupported_reason: str | None = None,
) -> ModelConfigurationStatus:
    required = minimum_memory_bytes
    if required is None and byte_size > 0:
        required = max(byte_size + 2 * _GIB, int(byte_size * 1.08))
    recommended = recommended_memory_bytes
    if recommended is None and required is not None:
        recommended = max(required + 2 * _GIB, int(required * 1.12))
    capacity = profile.runtime_memory_budget_bytes or profile.physical_memory_bytes
    reasons: list[str] = []
    if supported is False:
        reasons.append(unsupported_reason or "This format is not directly loadable by MFQ.")
        status: Literal["recommended", "warning", "unknown"] = "warning"
    elif supported is None:
        reasons.append("Runtime compatibility could not be verified from repository metadata.")
        status = "unknown"
    elif required is None or capacity is None:
        reasons.append("Configuration requirements could not be determined.")
        status = "unknown"
    elif required > capacity:
        reasons.append("Estimated memory requirement exceeds the detected runtime budget.")
        status = "warning"
    else:
        reasons.append("Fits within the detected runtime memory budget.")
        status = "recommended"
        if recommended is not None and recommended > capacity:
            reasons.append("Close other memory-heavy applications before loading.")
    return ModelConfigurationStatus(
        status=status,
        required_memory_bytes=required,
        recommended_memory_bytes=recommended,
        available_memory_bytes=capacity,
        reasons=reasons,
    )


def _precision_from_name(name: str) -> str | None:
    matches = list(_PRECISION_PATTERN.finditer(name))
    return matches[-1].group(1).replace("_", "-").upper() if matches else None


def _model_variants(
    files: list[HubModelFile],
    profile: HubSystemProfile,
    *,
    runtime_compatible: bool | None = True,
) -> list[HubModelVariant]:
    variants: list[HubModelVariant] = []
    mfq_groups: dict[str, list[HubModelFile]] = {}
    hf_files: list[HubModelFile] = []
    for item in files:
        parent, separator, basename = item.name.rpartition("/")
        match = _MFQ_SHARD_PATTERN.match(basename)
        if match:
            stem = match.group("name")
            if not separator:
                label = stem
            elif parent.rsplit("/", 1)[-1] == stem:
                label = parent
            else:
                label = f"{parent}/{stem}"
            mfq_groups.setdefault(label, []).append(item)
        elif basename.lower().endswith(".gguf"):
            configuration = _configuration_status(
                item.byte_size,
                profile,
                supported=False,
                unsupported_reason="GGUF must be converted before the MFQ runtime can load it.",
            )
            variants.append(
                HubModelVariant(
                    id=item.name,
                    label=basename.removesuffix(".gguf"),
                    format="gguf",
                    precision=_precision_from_name(basename),
                    files=[item.name],
                    byte_size=item.byte_size,
                    configuration=configuration,
                )
            )
        elif basename.lower().endswith((".safetensors", ".bin")):
            hf_files.append(item)
    for label, group in mfq_groups.items():
        size = sum(item.byte_size for item in group)
        variants.append(
            HubModelVariant(
                id=f"mfq:{label}",
                    label=label,
                    format="mfq",
                    precision=_precision_from_name(label),
                    files=sorted(item.name for item in group)[:256],
                    byte_size=size,
                    configuration=_configuration_status(
                        size,
                        profile,
                        supported=runtime_compatible,
                        unsupported_reason="This model architecture is not registered in MFQ.",
                    ),
                )
        )
    if hf_files:
        size = sum(item.byte_size for item in hf_files)
        variants.append(
            HubModelVariant(
                id="hf:original",
                label="Original weights",
                format="hf",
                precision=None,
                files=sorted(item.name for item in hf_files)[:256],
                byte_size=size,
                configuration=_configuration_status(
                    size,
                    profile,
                    supported=runtime_compatible,
                    unsupported_reason="This model architecture is not registered in MFQ.",
                ),
            )
        )
    if not variants and files:
        size = sum(item.byte_size for item in files)
        variants.append(
            HubModelVariant(
                id="repository:all",
                label="Complete repository",
                format="unknown",
                precision=None,
                files=[],
                byte_size=size,
                configuration=_configuration_status(
                    size,
                    profile,
                    supported=False,
                    unsupported_reason=(
                        "The repository can be downloaded, but direct runtime compatibility "
                        "has not been verified."
                    ),
                ),
            )
        )
    return sorted(variants, key=lambda item: (item.byte_size <= 0, item.byte_size, item.label))


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        result = converter()
        return result if isinstance(result, dict) else {}
    return {}


def _optional_text(value: Any, *, limit: int = 2048) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        text = ", ".join(str(item) for item in value if item is not None)
    elif isinstance(value, dict):
        text = ", ".join(f"{key}: {item}" for key, item in value.items())
    else:
        text = str(value)
    text = " ".join(text.split()).strip()
    return text[:limit] or None


class HubCatalog:
    def __init__(self) -> None:
        self._official_cache: dict[tuple[HubProvider, str], HubModelInfo | None] = {}
        self._official_cache_time = 0.0
        self._official_lock = asyncio.Lock()

    async def search(
        self, provider: HubProvider, query: str, *, limit: int
    ) -> HubModelSearchResult:
        if provider == "huggingface":
            return await asyncio.to_thread(self._search_huggingface, query, limit)
        return await self._search_modelscope(query, limit)

    async def info(
        self,
        provider: HubProvider,
        repo_id: str,
        revision: str | None,
        *,
        profile: HubSystemProfile | None = None,
    ) -> HubModelInfo:
        profile = profile or system_profile()
        if provider == "huggingface":
            return await asyncio.to_thread(
                self._info_huggingface, repo_id, revision, profile
            )
        return await asyncio.to_thread(self._info_modelscope, repo_id, revision, profile)

    async def resolve(
        self,
        reference: str,
        fallback_provider: HubProvider,
        *,
        profile: HubSystemProfile | None = None,
    ) -> HubModelInfo:
        provider, repo_id, revision = resolve_hub_reference(reference, fallback_provider)
        return await self.info(provider, repo_id, revision, profile=profile)

    async def official(
        self, *, profile: HubSystemProfile | None = None, refresh: bool = False
    ) -> OfficialModelList:
        profile = profile or system_profile()
        await self._refresh_official_cache(profile, refresh=refresh)
        models = [self._official_model(spec, profile) for spec in _OFFICIAL_MODELS]
        return OfficialModelList(system=profile, data=models)

    async def _refresh_official_cache(
        self, profile: HubSystemProfile, *, refresh: bool
    ) -> None:
        if (
            not refresh
            and self._official_cache
            and time.monotonic() - self._official_cache_time < 900
        ):
            return
        async with self._official_lock:
            if (
                not refresh
                and self._official_cache
                and time.monotonic() - self._official_cache_time < 900
            ):
                return
            sources = {
                (source.provider, source.repo_id)
                for model in _OFFICIAL_MODELS
                for source in model.sources
            }

            async def inspect(provider: HubProvider, repo_id: str) -> HubModelInfo | None:
                try:
                    return await self.info(provider, repo_id, None, profile=profile)
                except HubError:
                    return None

            keys = sorted(sources)
            values = await asyncio.gather(*(inspect(*key) for key in keys))
            self._official_cache = dict(zip(keys, values, strict=True))
            self._official_cache_time = time.monotonic()

    def _official_model(
        self, spec: _OfficialModelSpec, profile: HubSystemProfile
    ) -> OfficialModelInfo:
        source_models = [
            self._official_cache.get((source.provider, source.repo_id))
            for source in spec.sources
        ]
        selected_index = next(
            (index for index, value in enumerate(source_models) if value is not None), 0
        )
        source_info = source_models[selected_index]
        sources = [
            OfficialModelSource(
                provider=source.provider,
                repo_id=source.repo_id,
                revision=(source_models[index].revision if source_models[index] else None),
                url=source.url,
                available=source_models[index] is not None,
            )
            for index, source in enumerate(spec.sources)
        ]
        selected_source = sources[selected_index]
        minimum = spec.minimum_memory_gib * _GIB
        recommended = spec.recommended_memory_gib * _GIB
        variants = []
        if source_info is not None:
            for variant in _model_variants(
                source_info.files, profile, runtime_compatible=True
            ):
                if spec.supports_ssd_streaming:
                    variant = variant.model_copy(
                        update={
                            "configuration": _configuration_status(
                                variant.byte_size,
                                profile,
                                minimum_memory_bytes=minimum,
                                recommended_memory_bytes=recommended,
                            )
                        }
                    )
                variants.append(variant)
        if variants and not spec.supports_ssd_streaming:
            smallest = min(variants, key=lambda item: (item.byte_size <= 0, item.byte_size))
            configuration = smallest.configuration.model_copy(deep=True)
            configuration.reasons.insert(
                0, "At least one published precision tier fits this configuration."
            )
        else:
            configuration = _configuration_status(
                min((item.byte_size for item in variants if item.byte_size > 0), default=0),
                profile,
                minimum_memory_bytes=minimum,
                recommended_memory_bytes=recommended,
            )
        if source_info is None:
            configuration.reasons.append(
                "The catalog is available offline; repository metadata could not be refreshed."
            )
        return OfficialModelInfo(
            id=spec.id,
            name=spec.name,
            family=spec.family,
            architecture=spec.architecture,
            description=spec.description,
            description_zh=spec.description_zh,
            parameter_label=spec.parameter_label,
            active_parameter_label=spec.active_parameter_label,
            modalities=list(spec.modalities),
            capabilities=list(spec.capabilities),
            precision_options=list(spec.precision_options),
            license=source_info.license if source_info and source_info.license else spec.license,
            supports_ssd_streaming=spec.supports_ssd_streaming,
            sources=sources,
            selected_source=selected_source,
            revision=source_info.revision if source_info else "main",
            downloads=source_info.downloads if source_info else 0,
            likes=source_info.likes if source_info else 0,
            updated_at=source_info.updated_at if source_info else None,
            variants=variants,
            configuration=configuration,
        )

    @staticmethod
    def _search_huggingface(query: str, limit: int) -> HubModelSearchResult:
        try:
            from huggingface_hub import HfApi

            models = HfApi().list_models(
                search=query,
                sort="downloads",
                limit=limit,
                expand=["downloads", "likes", "lastModified"],
                token=os.environ.get("HF_TOKEN") or None,
            )
            return HubModelSearchResult(
                data=[
                    HubModelSummary(
                        provider="huggingface",
                        repo_id=item.id,
                        source_url=f"https://huggingface.co/{item.id}",
                        author=item.id.split("/", 1)[0] if "/" in item.id else None,
                        downloads=item.downloads or 0,
                        likes=item.likes or 0,
                        updated_at=item.last_modified,
                    )
                    for item in models
                ]
            )
        except Exception as error:
            raise HubError(str(error)) from error

    @staticmethod
    def _info_huggingface(
        repo_id: str, revision: str | None, profile: HubSystemProfile
    ) -> HubModelInfo:
        try:
            from huggingface_hub import HfApi

            info = HfApi().model_info(
                repo_id,
                revision=revision,
                files_metadata=True,
                token=os.environ.get("HF_TOKEN") or None,
            )
            files = []
            for item in info.siblings or []:
                lfs = _mapping(getattr(item, "lfs", None))
                sha256 = lfs.get("sha256")
                files.append(
                    HubModelFile(
                        name=item.rfilename,
                        byte_size=item.size or int(lfs.get("size") or 0),
                        sha256=sha256 if isinstance(sha256, str) else None,
                    )
                )
            card = _mapping(getattr(info, "card_data", None))
            config = _mapping(getattr(info, "config", None))
            runtime_compatible = (
                tensor_schema_for_config(config) is not None if config else None
            )
            architectures = config.get("architectures") or []
            if isinstance(architectures, str):
                architectures = [architectures]
            safetensors = getattr(info, "safetensors", None)
            parameter_count = getattr(safetensors, "total", None)
            if not isinstance(parameter_count, int):
                parameter_count = None
            tags = list(info.tags or [])
            modalities = [
                modality
                for modality in ("text", "image", "video", "audio")
                if modality in " ".join(tags).lower()
            ]
            return HubModelInfo(
                provider="huggingface",
                repo_id=info.id,
                source_url=f"https://huggingface.co/{info.id}",
                author=getattr(info, "author", None)
                or (info.id.split("/", 1)[0] if "/" in info.id else None),
                description=_optional_text(
                    card.get("model_description") or card.get("model_name")
                ),
                revision=revision or info.sha or "main",
                downloads=info.downloads or 0,
                likes=info.likes or 0,
                total_bytes=sum(item.byte_size for item in files),
                updated_at=info.last_modified,
                files=files,
                tags=tags,
                license=_optional_text(card.get("license"), limit=255),
                library=_optional_text(getattr(info, "library_name", None), limit=255),
                pipeline_tag=_optional_text(
                    getattr(info, "pipeline_tag", None), limit=255
                ),
                architectures=[str(value) for value in architectures],
                modalities=modalities,
                parameter_count=parameter_count,
                gated=bool(getattr(info, "gated", False)),
                runtime_compatible=runtime_compatible,
                variants=_model_variants(
                    files, profile, runtime_compatible=runtime_compatible
                ),
            )
        except Exception as error:
            raise HubError(str(error)) from error

    @staticmethod
    async def _search_modelscope(query: str, limit: int) -> HubModelSearchResult:
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                response = await client.put(
                    "https://modelscope.cn/api/v1/models",
                    json={"PageSize": limit, "Name": query},
                )
                response.raise_for_status()
                payload = response.json().get("Data", {})
            values = payload.get("Models", payload.get("models", []))
            result = []
            for item in values:
                owner = item.get("Path") or ""
                name = item.get("Name") or ""
                repo_id = f"{owner}/{name}" if owner and name else name or owner
                if not repo_id:
                    continue
                result.append(
                    HubModelSummary(
                        provider="modelscope",
                        repo_id=repo_id,
                        source_url=f"https://modelscope.cn/models/{repo_id}",
                        author=owner or None,
                        description=_optional_text(
                            item.get("Description") or item.get("ChineseName")
                        ),
                        downloads=int(item.get("Downloads") or 0),
                        likes=int(item.get("Likes") or item.get("Stars") or 0),
                        total_bytes=int(item.get("StorageSize") or 0),
                    )
                )
            return HubModelSearchResult(data=result[:limit])
        except Exception as error:
            raise HubError(str(error)) from error

    @staticmethod
    def _info_modelscope(
        repo_id: str, revision: str | None, profile: HubSystemProfile
    ) -> HubModelInfo:
        try:
            from modelscope_hub import HubApi

            api = HubApi()
            model = api.get_repo(repo_id, "model", revision=revision)
            files = [
                HubModelFile(
                    name=item.path,
                    byte_size=int(item.size or 0),
                    sha256=(
                        str(item.sha256)
                        if getattr(item, "sha256", None)
                        and re.fullmatch(r"[0-9a-f]{64}", str(item.sha256))
                        else None
                    ),
                )
                for item in api.list_repo_files(
                    repo_id, "model", revision=revision, recursive=True
                )
            ]
            tags = list(getattr(model, "tags", None) or [])
            license_name = getattr(model, "license", None)
            return HubModelInfo(
                provider="modelscope",
                repo_id=model.id,
                source_url=f"https://modelscope.cn/models/{model.id}",
                author=model.id.split("/", 1)[0] if "/" in model.id else None,
                description=_optional_text(getattr(model, "description", None)),
                revision=revision or "master",
                downloads=int(model.downloads or 0),
                likes=int(model.likes or 0),
                total_bytes=sum(item.byte_size for item in files),
                updated_at=model.last_modified,
                files=files,
                tags=tags,
                license=_optional_text(license_name, limit=255),
                modalities=[
                    modality
                    for modality in ("text", "image", "video", "audio")
                    if modality in " ".join(tags).lower()
                ],
                runtime_compatible=None,
                variants=_model_variants(
                    files, profile, runtime_compatible=None
                ),
            )
        except Exception as error:
            raise HubError(str(error)) from error


__all__ = [
    "HubCatalog",
    "HubError",
    "HubProvider",
    "resolve_hub_reference",
    "system_profile",
]
