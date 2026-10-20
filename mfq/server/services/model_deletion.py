from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

from mfq.formats.shards import matching_shard_paths, parse_shard_path
from mfq.server.protocol.models import ModelDeletionPlan
from mfq.server.state.catalog import DiscoveredModel


class ModelDeletionError(ValueError):
    pass


def model_deletion_plan(artifact: DiscoveredModel) -> ModelDeletionPlan:
    source = artifact.path
    root = source if artifact.resource.format == "hf" else source.parent
    if artifact.resource.format == "mfq":
        try:
            parsed = parse_shard_path(source)
        except ValueError:
            parsed = None
        files = [path for path in matching_shard_paths(parsed[0])
            if parse_shard_path(path)[2] == parsed[2]] if parsed else [source]
    else:
        files = [source / "config.json", *source.glob("*.safetensors")]
        index = source / "model.safetensors.index.json"
        if index.is_file():
            document = json.loads(index.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                raise ModelDeletionError("model_delete_unsafe")
            weight_map = document.get("weight_map", {})
            if not isinstance(weight_map, dict):
                raise ModelDeletionError("model_delete_unsafe")
            for name in weight_map.values():
                if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts or Path(name).suffix != ".safetensors":
                    raise ModelDeletionError("model_delete_unsafe")
                path = source / name
                if path.is_file() or path.is_symlink():
                    files.append(path)
            files.append(index)
    rows = []
    total = 0
    for path in sorted(set(files)):
        if not path.parent.resolve().is_relative_to(root.resolve()):
            raise ModelDeletionError("model_delete_unsafe")
        info = path.lstat()
        if not (stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)):
            raise ModelDeletionError("model_delete_unsafe")
        rows.append((str(path), info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns))
        total += info.st_size
    if not rows:
        raise ModelDeletionError("model_delete_changed")
    token = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
    return ModelDeletionPlan(model_id=artifact.resource.id, name=artifact.resource.name,
        files=[row[0] for row in rows], total_bytes=total, token=token)


def delete_model_files(artifact: DiscoveredModel, token: str) -> dict[str, int]:
    plan = model_deletion_plan(artifact)
    if plan.token != token:
        raise ModelDeletionError("model_delete_changed")
    order = {"model.safetensors.index.json": 1, "config.json": 2}
    for name in sorted(plan.files, key=lambda value: order.get(Path(value).name, 0)):
        Path(name).unlink()
    return {"files": len(plan.files), "total_bytes": plan.total_bytes}
