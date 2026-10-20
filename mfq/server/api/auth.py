"""Scoped API key issuance and authentication."""

from __future__ import annotations

import hashlib
import asyncio
import json
import os
import secrets
import tempfile
from dataclasses import dataclass
from pathlib import Path

from mfq.server.protocol.models import (
    ApiKeyResource,
    ApiKeyScope,
    ApiKeySecretResource,
    CreateApiKeyRequest,
)
from mfq.server.state.storage import SessionStore

ROLE_SCOPES = {
    "viewer": frozenset({"inference"}),
    "operator": frozenset({"inference", "models", "jobs"}),
    "administrator": frozenset({"admin"}),
}


def hash_api_key(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AuthenticatedKey:
    resource: ApiKeyResource | None
    scopes: frozenset[ApiKeyScope]


class ApiKeyManager:
    def __init__(self, store: SessionStore, root_key: str = "") -> None:
        self.store = store
        self.root_key = root_key

    def create(self, request: CreateApiKeyRequest) -> ApiKeySecretResource:
        token = f"mfq_{secrets.token_urlsafe(32)}"
        resource = self.store.create_api_key(
            request,
            hash_api_key(token),
            token[:12],
        )
        return ApiKeySecretResource(key=resource, token=token)

    def authenticate(self, token: str, *, include_root: bool = True) -> AuthenticatedKey | None:
        if include_root and self.root_key and secrets.compare_digest(token, self.root_key):
            return AuthenticatedKey(resource=None, scopes=frozenset({"admin"}))
        resource = self.store.authenticate_api_key(hash_api_key(token))
        if resource is None:
            return None
        scopes = frozenset(resource.scopes)
        if resource.role is not None:
            scopes = scopes | ROLE_SCOPES[resource.role]
        return AuthenticatedKey(resource=resource, scopes=scopes)

    def rotate(self, key_id) -> ApiKeySecretResource:
        token = f"mfq_{secrets.token_urlsafe(32)}"
        resource = self.store.rotate_api_key(
            key_id,
            hash_api_key(token),
            token[:12],
        )
        return ApiKeySecretResource(key=resource, token=token)

    @staticmethod
    def permits(authenticated: AuthenticatedKey, required: ApiKeyScope) -> bool:
        return "admin" in authenticated.scopes or required in authenticated.scopes


class AuthenticationPolicy:
    def __init__(self, root_key: str = "", *, enabled: bool = False, path: Path | None = None) -> None:
        self.path = path
        self._lock = asyncio.Lock()
        self._configuration = (enabled or bool(root_key), hash_api_key(root_key) if root_key else "")
        if path is not None and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            key_hash = data.get("key_hash", "")
            active = data.get("enabled")
            if not isinstance(active, bool) or not isinstance(key_hash, str) or (key_hash and (len(key_hash) != 64 or any(c not in "0123456789abcdef" for c in key_hash))) or (active and not key_hash):
                raise ValueError("invalid saved authentication policy")
            self._configuration = (active, key_hash)

    @property
    def enabled(self) -> bool:
        return self._configuration[0]

    def status(self) -> dict[str, bool]:
        enabled, key_hash = self._configuration
        return {"enabled": enabled, "has_key": bool(key_hash)}

    def matches(self, token: str) -> bool:
        key_hash = self._configuration[1]
        return bool(key_hash) and secrets.compare_digest(hash_api_key(token), key_hash)

    def _save(self, enabled: bool, key_hash: str) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(dir=self.path.parent, prefix=".auth-")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump({"enabled": enabled, "key_hash": key_hash}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    async def configure(self, enabled: bool, api_key: str | None) -> dict[str, bool]:
        async with self._lock:
            if api_key is not None and (not api_key or len(api_key) > 512 or any(ord(c) < 33 or ord(c) > 126 for c in api_key)):
                raise ValueError("API key must contain 1–512 printable ASCII characters without spaces")
            key_hash = hash_api_key(api_key) if api_key is not None else self._configuration[1]
            if enabled and not key_hash:
                raise ValueError("an API key is required to enable authentication")
            await asyncio.to_thread(self._save, enabled, key_hash)
            self._configuration = (enabled, key_hash)
            return self.status()


def required_scope(method: str, path: str) -> ApiKeyScope:
    if path == "/api/v1/hub/download-directory":
        return "models"
    if path.startswith("/api/v1/runtime/cache/entries") or path == "/api/v1/runtime/cache/purge":
        return "admin"
    if path in {"/api/v1/runtime/listener", "/api/v1/runtime/memory-policy", "/api/v1/runtime/model-aliases", "/api/v1/runtime/context-policy"} and method != "GET":
        return "admin"
    if path.startswith("/api/v1/auth/"):
        return "admin"
    if path.startswith("/api/v1/cluster/"):
        return "admin"
    if path.startswith("/api/v1/models/directories"):
        return "models"
    if path.startswith(("/api/v1/models", "/api/v1/runtime", "/api/v1/hub")):
        return "models" if method != "GET" else "inference"
    if path.startswith(
        ("/api/v1/jobs", "/api/v1/artifacts", "/api/v1/datasets", "/api/v1/evaluations", "/api/v1/quantization")
    ):
        return "jobs"
    return "inference"
