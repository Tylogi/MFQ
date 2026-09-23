from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("module", "group"),
    [
        ("auth", "api"),
        ("network", "api"),
        ("openai_compat", "api"),
        ("openapi", "api"),
        ("backend", "runtime"),
        ("capabilities", "runtime"),
        ("cluster", "runtime"),
        ("hf_tokenizer", "runtime"),
        ("host_memory", "runtime"),
        ("native", "runtime"),
        ("runtime_pool", "runtime"),
        ("catalog", "state"),
        ("storage", "state"),
        ("components", "services"),
        ("documents", "services"),
        ("hub", "services"),
        ("jobs", "services"),
        ("mcp", "services"),
        ("service", "services"),
        ("tool_jobs", "services"),
        ("models", "protocol"),
        ("input_protocols", "protocol"),
        ("output_protocols", "protocol"),
        ("processor_prompt_protocols", "protocol"),
        ("deepseek_v41_prompt", "protocol"),
        ("deepseek_v4_prompt", "protocol"),
        ("dsml", "protocol"),
        ("reasoning", "protocol"),
        ("xml_tool_calls", "protocol"),
    ],
)
def test_server_modules_live_only_in_functional_packages(
    module: str,
    group: str,
) -> None:
    module_name = f"mfq.server.{group}.{module}"
    grouped = importlib.import_module(module_name)
    server_root = Path(__file__).resolve().parents[1] / "mfq" / "server"

    assert grouped.__name__ == module_name
    assert Path(grouped.__file__).resolve() == server_root / group / f"{module}.py"
    assert not (server_root / f"{module}.py").exists()


def test_api_routes_are_split_by_feature() -> None:
    api_root = Path(__file__).resolve().parents[1] / "mfq" / "server" / "api"
    route_files = {
        path.name: path.read_text(encoding="utf-8")
        for path in (api_root / "routes").glob("*.py")
    }
    expected = {
        "auth.py",
        "jobs.py",
        "mcp.py",
        "media.py",
        "models.py",
        "openai.py",
        "runtime.py",
        "sessions.py",
    }
    assert expected <= route_files.keys()
    assert all("APIRouter()" in route_files[name] for name in expected)
    assert all(
        "def register_routes(" not in route_files[name]
        for name in expected
    )
    application = (api_root / "__init__.py").read_text(encoding="utf-8")
    assert "app.include_router(" in application
    assert not any(
        decorator in application
        for decorator in ("@app.get(", "@app.post(", "@app.put(", "@app.patch(", "@app.delete(", "@app.websocket(")
    )
