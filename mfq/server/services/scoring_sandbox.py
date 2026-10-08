from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import sysconfig
import tempfile
import psutil
from functools import lru_cache
from pathlib import Path
from typing import Any

from mfq.server.services.evaluation_datasets import dataset_error


def scoring_environment() -> tuple[str, str]:
    packages = str(sys._MEIPASS) if getattr(sys, "frozen", False) else sysconfig.get_path("purelib")
    return str(Path(sys.executable).resolve()), packages


def sandbox_profile(directory: Path) -> str:
    executable, packages = scoring_environment()
    paths = ["/System", "/usr/lib", sys.base_prefix, packages, str(directory.resolve())]
    reads = " ".join(f"(subpath {json.dumps(path, ensure_ascii=False)})" for path in paths)
    parents = " ".join(f"(literal {json.dumps(str(path), ensure_ascii=False)})" for path in Path(executable).parents)
    return f'(version 1)(deny default)(allow process-exec (literal {json.dumps(executable, ensure_ascii=False)}))(allow sysctl-read)(allow file-read-metadata {parents})(allow file-read* (literal "/") (literal {json.dumps(executable, ensure_ascii=False)}) {reads} (literal "/dev/urandom") (literal "/dev/null"))(allow file-write* (subpath {json.dumps(str(directory.resolve()), ensure_ascii=False)}) (literal "/dev/null"))'


@lru_cache(maxsize=1)
def sandbox_available() -> bool:
    if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        return False
    try:
        with tempfile.TemporaryDirectory(prefix="mfq-scoring-probe-") as directory:
            executable, _ = scoring_environment()
            arguments = ["_official-scoring-probe"] if getattr(sys, "frozen", False) else ["-I", "-S", "-c", "print(4)"]
            result = subprocess.run(["/usr/bin/sandbox-exec", "-p", sandbox_profile(Path(directory)),
                executable, *arguments], cwd=directory, env={"PATH": "/usr/bin:/bin"},
                capture_output=True, timeout=3)
        return result.returncode == 0 and result.stdout.strip() == b"4"
    except (OSError, subprocess.SubprocessError):
        return False


async def isolated_official_score(request: dict[str, Any], timeout: float) -> dict[str, Any]:
    if not sandbox_available():
        raise dataset_error("scoring_sandbox_unavailable", "official code and symbolic scoring requires an isolated worker; no unsafe fallback is allowed")
    executable, packages = scoring_environment()
    request = {**request, "site_packages": packages}
    if getattr(sys, "frozen", False):
        arguments = ["_official-scoring-worker"]
    else:
        worker = Path(__file__).with_name("official_scoring_worker.py").read_text(encoding="utf-8")
        arguments = ["-I", "-S", "-c", worker]
    with tempfile.TemporaryDirectory(prefix="mfq-official-worker-") as directory:
        process = await asyncio.create_subprocess_exec("/usr/bin/sandbox-exec", "-p", sandbox_profile(Path(directory)),
            executable, *arguments, cwd=directory,
            env={"PATH": "/usr/bin:/bin", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=True)
        async def read_limited(stream):
            output = bytearray()
            while True:
                chunk = await stream.read(65536)
                if not chunk:
                    return bytes(output)
                output.extend(chunk)
                if len(output) > 1024 * 1024:
                    if process.returncode is None:
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGKILL)
                    raise dataset_error("official_worker_failed", "official worker output exceeded its limit; no score was recorded")
        async def communicate():
            stdout = asyncio.create_task(read_limited(process.stdout))
            stderr = asyncio.create_task(read_limited(process.stderr))
            async def monitor_memory():
                while process.returncode is None:
                    try:
                        if psutil.Process(process.pid).memory_info().rss > 4 * 1024 ** 3:
                            os.killpg(process.pid, signal.SIGKILL)
                            raise dataset_error("official_worker_failed", "official worker exceeded its 4 GiB RSS limit; no score was recorded")
                    except psutil.NoSuchProcess:
                        return
                    await asyncio.sleep(.05)
            memory = asyncio.create_task(monitor_memory())
            try:
                process.stdin.write(json.dumps(request).encode("utf-8"))
                await process.stdin.drain()
                process.stdin.close()
                await process.wait()
                await memory
                return await stdout, await stderr
            finally:
                for task in (stdout, stderr, memory):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(stdout, stderr, memory, return_exceptions=True)
        try:
            stdout, stderr = await asyncio.wait_for(communicate(), timeout=timeout)
            if process.returncode != 0:
                raise dataset_error("official_worker_failed", f"official worker exited abnormally: {stderr.decode(errors='replace')[:500]}; no score was recorded")
            result = json.loads(stdout)
            if not isinstance(result, dict) or not isinstance(result.get("correct"), bool):
                raise ValueError("invalid official score")
            return result
        except (TimeoutError, ValueError) as error:
            raise dataset_error("official_worker_failed", f"official worker failed: {error}; no score was recorded") from error
        finally:
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
