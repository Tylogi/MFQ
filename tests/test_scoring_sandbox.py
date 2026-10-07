import asyncio
import base64
import json
import os
import pickle
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

from mfq.server.services.accuracy_benchmark import Question, code_tests
from mfq.server.services.jobs import JobExecutionError
from mfq.server.services.scoring_sandbox import sandbox_available, sandbox_profile, scoring_environment, isolated_official_score
from mfq.cli import main


def test_packaged_scoring_uses_its_bundle_not_a_system_site_packages(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert scoring_environment() == (str(Path(sys.executable).resolve()), str(tmp_path))


def test_cli_scoring_probe_is_an_explicit_internal_entry_point(capsys):
    assert main(["_official-scoring-probe"]) == 0
    assert capsys.readouterr().out == "4\n"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox profile")
@pytest.mark.parametrize("action", ["read", "network", "fork", "execute", "write"])
def test_scoring_isolation_denies_user_files_network_processes_and_writes(tmp_path, action):
    assert sandbox_available()
    protected = tmp_path / "protected.txt"
    protected.write_text("dummy private data")
    worker = tmp_path / "worker"
    worker.mkdir()
    operations = {
        "read": f"open({str(protected)!r}).read()",
        "write": f"open({str(protected)!r},'w').write('changed')",
        "network": "__import__('socket').create_connection(('127.0.0.1',8090),timeout=1)",
        "fork": "__import__('os').fork()",
        "execute": "__import__('subprocess').run(['/bin/echo','unexpected'])",
    }
    code = "try:\n " + operations[action] + "\nexcept PermissionError:\n print('blocked')\nelse:\n print('unexpected')"
    executable, _ = scoring_environment()
    result = subprocess.run(["/usr/bin/sandbox-exec", "-p", sandbox_profile(worker), executable,
        "-I", "-S", "-c", code], cwd=worker, env={"PATH": "/usr/bin:/bin"}, capture_output=True, timeout=3)
    assert result.returncode == 0 and result.stdout.strip() == b"blocked", (result.stdout, result.stderr)
    assert protected.read_text() == "dummy private data"


def test_private_test_decoding_matches_plain_and_compressed_official_strings():
    tests = [{"input": "1\n2", "output": "3", "testtype": "functional"}]
    row = {"public_test_cases": "[]", "private_test_cases": json.dumps(tests), "metadata": '{"func_name":"add"}'}
    question = Question(question="Add numbers", answer="all-tests", protocol="code", official_row=row)
    expected = {"inputs": ["1\n2"], "outputs": ["3"], "fn_name": "add"}
    assert code_tests(question) == expected
    row["private_test_cases"] = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tests)))).decode()
    assert code_tests(question) == expected


def test_private_test_pickle_may_not_execute_objects():
    class Malicious:
        def __reduce__(self):
            return (os.system, ("false",))
    encoded = base64.b64encode(zlib.compress(pickle.dumps(Malicious()))).decode()
    question = Question(question="code", answer="all-tests", protocol="code",
        official_row={"public_test_cases": "[]", "private_test_cases": encoded, "metadata": "{}"})
    with pytest.raises(JobExecutionError, match="instantiate objects"):
        code_tests(question)


def test_official_worker_has_no_unsafe_fallback(monkeypatch):
    monkeypatch.setattr("mfq.server.services.scoring_sandbox.sandbox_available", lambda: False)
    with pytest.raises(JobExecutionError, match="no unsafe fallback"):
        asyncio.run(isolated_official_score({}, 1))


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox profile")
def test_scoring_profile_preserves_unicode_package_paths(tmp_path):
    worker = tmp_path / "工作目录"
    worker.mkdir()
    executable, packages = scoring_environment()
    code = f"import sys;sys.path.insert(0,{packages!r});import regex,numpy;print('ready')"
    result = subprocess.run(["/usr/bin/sandbox-exec", "-p", sandbox_profile(worker), executable,
        "-I", "-S", "-c", code], cwd=worker, env={"PATH": "/usr/bin:/bin"}, capture_output=True, timeout=3)
    assert result.returncode == 0 and result.stdout.strip() == b"ready", result.stderr
