import contextlib
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench import model_bench
from mfq.server.services.inference_benchmark import validate_benchmark_run
from mfq.server.services.jobs import JobExecutionError


def native_result():
    return {
        "metrics": {"prefill_tokens": 32, "prefill_ms": 10, "decode_ms": 20,
                    "ttft_ms": 12, "mtp_used": 0, "mtp_drafted_tokens": 0,
                    "mtp_accepted_tokens": 0},
        "usage": {"prompt_tokens": 32, "completion_tokens": 8},
        "finish_reason": "length",
    }


@pytest.fixture
def runtime_script(tmp_path):
    def create(body):
        path = tmp_path / "runtime.py"
        path.write_text(
            f"#!{sys.executable}\n"
            "import json, sys, time\n"
            f"result = {native_result()!r}\n"
            "def send(kind, data=None, request_id=None):\n"
            "    frame = {'v': 1, 'type': kind}\n"
            "    if kind != 'ready': frame['id'] = request['id'] if request_id is None else request_id\n"
            "    if data is not None: frame['data'] = data\n"
            "    print(json.dumps(frame), flush=True)\n"
            "send('ready')\n"
            "for line in sys.stdin:\n"
            "    request = json.loads(line)\n"
            "    if request.get('op') == 'shutdown': break\n"
            + textwrap.indent(textwrap.dedent(body), "    ")
        )
        path.chmod(0o755)
        return path
    return create


def process(path):
    return model_bench.RuntimeProcess([sys.executable, "-u", str(path)], os.environ, path.parent)


def test_correlated_result_ignores_other_requests(runtime_script):
    path = runtime_script("""
        send('event', {'metrics': {}, 'usage': {}}, 'stale')
        send('done', request_id='stale')
        send('event', result)
        send('done')
    """)
    runtime = process(path)
    try:
        result = runtime.generate("current", 32, 8, False, 2)
        assert result["status"] == "ok"
        assert result["metrics"]["prompt_tokens"] == 32
        assert result["metrics"]["prefill_tps"] == 3200
        assert result["metrics"]["decode_tps"] == 350
    finally:
        runtime.close()
    assert runtime.proc.returncode == 0


@pytest.mark.parametrize("body", [
    "time.sleep(2)\nsend('event', result)\nsend('done')\n",
    "for _ in range(200):\n    send('event', {'event': 'progress'})\n    time.sleep(.01)\n",
])
def test_timeout_is_request_wide_and_terminates_runtime(runtime_script, body):
    runtime = process(runtime_script(body))
    started = time.monotonic()
    try:
        assert runtime.generate("first", 32, 8, False, .1)["status"] == "timeout"
        assert time.monotonic() - started < 1.5
        assert runtime.proc.poll() is not None
        assert not runtime.stdout_thread.is_alive()
        assert runtime.generate("second", 32, 8, False, .1)["status"] == "error"
    finally:
        runtime.close()


@pytest.mark.parametrize("body", [
    "send('done')\n",
    "print('not-json', flush=True)\n",
    "print('[]', flush=True)\n",
    "sys.exit(0)\n",
])
def test_incomplete_or_malformed_response_fails(runtime_script, body):
    runtime = process(runtime_script(body))
    try:
        assert runtime.generate("broken", 32, 8, False, 1)["status"] == "error"
        assert runtime.proc.poll() is not None
    finally:
        runtime.close()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process group cleanup")
def test_timeout_reaps_wrapper_children_holding_output_pipes(runtime_script):
    runtime = process(runtime_script("""
        import subprocess
        subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])
        time.sleep(10)
    """))
    started = time.monotonic()
    try:
        assert runtime.generate("wrapper", 32, 8, False, .1)["status"] == "timeout"
        assert time.monotonic() - started < 1.5
        assert not runtime.stdout_thread.is_alive() and not runtime.stderr_thread.is_alive()
    finally:
        runtime.close()


@pytest.mark.parametrize("section,key,value", [
    ("usage", "prompt_tokens", 16),
    ("usage", "completion_tokens", 0),
    ("usage", "completion_tokens", 9),
    ("usage", "completion_tokens", True),
    ("metrics", "prefill_tokens", 0),
    ("metrics", "prefill_ms", 0),
    ("metrics", "decode_ms", float("nan")),
    ("metrics", "ttft_ms", float("inf")),
    ("metrics", "ttft_ms", None),
    ("metrics", "prefill_tps", float("nan")),
    ("metrics", "mtp_used", "false"),
    ("metrics", "mtp_used", 1),
    ("metrics", "mtp_available", "unsupported"),
    ("metrics", "mtp_drafted_tokens", -1),
    ("metrics", "mtp_accepted_tokens", 1),
])
def test_cli_and_server_reject_the_same_invalid_metrics(runtime_script, section, key, value):
    result = native_result()
    result[section][key] = value
    # JSON also exercises the native wire representation of non-finite numbers.
    body = f"result = json.loads({json.dumps(result)!r})\nsend('event', result)\nsend('done')\n"
    runtime = process(runtime_script(body))
    try:
        assert runtime.generate("invalid", 32, 8, False, 1)["status"] == "error"
    finally:
        runtime.close()
    with pytest.raises(JobExecutionError):
        validate_benchmark_run(result, SimpleNamespace(prompt_tokens=32, output_tokens=8), "decode")


def test_requested_mtp_must_execute(runtime_script):
    runtime = process(runtime_script("send('event', result)\nsend('done')\n"))
    try:
        assert runtime.generate("mtp", 32, 8, True, 1)["status"] == "error"
    finally:
        runtime.close()


@pytest.mark.parametrize("available", [0, 1])
def test_mtp_execution_agrees_with_capability(runtime_script, available):
    runtime = process(runtime_script(f"""
        result['metrics'].update(mtp_available={available}, mtp_used=1, mtp_drafted_tokens=2, mtp_accepted_tokens=1)
        send('event', result)
        send('done')
    """))
    try:
        result = runtime.generate("mtp", 32, 8, True, 1)
        assert result["status"] == ("ok" if available else "error")
    finally:
        runtime.close()


def test_bad_greeting_reaps_child(tmp_path, monkeypatch):
    child = tmp_path / "bad_greeting.py"
    child.write_text("import time\nprint('{}', flush=True)\ntime.sleep(10)\n")
    processes = []
    popen = subprocess.Popen

    def capture(*args, **kwargs):
        processes.append(popen(*args, **kwargs))
        return processes[-1]

    monkeypatch.setattr(model_bench.subprocess, "Popen", capture)
    with pytest.raises(RuntimeError, match="did not greet"):
        process(child)
    assert processes[0].poll() is not None
    assert processes[0].stdout.closed and processes[0].stderr.closed


@pytest.mark.skipif(os.name != "posix", reason="POSIX startup interruption")
def test_startup_interrupt_reaps_child_and_stops_readers(tmp_path):
    marker = tmp_path / "runtime.pid"
    status = tmp_path / "cleanup.json"
    script = textwrap.dedent("""
        import json, os, sys, threading
        from pathlib import Path
        from bench import model_bench

        marker, status = map(Path, sys.argv[1:])
        processes = []
        popen = model_bench.subprocess.Popen
        read_message = model_bench.RuntimeProcess.read_message

        def capture(*args, **kwargs):
            processes.append(popen(*args, **kwargs))
            return processes[-1]

        def wait_for_greeting(self, timeout):
            marker.write_text(str(self.proc.pid))
            return read_message(self, timeout)

        model_bench.subprocess.Popen = capture
        model_bench.RuntimeProcess.read_message = wait_for_greeting
        try:
            model_bench.RuntimeProcess(
                [sys.executable, '-c', 'import time; time.sleep(30)'],
                os.environ, marker.parent)
        except KeyboardInterrupt:
            process = processes[0]
            status.write_text(json.dumps({
                'returncode': process.poll(),
                'stdout_closed': process.stdout.closed,
                'stderr_closed': process.stderr.closed,
                'readers_alive': any(thread is not threading.current_thread()
                                     for thread in threading.enumerate()),
            }))
        else:
            raise AssertionError('startup interruption was not propagated')
    """)
    parent = subprocess.Popen(
        [sys.executable, "-c", script, str(marker), str(status)],
        cwd=Path(model_bench.__file__).resolve().parents[1],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            if parent.poll() is not None:
                break
            time.sleep(.01)
        assert marker.exists(), parent.communicate(timeout=1)
        parent.send_signal(signal.SIGINT)
        output = parent.communicate(timeout=5)
        assert parent.returncode == 0, output
        cleanup = json.loads(status.read_text())
        assert cleanup == {"returncode": -signal.SIGKILL, "stdout_closed": True,
                           "stderr_closed": True, "readers_alive": False}
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.communicate()
        if marker.exists():
            with contextlib.suppress(ProcessLookupError):
                os.killpg(int(marker.read_text()), signal.SIGKILL)


def test_decode_without_mtp_capability_is_valid(runtime_script):
    runtime = process(runtime_script("""
        for key in ('mtp_used', 'mtp_drafted_tokens', 'mtp_accepted_tokens'):
            del result['metrics'][key]
        result['metrics']['mtp_available'] = 0
        send('event', result)
        send('done')
    """))
    try:
        assert runtime.generate("decode", 32, 8, False, 1)["status"] == "ok"
    finally:
        runtime.close()


@pytest.mark.parametrize("failure", ["send('done')", "time.sleep(2)"])
@pytest.mark.skipif(os.name != "posix", reason="fixture requires POSIX executable scripts")
def test_warmup_does_not_mask_failed_measurement(runtime_script, tmp_path, failure):
    path = runtime_script(f"""
        if request['id'].endswith('-0'):
            send('event', result)
            send('done')
        else:
            {failure}
    """)
    output = tmp_path / "results.json"
    completed = subprocess.run(
        [sys.executable, "-S", model_bench.__file__, "model.mfq", "--runtime", str(path),
         "--prefill-lengths", "32,64", "--decode-tokens", "8", "--repetitions", "1",
         "--timeout", ".1", "--output", str(output)],
        capture_output=True, text=True, timeout=5,
    )
    assert completed.returncode == 1, completed.stderr
    document = json.loads(output.read_text())
    assert document["summary"] == []
    assert len(document["results"][0]["runs"]) == 2
    assert document["results"][0]["error"]


@pytest.mark.parametrize("arguments", [
    ["--prefill-lengths", "0"], ["--prefill-lengths", "32,32"],
    ["--repetitions", "0"], ["--warmup-runs", "-1"],
    ["--decode-tokens", "1"], ["--timeout", "nan"], ["--timeout", "0"],
])
def test_invalid_cli_parameters_fail_before_loading(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["model_bench.py", "model.mfq", *arguments])
    with pytest.raises(SystemExit) as error:
        model_bench.main()
    assert error.value.code == 2
