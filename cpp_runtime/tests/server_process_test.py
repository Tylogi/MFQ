"""Exercise the native server as a real process; the product needs no Python."""

import contextlib
import http.client
import json
import os
from pathlib import Path
import runpy
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from uuid import UUID, uuid4

EXECUTABLE = Path(sys.argv.pop(1)).resolve()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mfq import __version__

# Load the existing lock contract without importing the Python web stack.
contract = runpy.run_path(str(Path(__file__).resolve().parents[2] / "mfq/server/instance.py"))
ServerInstanceLock = contract["ServerInstanceLock"]
server_identity = contract["server_identity"]


class Server:
    def __init__(self, process, host, port):
        self.process = process
        self.host = host
        self.port = port

    def request(self, path, method="GET", headers=None):
        connection = http.client.HTTPConnection(self.host, self.port, timeout=2)
        try:
            connection.request(method, path, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def identity(self):
        status, _, body = self.request("/health/identity")
        assert status == 200, body
        return json.loads(body)


@contextlib.contextmanager
def serve(data, *arguments):
    log_path = data.parent / f"server-{uuid4().hex}.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [str(EXECUTABLE), "--port", "0", "--data-dir", str(data), *map(str, arguments)],
            stdin=subprocess.PIPE, stdout=log, stderr=log,
            env={**os.environ, "PATH": ""},
        )
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                output = log_path.read_text(encoding="utf-8")
                if process.poll() is not None:
                    raise AssertionError(f"server exited during startup: {output}")
                first = output.splitlines()
                if first:
                    event = json.loads(first[0])
                    server = Server(process, event["host"], event["port"])
                    if server.request("/health")[0] == 200:
                        yield server
                        break
                time.sleep(0.01)
            else:
                raise AssertionError(f"server did not become ready: {output}")
        finally:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
                raise AssertionError("server did not stop")


class NativeServerTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="mfq-server-test-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.data = self.root / "data"

    def command(self, *arguments):
        return subprocess.run(
            [str(EXECUTABLE), "--data-dir", str(self.data), *map(str, arguments)],
            capture_output=True, text=True, timeout=5,
        )

    def test_cli_rejects_bad_configuration(self):
        not_a_directory = self.root / "file"
        not_a_directory.write_text("not a directory")
        for arguments in [
            ["--unknown"], ["--port"], ["--port", "-1"], ["--port", "+80"],
            ["--port", "65536"], ["--port", "80junk"], ["--port", "9" * 30],
            ["--host", "0.0.0.0"], ["--host", "localhost"], ["--host", "::"],
            ["--studio-instance-id", str(uuid4())],
            ["--studio-instance-id", "invalid", "--exit-on-stdin-close"],
            ["--web-root", str(self.root / "missing")],
            ["--data-dir", str(not_a_directory)],
        ]:
            with self.subTest(arguments=arguments):
                result = self.command(*arguments)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("MFQ Server not started:", result.stderr)
        self.assertFalse(self.data.exists())
        self.assertEqual(self.command("--version").stdout.strip(), __version__)
        help_result = self.command("--help")
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("Usage: mfq-server", help_result.stdout)

    def test_health_and_stable_identity_without_python(self):
        with serve(self.data) as server:
            status, headers, body = server.request("/health")
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body), {
                "status": "ok", "service": "mfq-server", "protocol_version": "1.0",
            })
            self.assertIn("application/json", headers["Content-Type"])
            self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
            identity = server.identity()
            self.assertEqual(identity, server.identity())
            expected = server_identity(self.data, instance_id=identity["instance_id"])
            expected["pid"] = server.process.pid
            self.assertEqual(identity, expected)
            self.assertEqual(UUID(identity["instance_id"]).version, 4)
            self.assertNotIn(str(self.data), json.dumps(identity))
            status, _, body = server.request("/api/v1/models", headers={"X-MFQ-Instance-ID": str(uuid4())})
            self.assertEqual(status, 409)
            self.assertEqual(json.loads(body)["error"]["code"], "server_instance_changed")
            self.assertEqual(server.request("/health", headers={"X-MFQ-Instance-ID": identity["instance_id"]})[0], 200)
            self.assertEqual(server.request("/api/v1/models")[0], 404)
        self.assertFalse(self.data.exists())

    def test_studio_identity_and_stdin_eof_release_lock(self):
        supplied = str(uuid4()).upper()
        with serve(self.data, "--exit-on-stdin-close", "--studio-instance-id", supplied) as server:
            identity = server.identity()
            self.assertEqual(identity["instance_id"], supplied.lower())
            self.assertEqual(identity["launch_mode"], "studio")
            server.process.stdin.write(b"parent is still alive\n")
            server.process.stdin.flush()
            self.assertEqual(server.request("/health")[0], 200)
            server.process.stdin.close()
            self.assertEqual(server.process.wait(timeout=5), 0)
        with serve(self.data) as replacement:
            self.assertEqual(replacement.identity()["workspace_id"], identity["workspace_id"])
            self.assertNotEqual(replacement.identity()["instance_id"], identity["instance_id"])

    def test_unicode_workspace_identity(self):
        data = self.root / "数据 space"
        with serve(data) as server:
            self.assertEqual(server.identity()["workspace_id"], server_identity(data)["workspace_id"])

    def test_ipv6_loopback(self):
        try:
            with socket.socket(socket.AF_INET6) as probe:
                probe.bind(("::1", 0))
        except OSError:
            self.skipTest("IPv6 loopback is unavailable")
        with serve(self.data, "--host", "::1") as server:
            self.assertEqual(server.request("/health")[0], 200)

    def test_same_directory_lock_is_shared_with_python(self):
        with ServerInstanceLock(self.data) as instance:
            result = self.command("--port", "0")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already managed", result.stderr)
            self.assertEqual(json.loads(instance.path.read_bytes()[1:]), instance.identity)
        with serve(self.data) as server:
            with self.assertRaisesRegex(RuntimeError, "already managed"), ServerInstanceLock(self.data):
                self.fail("Python must not bypass the native instance lock")
            result = self.command("--port", "0")
            self.assertIn("already managed", result.stderr)
            self.assertEqual(server.identity()["pid"], server.process.pid)
        with ServerInstanceLock(self.data):
            pass

    @unittest.skipIf(os.name == "nt", "symlinks require additional privileges on Windows")
    def test_aliases_and_deleted_data_directory_cannot_bypass_lock(self):
        self.data.mkdir()
        alias = self.root / "alias"
        alias.symlink_to(self.data, target_is_directory=True)
        with serve(self.data):
            result = self.command("--data-dir", alias, "--port", "0")
            self.assertIn("already managed", result.stderr)
            shutil.rmtree(self.data)
            self.data.mkdir()
            self.assertIn("already managed", self.command("--port", "0").stderr)

    def test_port_conflict_releases_failed_instances_lock(self):
        other_data = self.root / "other"
        with serve(other_data) as server:
            result = self.command("--port", server.port)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("cannot bind listener", result.stderr)
            with ServerInstanceLock(self.data):
                pass
            with serve(self.data):
                pass

    @unittest.skipIf(os.name == "nt", "Windows terminate does not deliver POSIX signals")
    def test_signals_and_process_death_release_lock(self):
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGKILL):
            with self.subTest(signal=sig), serve(self.data) as server:
                server.process.send_signal(sig)
                result = server.process.wait(timeout=5)
                self.assertEqual(result, -signal.SIGKILL if sig == signal.SIGKILL else 0)
                with ServerInstanceLock(self.data) as instance:
                    self.assertTrue(instance.path.is_file())

    def test_existing_databases_are_untouched(self):
        self.data.mkdir()
        databases = [self.data / "server.sqlite3", self.data / "mfq-server.sqlite3"]
        for database in databases:
            database.write_bytes(b"existing database must not be opened or replaced")
        with serve(self.data):
            for database in databases:
                self.assertEqual(database.read_bytes(), b"existing database must not be opened or replaced")

    def test_static_files_navigation_and_api_errors(self):
        web = self.root / "web"
        (web / "assets").mkdir(parents=True)
        (web / "guide").mkdir()
        (web / "api").mkdir()
        index = b"<!doctype html><title>MFQ Studio</title>"
        (web / "index.html").write_bytes(index)
        (web / "assets" / "app.js").write_text("console.log('ok');")
        (web / "guide" / "index.html").write_text("guide")
        (web / "api" / "secret").write_text("not an API")
        (web / "health").write_text("not health")
        with serve(self.data, "--web-root", str(web) + "/") as server:
            for path in ("/", "/chat", "/settings/profile", "/future/page?session=1"):
                with self.subTest(path=path):
                    status, headers, body = server.request(path, headers={"Accept": "text/html"})
                    self.assertEqual((status, body), (200, index))
                    self.assertIn("text/html", headers["Content-Type"])
                    status, headers, body = server.request(path, method="HEAD", headers={"Accept": "text/html"})
                    self.assertEqual((status, body), (200, b""))
                    self.assertEqual(int(headers["Content-Length"]), len(index))
            self.assertEqual(server.request("/assets/app.js")[2], b"console.log('ok');")
            self.assertEqual(server.request("/guide/")[2], b"guide")
            self.assertEqual(json.loads(server.request("/health")[2])["service"], "mfq-server")
            for path in ("/api/secret", "/v1/missing", "/health/missing", "/docs/missing",
                         "/assets/missing", "/missing.js/", "/%2e%2e/secret.txt", "/%5csecret", "/%00secret"):
                with self.subTest(path=path):
                    status, _, body = server.request(path, headers={"Accept": "text/html"})
                    self.assertEqual(status, 404)
                    self.assertNotEqual(body, index)
            for headers in ({"Accept": "*/*"}, {"Accept": "application/json"},
                            {"Accept": "text/html", "Sec-Fetch-Dest": "script"}):
                self.assertEqual(server.request("/chat", headers=headers)[0], 404)
            self.assertNotEqual(server.request("/chat", method="POST", headers={"Accept": "text/html"})[0], 200)

    @unittest.skipIf(os.name == "nt", "symlinks require additional privileges on Windows")
    def test_static_symlinks_cannot_escape_web_root(self):
        web = self.root / "web"
        web.mkdir()
        (web / "index.html").write_text("index")
        secret = self.root / "secret.txt"
        secret.write_text("private content")
        (web / "outside.txt").symlink_to(secret)
        with serve(self.data, "--web-root", web) as server:
            status, _, body = server.request("/outside.txt")
            self.assertEqual(status, 403)
            self.assertNotIn(b"private content", body)
            (web / "index.html").unlink()
            (web / "index.html").symlink_to(secret)
            self.assertEqual(server.request("/chat", headers={"Accept": "text/html"})[0], 403)
        result = self.command("--web-root", web)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--web-root", result.stderr)


if __name__ == "__main__":
    unittest.main()
