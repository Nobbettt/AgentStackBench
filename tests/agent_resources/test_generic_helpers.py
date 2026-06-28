# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _fake_python_bin(tmp_path: Path) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    call_log = tmp_path / "python-calls.log"
    python_path = bin_dir / "python"
    python_path.write_text(
        "#!/usr/bin/env sh\n"
        "printf '%s\\n' \"$*\" >> \"$PYTHON_CALL_LOG\"\n",
        encoding="utf-8",
    )
    python_path.chmod(0o755)
    return bin_dir, call_log


def test_python_source_bootstrap_noops_without_python_metadata(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    bin_dir, call_log = _fake_python_bin(tmp_path)

    result = subprocess.run(
        ["sh", str(REPO_ROOT / "agent-resources/python-source-bootstrap.sh")],
        cwd=repo,
        env={
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "PYTHON_CALL_LOG": str(call_log),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "no Python packaging metadata found; skipping" in result.stdout
    assert not call_log.exists()


def test_python_source_bootstrap_builds_setup_py_then_editable_installs(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "setup.py").write_text("from setuptools import setup\nsetup(name='demo')\n", encoding="utf-8")
    bin_dir, call_log = _fake_python_bin(tmp_path)

    result = subprocess.run(
        ["sh", str(REPO_ROOT / "agent-resources/python-source-bootstrap.sh")],
        cwd=repo,
        env={
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "PYTHON_CALL_LOG": str(call_log),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert call_log.read_text(encoding="utf-8").splitlines() == [
        "setup.py build_ext --inplace",
        "-m pip install --no-index --no-build-isolation --no-deps -e .",
    ]


def test_python_source_bootstrap_editable_installs_pyproject_without_setup_py(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(
        "[build-system]\nrequires = ['setuptools']\nbuild-backend = 'setuptools.build_meta'\n",
        encoding="utf-8",
    )
    bin_dir, call_log = _fake_python_bin(tmp_path)

    result = subprocess.run(
        ["sh", str(REPO_ROOT / "agent-resources/python-source-bootstrap.sh")],
        cwd=repo,
        env={
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "PYTHON_CALL_LOG": str(call_log),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert call_log.read_text(encoding="utf-8").splitlines() == [
        "-m pip install --no-index --no-build-isolation --no-deps -e .",
    ]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_mcp_tool_can_list_and_call_configured_stdio_mcp_server(tmp_path: Path) -> None:
    server = tmp_path / "fake-mcp-server.mjs"
    server.write_text(
        """
process.stdin.setEncoding("utf8");
let buffer = "";

function write(message) {
  process.stdout.write(`${JSON.stringify(message)}\\n`);
}

process.stdin.on("data", (chunk) => {
  buffer += chunk;
  let newline = buffer.indexOf("\\n");
  while (newline !== -1) {
    const line = buffer.slice(0, newline).trim();
    buffer = buffer.slice(newline + 1);
    if (line) {
      const message = JSON.parse(line);
      if (message.method === "initialize") {
        write({ jsonrpc: "2.0", id: message.id, result: { protocolVersion: "2025-03-26", capabilities: {} } });
      } else if (message.method === "tools/list") {
        write({ jsonrpc: "2.0", id: message.id, result: { tools: [{ name: "echo", inputSchema: { type: "object" } }] } });
      } else if (message.method === "tools/call") {
        write({ jsonrpc: "2.0", id: message.id, result: { content: [{ type: "text", text: message.params.arguments.text }] } });
      }
    }
    newline = buffer.indexOf("\\n");
  }
});
""",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "CONTEXTBENCH_MCP_COMMAND": "node",
        "CONTEXTBENCH_MCP_ARGS_JSON": json.dumps([str(server)]),
        "CONTEXTBENCH_MCP_TIMEOUT_MS": "5000",
    }

    list_result = subprocess.run(
        ["node", str(REPO_ROOT / "agent-resources/mcp-tool.mjs"), "--list"],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )
    assert json.loads(list_result.stdout)["tools"][0]["name"] == "echo"

    call_result = subprocess.run(
        ["node", str(REPO_ROOT / "agent-resources/mcp-tool.mjs"), "echo", '{"text":"hello"}'],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )
    assert json.loads(call_result.stdout)["content"][0]["text"] == "hello"


def test_http_json_rewrite_proxy_rewrites_matching_json_fields_from_env() -> None:
    seen_payloads: list[dict[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            body = self.rfile.read(int(self.headers.get("content-length", "0")))
            payload = json.loads(body.decode("utf-8"))
            seen_payloads.append(payload)
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            return

    upstream_port = _free_port()
    proxy_port = _free_port()
    upstream = ThreadingHTTPServer(("127.0.0.1", upstream_port), Handler)
    process: subprocess.Popen[str] | None = None
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        env = {
            **os.environ,
            "CONTEXTBENCH_PROXY_PORT": str(proxy_port),
            "CONTEXTBENCH_PROXY_UPSTREAM_ORIGIN": f"http://127.0.0.1:{upstream_port}",
            "CONTEXTBENCH_PROXY_REWRITE_RULES_JSON": json.dumps(
                [
                    {
                        "path": "/api/device/auth",
                        "methods": ["POST"],
                        "set": {"deviceId": {"env": "TEST_DEVICE_ID"}},
                    }
                ]
            ),
            "TEST_DEVICE_ID": "host-device",
        }
        process = subprocess.Popen(
            ["node", str(REPO_ROOT / "agent-resources/http-json-rewrite-proxy.mjs")],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        deadline = time.monotonic() + 5
        while True:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{proxy_port}/health", timeout=0.2) as response:
                    assert response.status == 200
                    break
            except urllib.error.URLError:
                if time.monotonic() > deadline:
                    raise AssertionError("proxy did not become ready")
                time.sleep(0.05)

        request = urllib.request.Request(
            f"http://127.0.0.1:{proxy_port}/api/device/auth",
            data=json.dumps({"deviceId": "container-device", "other": 1}).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )

        # Handle one proxied request in the Python test process so assertions can inspect it.
        with urllib.request.urlopen(request, timeout=5) as response:
            response_payload = json.loads(response.read().decode("utf-8"))

        assert response_payload == {"deviceId": "host-device", "other": 1}
        assert seen_payloads == [response_payload]
    finally:
        upstream.shutdown()
        upstream.server_close()
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
