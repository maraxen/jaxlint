"""Black-box LSP stdio tests (``jaxlint[lsp]``)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pygls")


def _write_message(stream, obj: dict[str, Any]) -> None:
    body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    stream.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    stream.write(body)
    stream.flush()


def _read_message(stream) -> dict[str, Any] | None:
    headers = b""
    while True:
        line = stream.readline()
        if not line:
            return None
        headers += line
        if headers.endswith(b"\r\n\r\n"):
            break
    length: int | None = None
    for h in headers.decode("ascii", errors="replace").strip().split("\r\n"):
        if h.lower().startswith("content-length:"):
            length = int(h.split(":", 1)[1].strip())
            break
    if length is None:
        return None
    body = stream.read(length)
    if len(body) != length:
        return None
    return json.loads(body.decode("utf-8"))


@pytest.mark.integration
@pytest.mark.lsp
def test_lsp_publish_diagnostics_on_did_open(fixtures_dir: Path, tmp_path: Path) -> None:
    src = (fixtures_dir / "jd001_missing_docstring.py").read_text(encoding="utf-8")
    doc_path = tmp_path / "jd001_missing_docstring.py"
    doc_path.write_text(src, encoding="utf-8")
    uri = doc_path.resolve().as_uri()

    proc = subprocess.Popen(
        [sys.executable, "-m", "jaxlint.lsp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=tmp_path,
        text=False,
    )
    assert proc.stdin is not None and proc.stdout is not None

    try:
        _write_message(
            proc.stdin,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "processId": None,
                    "rootUri": None,
                    "capabilities": {},
                    "clientInfo": {"name": "pytest-jaxlint", "version": "0"},
                },
            },
        )
        while True:
            msg = _read_message(proc.stdout)
            assert msg is not None
            if msg.get("id") == 1:
                break

        _write_message(proc.stdin, {"jsonrpc": "2.0", "method": "initialized", "params": {}})

        _write_message(
            proc.stdin,
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": uri,
                        "languageId": "python",
                        "version": 1,
                        "text": src,
                    }
                },
            },
        )

        found: list[str] = []
        for _ in range(50):
            msg = _read_message(proc.stdout)
            assert msg is not None
            if msg.get("method") == "textDocument/publishDiagnostics":
                params = msg.get("params") or {}
                for d in params.get("diagnostics") or []:
                    code = d.get("code")
                    if code is not None:
                        found.append(str(code))
                break
        assert "JD001" in found
    finally:
        proc.stdin.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.mark.integration
@pytest.mark.lsp
def test_lsp_did_change_updates_diagnostics(fixtures_dir: Path, tmp_path: Path) -> None:
    bad = (fixtures_dir / "jd001_missing_docstring.py").read_text(encoding="utf-8")
    good = (fixtures_dir / "clean_minimal.py").read_text(encoding="utf-8")
    doc_path = tmp_path / "flip.py"
    doc_path.write_text(bad, encoding="utf-8")
    uri = doc_path.resolve().as_uri()

    proc = subprocess.Popen(
        [sys.executable, "-m", "jaxlint.lsp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=tmp_path,
        text=False,
    )
    assert proc.stdin is not None and proc.stdout is not None

    try:
        _write_message(
            proc.stdin,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "processId": None,
                    "rootUri": None,
                    "capabilities": {},
                    "clientInfo": {"name": "pytest-jaxlint", "version": "0"},
                },
            },
        )
        while True:
            msg = _read_message(proc.stdout)
            assert msg is not None
            if msg.get("id") == 1:
                break

        _write_message(proc.stdin, {"jsonrpc": "2.0", "method": "initialized", "params": {}})

        _write_message(
            proc.stdin,
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": uri,
                        "languageId": "python",
                        "version": 1,
                        "text": bad,
                    }
                },
            },
        )

        def next_publish() -> list[dict[str, Any]]:
            for _ in range(50):
                msg = _read_message(proc.stdout)
                assert msg is not None
                if msg.get("method") == "textDocument/publishDiagnostics":
                    return list((msg.get("params") or {}).get("diagnostics") or [])
            raise AssertionError("no publishDiagnostics")

        first = next_publish()
        assert any(str(d.get("code")) == "JD001" for d in first)

        _write_message(
            proc.stdin,
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didChange",
                "params": {
                    "textDocument": {"uri": uri, "version": 2},
                    "contentChanges": [{"text": good}],
                },
            },
        )

        second = next_publish()
        assert not any(str(d.get("code")) == "JD001" for d in second)
    finally:
        proc.stdin.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.mark.integration
@pytest.mark.lsp
def test_lsp_did_close_clears_diagnostics(fixtures_dir: Path, tmp_path: Path) -> None:
    src = (fixtures_dir / "jd001_missing_docstring.py").read_text(encoding="utf-8")
    doc_path = tmp_path / "jd001_missing_docstring.py"
    doc_path.write_text(src, encoding="utf-8")
    uri = doc_path.resolve().as_uri()

    proc = subprocess.Popen(
        [sys.executable, "-m", "jaxlint.lsp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=tmp_path,
        text=False,
    )
    assert proc.stdin is not None and proc.stdout is not None

    try:
        _write_message(
            proc.stdin,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "processId": None,
                    "rootUri": None,
                    "capabilities": {},
                    "clientInfo": {"name": "pytest-jaxlint", "version": "0"},
                },
            },
        )
        while True:
            msg = _read_message(proc.stdout)
            assert msg is not None
            if msg.get("id") == 1:
                break

        _write_message(proc.stdin, {"jsonrpc": "2.0", "method": "initialized", "params": {}})

        _write_message(
            proc.stdin,
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": uri,
                        "languageId": "python",
                        "version": 1,
                        "text": src,
                    }
                },
            },
        )

        def next_publish() -> tuple[str, list[dict[str, Any]]]:
            for _ in range(50):
                msg = _read_message(proc.stdout)
                assert msg is not None
                if msg.get("method") == "textDocument/publishDiagnostics":
                    params = msg.get("params") or {}
                    return str(params.get("uri", "")), list(params.get("diagnostics") or [])
            raise AssertionError("no publishDiagnostics")

        first_uri, first_diags = next_publish()
        assert first_uri == uri
        assert any(str(d.get("code")) == "JD001" for d in first_diags)

        _write_message(
            proc.stdin,
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didClose",
                "params": {"textDocument": {"uri": uri}},
            },
        )

        close_uri, close_diags = next_publish()
        assert close_uri == uri
        assert close_diags == []
    finally:
        proc.stdin.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
