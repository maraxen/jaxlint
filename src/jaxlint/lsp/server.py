"""Diagnostics-only Language Server (stdio, ``jaxlint[lsp]``)."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from urllib.parse import unquote, urlparse

from jaxlint import __version__ as jaxlint_version
from jaxlint.core.config import load_config
from jaxlint.core.runner import diagnostics_for_python_source
from jaxlint.lsp.convert import jaxlint_diagnostics_to_lsp

from lsprotocol import types
from pygls.lsp.server import LanguageServer

_LOG = logging.getLogger(__name__)


def uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        msg = f"unsupported URI scheme {parsed.scheme!r} (only file:// is supported)"
        raise ValueError(msg)
    path = unquote(parsed.path)
    if len(path) >= 3 and path.startswith("/") and path[2] == ":":
        path = path[1:]
    return Path(path)


def build_server() -> LanguageServer:
    """Construct the jaxlint language server (full document sync, diagnostics only)."""

    server = LanguageServer(
        "jaxlint",
        jaxlint_version,
        text_document_sync_kind=types.TextDocumentSyncKind.Full,
    )

    def publish_empty(uri: str, version: int | None) -> None:
        server.text_document_publish_diagnostics(
            types.PublishDiagnosticsParams(uri=uri, version=version, diagnostics=[])
        )

    async def lint_and_publish(uri: str, version: int | None) -> None:
        try:
            path = uri_to_path(uri)
        except ValueError:
            publish_empty(uri, version)
            return
        if path.suffix != ".py":
            publish_empty(uri, version)
            return
        try:
            doc = server.workspace.get_text_document(uri)
        except Exception:  # pragma: no cover - defensive
            publish_empty(uri, version)
            return

        cfg = load_config(path.parent)

        def work():
            return diagnostics_for_python_source(
                path,
                doc.source,
                cfg=cfg,
                perf=True,
                doc=True,
                mathjx=True,
            )

        try:
            jax_diags = await asyncio.to_thread(work)
        except Exception as exc:  # pragma: no cover - defensive
            _LOG.exception("lint failed for %s: %s", uri, exc)
            publish_empty(uri, version)
            return

        lsp_diags = jaxlint_diagnostics_to_lsp(jax_diags)
        server.text_document_publish_diagnostics(
            types.PublishDiagnosticsParams(
                uri=uri,
                version=version,
                diagnostics=lsp_diags,
            )
        )

    @server.feature(types.TEXT_DOCUMENT_DID_OPEN)
    async def did_open(ls: LanguageServer, params: types.DidOpenTextDocumentParams) -> None:
        _ = ls
        await lint_and_publish(
            params.text_document.uri,
            params.text_document.version,
        )

    @server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
    async def did_change(ls: LanguageServer, params: types.DidChangeTextDocumentParams) -> None:
        _ = ls
        doc = server.workspace.get_text_document(params.text_document.uri)
        await lint_and_publish(params.text_document.uri, doc.version)

    @server.feature(types.TEXT_DOCUMENT_DID_CLOSE)
    def did_close(ls: LanguageServer, params: types.DidCloseTextDocumentParams) -> None:
        _ = ls
        publish_empty(params.text_document.uri, None)

    return server
