"""Diagnostics-only Language Server (stdio, ``jaxlint[lsp]``)."""

from __future__ import annotations

import asyncio
import logging
import os
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


def canonical_file_uri(uri: str) -> str:
    """Return a stable dict key for a ``file://`` URI (resolved path, normalized URI)."""
    return uri_to_path(uri).resolve().as_uri()


def _lsp_debounce_ms() -> int:
    raw = os.environ.get("JAXLINT_LSP_DEBOUNCE_MS", "200")
    try:
        ms = int(str(raw).strip())
    except ValueError:
        return 200
    return ms if ms >= 0 else 0


def build_server() -> LanguageServer:
    """Construct the jaxlint language server (full document sync, diagnostics only)."""

    server = LanguageServer(
        "jaxlint",
        jaxlint_version,
        text_document_sync_kind=types.TextDocumentSyncKind.Full,
    )

    debounce_ms = _lsp_debounce_ms()
    generation: dict[str, int] = {}
    pending_tasks: dict[str, asyncio.Task[None]] = {}
    client_uri: dict[str, str] = {}
    closed: set[str] = set()

    def publish_empty(uri: str, version: int | None) -> None:
        server.text_document_publish_diagnostics(
            types.PublishDiagnosticsParams(uri=uri, version=version, diagnostics=[])
        )

    async def lint_and_publish(uri_client: str, canon: str, *, expected_gen: int | None) -> None:
        def work():
            try:
                doc = server.workspace.get_text_document(uri_client)
            except Exception:  # pragma: no cover - defensive
                return None, None
            try:
                path = uri_to_path(uri_client)
            except ValueError:
                return None, doc.version
            if path.suffix != ".py":
                return [], doc.version
            cfg = load_config(path.parent)
            jax_diags = diagnostics_for_python_source(
                path,
                doc.source,
                cfg=cfg,
                perf=True,
                doc=True,
                mathjx=True,
            )
            return jax_diags, doc.version

        try:
            result = await asyncio.to_thread(work)
        except Exception as exc:  # pragma: no cover - defensive
            _LOG.exception("lint failed for %s: %s", uri_client, exc)
            publish_empty(uri_client, None)
            return

        jax_diags, version = result
        if jax_diags is None:
            publish_empty(uri_client, version)
            return

        if (
            (expected_gen is not None and generation.get(canon, 0) != expected_gen)
            or canon in closed
        ):
            publish_empty(uri_client, version)
            return

        lsp_diags = jaxlint_diagnostics_to_lsp(jax_diags)
        server.text_document_publish_diagnostics(
            types.PublishDiagnosticsParams(
                uri=uri_client,
                version=version,
                diagnostics=lsp_diags,
            )
        )

    def bump_generation(canon: str) -> None:
        generation[canon] = generation.get(canon, 0) + 1

    @server.feature(types.TEXT_DOCUMENT_DID_OPEN)
    async def did_open(ls: LanguageServer, params: types.DidOpenTextDocumentParams) -> None:
        _ = ls
        uri = params.text_document.uri
        try:
            canon = canonical_file_uri(uri)
        except ValueError:
            publish_empty(uri, params.text_document.version)
            return

        closed.discard(canon)
        client_uri[canon] = uri
        # Open must publish even if didChange notifications bump generation while we await I/O.
        await lint_and_publish(uri, canon, expected_gen=None)

    @server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
    async def did_change(ls: LanguageServer, params: types.DidChangeTextDocumentParams) -> None:
        _ = ls
        uri = params.text_document.uri
        try:
            canon = canonical_file_uri(uri)
        except ValueError:
            publish_empty(uri, params.text_document.version)
            return

        client_uri[canon] = uri

        if debounce_ms == 0:
            expected_gen = generation.get(canon, 0)
            await lint_and_publish(uri, canon, expected_gen=expected_gen)
            return

        prior = pending_tasks.pop(canon, None)
        if prior is not None and not prior.done():
            prior.cancel()

        bump_generation(canon)
        my_gen = generation[canon]

        async def debounced_lint() -> None:
            current = asyncio.current_task()
            try:
                try:
                    await asyncio.sleep(debounce_ms / 1000.0)
                except asyncio.CancelledError:
                    raise
                uri_live = client_uri.get(canon, uri)
                await lint_and_publish(uri_live, canon, expected_gen=my_gen)
            finally:
                if current is not None and pending_tasks.get(canon) is current:
                    pending_tasks.pop(canon, None)

        pending_tasks[canon] = asyncio.create_task(debounced_lint())

    @server.feature(types.TEXT_DOCUMENT_DID_CLOSE)
    def did_close(ls: LanguageServer, params: types.DidCloseTextDocumentParams) -> None:
        _ = ls
        uri = params.text_document.uri
        try:
            canon = canonical_file_uri(uri)
        except ValueError:
            publish_empty(uri, None)
            return

        bump_generation(canon)
        task = pending_tasks.pop(canon, None)
        if task is not None and not task.done():
            task.cancel()

        closed.add(canon)
        publish_empty(uri, None)

    return server
