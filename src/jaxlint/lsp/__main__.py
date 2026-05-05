"""``python -m jaxlint.lsp`` — stdio language server."""

from __future__ import annotations


def main() -> None:
    try:
        from jaxlint.lsp.server import build_server
    except ImportError as exc:
        missing = "pygls" if "pygls" in str(exc).lower() or "lsprotocol" in str(exc).lower() else str(exc)
        raise SystemExit(
            f"jaxlint LSP requires optional dependencies ({missing}). "
            "Install with: uv sync --extra lsp   or   pip install 'jaxlint[lsp]'"
        ) from exc

    server = build_server()
    server.start_io()


if __name__ == "__main__":
    main()
