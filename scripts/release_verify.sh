#!/usr/bin/env bash
# Optional local smoke: build → twine check → editable-ish wheel install → CLI entrypoints.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

uv sync --all-groups
uv run python -m build
uv run twine check dist/*

rm -rf .release-venv
uv venv .release-venv
# shellcheck disable=SC1091
source .release-venv/bin/activate

uv pip install "jaxlint[lsp,mcp]" --find-links dist --no-index --upgrade

jaxlint version
jaxlint-lsp --help
jaxlint-mcp --help
