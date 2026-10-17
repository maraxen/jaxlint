"""``jaxlint.hooks.edit_hook`` runs after every Write/Edit call: it must import only the stdlib."""

from __future__ import annotations

import subprocess
import sys

_PROBE = (
    "import sys, jaxlint.hooks.edit_hook; "
    "heavy = sorted(m for m in sys.modules if m.split('.')[0] in ('rich', 'jax', 'numpy') "
    "or m == 'jaxlint.core' or m.startswith('jaxlint.core.')); "
    "assert not heavy, heavy"
)


def test_edit_hook_import_loads_no_rich_jax_numpy_or_core() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
