"""``jaxlint.core.runner`` must import without pulling in ``rich``.

The edit-hook path imports the runner on every file edit; rich is only needed
when rendering the ``rich`` output format, so it is imported lazily there.
"""

from __future__ import annotations

import subprocess
import sys

_PROBE = (
    "import sys, jaxlint.core.runner; "
    "assert not any(m == 'rich' or m.startswith('rich.') for m in sys.modules), "
    "sorted(m for m in sys.modules if m.startswith('rich'))"
)


def test_core_runner_import_does_not_load_rich() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
