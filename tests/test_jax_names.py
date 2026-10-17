"""Tests for the shared JAX-name resolver used by the JL003/JL004/JL005 sensors."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from jaxlint.core.config import LintConfig
from jaxlint.core.runner import run_checks
from jaxlint.core.sensors._jax_names import (
    alias_map,
    is_kernel_decorator,
    jitted_name_table,
    resolve_dotted,
)


def _decorators(src: str) -> list[ast.expr]:
    """Return the decorator expressions of the first function in ``src``."""
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    return list(fn.decorator_list)


def _kernel_flags(src: str) -> list[bool]:
    """Evaluate ``is_kernel_decorator`` on every decorator of the first function in ``src``."""
    tree = ast.parse(src)
    aliases = alias_map(tree)
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.decorator_list
    )
    return [is_kernel_decorator(d, aliases) for d in fn.decorator_list]


# ---------------------------------------------------------------------------
# alias_map
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_alias_map_plain_import() -> None:
    aliases = alias_map(ast.parse("import jax\nimport functools\n"))
    assert aliases == {"jax": "jax", "functools": "functools"}


@pytest.mark.unit
def test_alias_map_import_as() -> None:
    aliases = alias_map(ast.parse("import jax as j\nimport equinox as eqx\n"))
    assert aliases["j"] == "jax"
    assert aliases["eqx"] == "equinox"


@pytest.mark.unit
def test_alias_map_dotted_import_binds_top_name() -> None:
    aliases = alias_map(ast.parse("import jax.numpy\nimport jax.numpy as jnp\n"))
    assert aliases["jax"] == "jax"
    assert aliases["jnp"] == "jax.numpy"


@pytest.mark.unit
def test_alias_map_from_import_and_alias() -> None:
    src = (
        "from jax import jit\nfrom equinox import filter_jit as fj\nfrom functools import partial\n"
    )
    aliases = alias_map(ast.parse(src))
    assert aliases["jit"] == "jax.jit"
    assert aliases["fj"] == "equinox.filter_jit"
    assert aliases["partial"] == "functools.partial"


@pytest.mark.unit
def test_alias_map_ignores_relative_and_star_imports() -> None:
    aliases = alias_map(ast.parse("from . import jit\nfrom jax import *\n"))
    assert "jit" not in aliases


# ---------------------------------------------------------------------------
# resolve_dotted
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resolve_dotted_expands_module_alias() -> None:
    aliases = {"eqx": "equinox"}
    expr = ast.parse("eqx.filter_jit", mode="eval").body
    assert resolve_dotted(expr, aliases) == "equinox.filter_jit"


@pytest.mark.unit
def test_resolve_dotted_expands_from_import_name() -> None:
    aliases = {"jit": "jax.jit"}
    expr = ast.parse("jit", mode="eval").body
    assert resolve_dotted(expr, aliases) == "jax.jit"


@pytest.mark.unit
def test_resolve_dotted_unbound_name_passes_through() -> None:
    expr = ast.parse("jax.jit", mode="eval").body
    assert resolve_dotted(expr, {}) == "jax.jit"


@pytest.mark.unit
def test_resolve_dotted_non_name_expression_is_none() -> None:
    expr = ast.parse("f()[0]", mode="eval").body
    assert resolve_dotted(expr, {}) is None


# ---------------------------------------------------------------------------
# is_kernel_decorator: the forms that were already detected
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "src",
    [
        "import jax\n@jax.jit\ndef f(x): ...\n",
        "import jax\n@jax.grad\ndef f(x): ...\n",
        "import jax\n@jax.jit(static_argnames=('n',))\ndef f(x, n): ...\n",
        "import jax\n@jax.grad()\ndef f(x): ...\n",
    ],
)
def test_existing_forms_still_detected(src: str) -> None:
    assert _kernel_flags(src) == [True]


# ---------------------------------------------------------------------------
# is_kernel_decorator: newly covered forms
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_bare_jit_after_from_import() -> None:
    assert _kernel_flags("from jax import jit\n@jit\ndef f(x): ...\n") == [True]


@pytest.mark.unit
def test_bare_jit_with_call_form() -> None:
    assert _kernel_flags("from jax import jit\n@jit(static_argnums=0)\ndef f(x): ...\n") == [True]


@pytest.mark.unit
def test_eqx_filter_jit_attribute_form() -> None:
    assert _kernel_flags("import equinox as eqx\n@eqx.filter_jit\ndef f(x): ...\n") == [True]


@pytest.mark.unit
def test_eqx_filter_jit_full_module_name() -> None:
    assert _kernel_flags("import equinox\n@equinox.filter_jit\ndef f(x): ...\n") == [True]


@pytest.mark.unit
def test_eqx_filter_jit_call_form() -> None:
    assert _kernel_flags(
        "import equinox as eqx\n@eqx.filter_jit(donate='none')\ndef f(x): ...\n"
    ) == [True]


@pytest.mark.unit
def test_bare_filter_jit_after_from_import() -> None:
    assert _kernel_flags("from equinox import filter_jit\n@filter_jit\ndef f(x): ...\n") == [True]


@pytest.mark.unit
def test_partial_jax_jit() -> None:
    src = "import functools\nimport jax\n@functools.partial(jax.jit, static_argnums=0)\ndef f(x): ...\n"
    assert _kernel_flags(src) == [True]


@pytest.mark.unit
def test_bare_partial_jax_jit() -> None:
    src = "from functools import partial\nimport jax\n@partial(jax.jit, static_argnames=('n',))\ndef f(x, n): ...\n"
    assert _kernel_flags(src) == [True]


@pytest.mark.unit
def test_partial_filter_jit() -> None:
    src = "from functools import partial\nimport equinox as eqx\n@partial(eqx.filter_jit, donate='none')\ndef f(x): ...\n"
    assert _kernel_flags(src) == [True]


@pytest.mark.unit
def test_import_alias_of_jax() -> None:
    assert _kernel_flags("import jax as j\n@j.jit\ndef f(x): ...\n") == [True]


@pytest.mark.unit
def test_from_import_alias_of_jit() -> None:
    assert _kernel_flags("from jax import jit as J\n@J\ndef f(x): ...\n") == [True]


@pytest.mark.unit
def test_pmap_and_value_and_grad_families() -> None:
    src = (
        "import jax\nimport equinox as eqx\n"
        "@jax.pmap\ndef a(x): ...\n"
        "@jax.value_and_grad\ndef b(x): ...\n"
        "@eqx.filter_grad\ndef c(x): ...\n"
        "@eqx.filter_value_and_grad\ndef d(x): ...\n"
        "@eqx.filter_pmap\ndef e(x): ...\n"
    )
    tree = ast.parse(src)
    aliases = alias_map(tree)
    flags = [
        is_kernel_decorator(fn.decorator_list[0], aliases)
        for fn in tree.body
        if isinstance(fn, ast.FunctionDef)
    ]
    assert flags == [True, True, True, True, True]


# ---------------------------------------------------------------------------
# is_kernel_decorator: negatives
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_user_function_named_jit_not_from_jax_is_not_kernel() -> None:
    src = "def jit(f):\n    return f\n\n@jit\ndef g(x): ...\n"
    assert _kernel_flags(src) == [False]


@pytest.mark.unit
def test_jit_imported_from_other_module_is_not_kernel() -> None:
    assert _kernel_flags("from numba import jit\n@jit\ndef f(x): ...\n") == [False]


@pytest.mark.unit
def test_partial_print_is_not_kernel() -> None:
    assert _kernel_flags("import functools\n@functools.partial(print)\ndef f(x): ...\n") == [False]


@pytest.mark.unit
def test_partial_without_args_is_not_kernel() -> None:
    assert _kernel_flags("import functools\n@functools.partial()\ndef f(x): ...\n") == [False]


@pytest.mark.unit
def test_plain_decorators_are_not_kernel() -> None:
    assert _kernel_flags("import functools\n@functools.wraps(print)\ndef f(x): ...\n") == [False]


# ---------------------------------------------------------------------------
# jitted_name_table
# ---------------------------------------------------------------------------


def _table(src: str) -> dict[str, str]:
    tree = ast.parse(src)
    return jitted_name_table(tree, alias_map(tree))


@pytest.mark.unit
def test_table_includes_jit_decorated_function() -> None:
    assert _table("import jax\n@jax.jit\ndef f(x): ...\n") == {"f": "jax.jit"}


@pytest.mark.unit
def test_table_includes_eqx_decorated_function() -> None:
    assert _table("import equinox as eqx\n@eqx.filter_jit\ndef f(x): ...\n") == {
        "f": "equinox.filter_jit"
    }


@pytest.mark.unit
def test_table_assignment_jax_jit_call() -> None:
    src = "import jax\ndef f(x): ...\nf_jit = jax.jit(f)\n"
    assert _table(src) == {"f_jit": "jax.jit"}


@pytest.mark.unit
def test_table_assignment_filter_jit_call() -> None:
    src = "import equinox as eqx\ndef f(x): ...\nf_jit = eqx.filter_jit(f)\n"
    assert _table(src) == {"f_jit": "equinox.filter_jit"}


@pytest.mark.unit
def test_table_assignment_partial_jit_call() -> None:
    src = "import functools\nimport jax\ndef f(x, n): ...\nf_jit = functools.partial(jax.jit, static_argnums=1)(f)\n"
    assert _table(src) == {"f_jit": "jax.jit"}


@pytest.mark.unit
def test_table_ignores_grad_and_plain_calls() -> None:
    src = "import jax\ndef f(x): ...\ng = jax.grad(f)\nh = len(f)\n"
    assert _table(src) == {}


@pytest.mark.unit
def test_table_ignores_unrelated_jit_name() -> None:
    src = "def jit(f):\n    return f\nf_jit = jit(f)\n"
    assert _table(src) == {}


# ---------------------------------------------------------------------------
# Fixtures: JL003 / JL004 / JL005 fire under the newly covered forms
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize(
    "filename",
    ["jax_names_eqx_filter_jit.py", "jax_names_bare_jit.py"],
)
def test_newly_covered_forms_fire_jl003_jl004_jl005(fixtures_dir: Path, filename: str) -> None:
    p = fixtures_dir / filename
    cfg = LintConfig(select=frozenset(["JL"]))
    diags = run_checks([p], cfg=cfg, perf=True, doc=False, mathjx=False)
    ids = {d.rule_id for d in diags}
    assert {"JL003", "JL004", "JL005"} <= ids
