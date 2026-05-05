# Python API reference

Stable entry points for embedding jaxlint. These pages use **[autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html)** and a compact **[autosummary](https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html)** index of `jaxlint.core` re-exports.

`jaxlint.core` does **not** import JAX. Optional **`jaxlint[hlo]`** is documented narratively in [](hlo.md); API HTML for `jaxlint.hlo` is not built here so doc builds stay free of the JAX stack.

```{eval-rst}
.. currentmodule:: jaxlint.core

.. rubric:: ``jaxlint.core`` summary

.. autosummary::
   :nosignatures:

   Diagnostic
   Severity
   RULES
   describe_rules
   resolve_paths
   run_checks
   format_compact
   format_text
   format_json
   format_github_annotations_json
   format_sarif
   format_diagnostics

```
```{eval-rst}
.. automodule:: jaxlint.core
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: jaxlint.core.config
   :members:
   :undoc-members:
   :show-inheritance:

```
