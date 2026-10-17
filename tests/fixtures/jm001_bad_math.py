# jaxlint-expect: JD002, JM001
"""Doc with bad math."""


def f() -> None:
    r"""Broken delimiter.

    Equation $$E = mc^2
    """
    pass
