# jaxlint-expect: JD004
"""JD004: undocumented parameter."""


def mismatch(a: int, b: int) -> int:
    """Combine values.

    Parameters
    ----------
    a : int
        First operand.

    Returns
    -------
    int
        Sum.
    """
    return a + b
