# jaxlint-expect: none
"""Module doc."""


def ok(x: float) -> float:
    """Scale input.

    Parameters
    ----------
    x : float
        Value.

    Returns
    -------
    float
        Twice x.
    """
    return 2.0 * x
