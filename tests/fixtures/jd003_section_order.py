"""JD003: Returns section appears before Parameters (wrong canonical order)."""


def returns_before_params(x: float) -> float:
    """Scaled value.

    Returns
    -------
    float
        Scaled input.

    Parameters
    ----------
    x : float
        Input magnitude.
    """
    return 2.0 * x
