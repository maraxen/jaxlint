# jaxlint-expect: JD002, JD005
"""JD005: return annotation without Returns section."""


def annotated_no_returns(x: int) -> int:
    """Do work on x.

    Parameters
    ----------
    x : int
        Input.
    """
    return x + 1
