import numpy as np


def cyclic_encode(value: float, max_val: int) -> tuple[float, float]:
    """
    Encode a cyclic feature (e.g., hour, day, month) into sine and cosine.

    Parameters
    ----------
    value : float
        The input value to encode (e.g., 12.5 for half past noon).
    max_val : int
        The maximum value of the cycle (e.g., 24 for hours, 7 for days, 12 for months).

    Returns
    -------
    tuple[float, float]
        A tuple containing (sin(angle), cos(angle)), where angle = 2π * value / max_val.
    """
    angle = 2 * np.pi * float(value) / max_val
    return np.sin(angle), np.cos(angle)
