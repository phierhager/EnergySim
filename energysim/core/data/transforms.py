import time

from energysim.core.utils.functions import cyclic_encode
import numpy as np


def get_time_features(timestamp: int) -> np.ndarray:
    """
    Add cyclic time features (hour of day, day of week, month of year)
    encoded as sine and cosine to the sample dict.

    Parameters
    ----------
    timestamp : int
        UNIX timestamp.

    Returns
    -------
    dict
        The same dictionary, augmented with "time_features" as a float32 array.
    """
    dt = time.gmtime(timestamp)

    # Encode hour (0-23), day (0-6), month (0-11)
    features = [
        *cyclic_encode(dt.tm_hour, 24),  # hour of day
        *cyclic_encode(dt.tm_wday, 7),  # day of week
        *cyclic_encode(dt.tm_mon - 1, 12),  # month of year (0-based)
    ]

    return np.array(features, dtype=np.float32)
