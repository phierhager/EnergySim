from dataclasses import is_dataclass, fields
from typing import Any
import numpy as np


def to_dict_filtered(obj, exclude=["type"], recursion=False):
    """Convert a dataclass to a dictionary, excluding specified fields."""
    if not is_dataclass(obj):
        raise ValueError("to_dict_filtered expects a dataclass instance")

    result = {}
    for f in fields(obj):
        if f.name in exclude:
            continue  # skip excluded fields
        value = getattr(obj, f.name)
        # Optionally recurse for nested dataclasses:
        if is_dataclass(value) and recursion:
            value = to_dict_filtered(value)
        result[f.name] = value
    return result


def numpy_to_python(data: Any) -> Any:
    """
    Recursively convert NumPy scalars and arrays to native Python types.
    - np.generic -> int or float
    - np.ndarray -> list (recursively converted)
    """
    if isinstance(data, np.generic):
        return data.item()  # np.int64, np.float64 → int, float
    elif isinstance(data, np.ndarray):
        return data.tolist()  # recursively converts elements
    elif isinstance(data, dict):
        return {k: numpy_to_python(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [numpy_to_python(x) for x in data]
    else:
        return data
