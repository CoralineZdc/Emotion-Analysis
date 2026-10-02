"""Parsing utilities for command line arguments and path management."""

from pathlib import Path
from typing import Any, Callable, List, Tuple, TypeVar

T = TypeVar("T")


def get_project_root() -> Path:
    """Returns the absolute root directory of the project."""
    return Path(__file__).resolve().parents[2]


def parse_csv_list(value: str, item_type: Callable[[str], T] = str) -> List[T]:
    """Generic CSV string parser returning a typed list of items.
    
    Args:
        value: CSV string to parse.
        item_type: Callable to convert each item to the desired type (default: str).
    Returns:
        List of items of the specified type.
    """
    if not value or not value.strip():
        return []
    return [item_type(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_floats(value):
    """Parses a CSV string into a list of floats."""
    return parse_csv_list(value, float)


def parse_csv_ints(value):
    """Parses a CSV string into a list of integers."""
    return parse_csv_list(value, int)


def parse_csv_strings(value):
    """Parses a CSV string into a list of strings."""
    return parse_csv_list(value, str)


def parse_csv_bool_tuples(value):
    """Parses a CSV string into a list of boolean tuples."""
    result = []
    for item in value.split(";"):
        item = item.strip("(").strip(")")
        if not item:
            continue
        parts = item.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid format for boolean tuple: {item}")
        parts = [bool(int(x.strip())) for x in parts]
        result.append(tuple(parts))
    return result


def get_simu_params(state_dict_path: Path) -> dict[str, Any]:
    """
    Extracts simulation parameters from the directory name of the state_dict_path.
    
    Args:
        state_dict_path: Path to the state dict file.
    Returns:
        Dictionary of simulation parameters extracted from the directory name.
    """
    dir_name = state_dict_path.parent.name
    params = {}
    for param in dir_name.split("_"):
        if "-" in param:
            key, value = param.split("-", 1)
            params[key] = value
        else:
            idx = next((i for i, c in enumerate(param) if c.isdigit()), len(param))
            key, value = param[:idx], param[idx:]
            if key:
                params[key] = value
    return params
