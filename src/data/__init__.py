"""Utility functions for data processing and manipulation."""

from pathlib import Path
from typing import Optional

from .preprocess import PreprocessAFEW, PreprocessDataset, PreprocessEMOTIC, PreprocessHECO, VADSample

DATASET_REGISTRY = {
    "afew": PreprocessAFEW,
    "emotic": PreprocessEMOTIC,
    "heco": PreprocessHECO,
}


def get_dataset_filename(split_name: str, dataset_name: str, age_suffix: Optional[str] = None) -> str:
    """
    Generate the normalized filename for a given split.
    
    Args:
        split_name: The name of the split.
        dataset_name: The name of the dataset.
        age_suffix: An optional suffix to include in the filename.

    Returns:
        The normalized filename.
    """
    suffix = f"-{age_suffix.lower()}" if age_suffix else ""
    return f"{split_name.lower()}-{dataset_name.lower()}{suffix}.csv"


__all__ = [
    "VADSample",
    "PreprocessDataset",
    "PreprocessAFEW",
    "PreprocessEMOTIC",
    "PreprocessHECO",
    "DATASET_REGISTRY",
    "get_dataset_filename",
]