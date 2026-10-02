"""VAD / VA Dataset module supporting 2D (VA) and 3D (VAD) target representations."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils.parsing_utils import get_project_root


@dataclass
class DatasetStats:
    label_mid: torch.Tensor
    label_half_range: torch.Tensor
    image_mean: List[float]
    image_std: List[float]
    image_size: int


class VADDataset(Dataset):
    """Dataset for continuous Valence-Arousal-Dominance (VAD) facial regression."""

    AVAILABLE_TARGETS = ["Valence", "Arousal", "Dominance"]

    def __init__(
        self,
        dataset: str = "fer",
        split: str = "Train",
        include_V: bool = True,
        include_A: bool = True,
        include_D: bool = True,
        num_channels: int = 3,
        transform: Optional[Callable] = None,
        split_overrides: Optional[dict[str, str]] = None,
        display: bool = True,
    ) -> None:
        super().__init__()
        if split not in {"Train", "Test", "Val"}:
            raise ValueError("Unknown split: {}".format(split))

        self.dataset = dataset
        self.split = split
        self.num_channels = num_channels
        self.transform = transform
        self.display = display
        self.split_overrides = split_overrides or {}

        # Resolve data path and compute/cache statistics for dataset split
        file_path = self._resolve_data_file(split, dataset)
        df = pd.read_csv(file_path)

        # Select target columns
        requested_targets = {
            "Valence": include_V,
            "Arousal": include_A,
            "Dominance": include_D,
        }
        self.active_columns = [
            col for col in self.AVAILABLE_TARGETS
            if col in df.columns and requested_targets.get(col, False)
        ]

        if not self.active_columns:
            raise ValueError(f"No active VAD target columns found in file '{file_path}'.")

        # Process and normalize labels & images in a single pass
        self.stats = self._compute_stats(df, self.active_columns)
        self.images, self.labels = self._process_dataframe(df)


    def _resolve_data_file(self, split: str, dataset: str) -> Path:
        """
        Resolve the path to the dataset CSV file for the given split and dataset.
        
        Args:
            split: Dataset split name (e.g., "Train", "Test", "Val").
            dataset: Dataset name (e.g., "fer", "afew").
        Returns:
            Path to the resolved CSV file.
        Raises:
            FileNotFoundError: If the CSV file cannot be found in any of the expected locations.
        """
        if split in self.split_overrides:
            return Path(self.split_overrides[split])

        filename = f"{split.lower()}-{dataset}.csv"
        root = get_project_root()
        candidates = [
            Path(filename),
            root / "data" / filename,
            root / filename,
        ]

        for path in candidates:
            if path.exists():
                return path.resolve()

        raise FileNotFoundError(f"Dataset split CSV not found for '{dataset}' ({split}).")


    def _compute_stats(self, df: pd.DataFrame, columns: List[str]) -> DatasetStats:
        """
        Compute dataset statistics (midpoint, half-range, image size) for normalization.
        
        Args:
            df: DataFrame containing the dataset split.
            columns: List of target columns to compute statistics for.
        Returns:
            DatasetStats object containing label midpoints, half-ranges, image mean/std, and image size.
        """
        label_arr = df[columns].to_numpy(dtype=np.float32)
        valid_mask = np.all(np.isfinite(label_arr), axis=1)
        valid_labels = label_arr[valid_mask]

        min_vals = valid_labels.min(axis=0)
        max_vals = valid_labels.max(axis=0)
        mid_vals = (min_vals + max_vals) / 2.0
        half_ranges = (max_vals - min_vals) / 2.0
        half_ranges[half_ranges == 0] = 1.0

        # Estimate spatial size from pixel string length
        first_pixel_str = str(df["pixels"].iloc[0]).strip()
        pixel_count = len(first_pixel_str.split())
        image_size = int(np.sqrt(pixel_count))

        return DatasetStats(
            label_mid=torch.tensor(mid_vals, dtype=torch.float32),
            label_half_range=torch.tensor(half_ranges, dtype=torch.float32),
            image_mean=[0.5] * self.num_channels,
            image_std=[0.5] * self.num_channels,
            image_size=image_size,
        )


    def _process_dataframe(self, df: pd.DataFrame) -> Tuple[List[np.ndarray], torch.Tensor]:
        """
        Process the DataFrame to extract images and normalized labels, filtering out invalid samples.

        Args:
            df: DataFrame containing the dataset split.
        Returns:
            Tuple of (list of processed images, tensor of normalized labels).
        """
        processed_images = []
        processed_labels = []
        expected_pixels = self.stats.image_size ** 2
        dropped = 0
        total_rows = len(df)

        for row_number, (_, row) in enumerate(df.iterrows(), start=1):
            progress = row_number / total_rows
            bar = "█" * int(progress * 20) + " " * int(20 - int(progress * 20))
            print(f"Processing {self.split} Data: |{bar}| {row_number}/{total_rows} [{progress * 100:.2f}%]", end="\r") if self.display else None

            pixel_str = str(row["pixels"]).strip()
            if not pixel_str or pixel_str.lower() == "nan":
                dropped += 1
                continue

            try:
                pixels = np.fromstring(pixel_str, dtype=np.uint8, sep=" ")
            except ValueError:
                dropped += 1
                continue

            if len(pixels) != expected_pixels:
                dropped += 1
                continue

            label_vals = row[self.active_columns].to_numpy(dtype=np.float32)
            if not np.isfinite(label_vals).all():
                dropped += 1
                continue

            arr_2d = pixels.reshape(self.stats.image_size, self.stats.image_size)
            arr_chan = arr_2d if self.num_channels == 1 else np.stack([arr_2d] * 3, axis=-1)

            processed_images.append(arr_chan)
            processed_labels.append(label_vals)

        if dropped > 0 and self.display:
            print(f"[{self.split} Warning] Dropped {dropped} invalid/corrupted samples.")

        # Normalize labels to [-1, 1] range based on computed midpoints and half-ranges
        raw_labels = torch.tensor(np.array(processed_labels), dtype=torch.float32)
        norm_labels = (raw_labels - self.stats.label_mid) / self.stats.label_half_range

        return processed_images, norm_labels


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset by index.

        Args:
            index: Index of the sample to retrieve.
        Returns:
            Tuple of (image tensor, label tensor).
        """
        img = Image.fromarray(self.images[index])
        if self.transform is not None:
            img = self.transform(img)
        target = self.labels[index]
        return img, target


    def __len__(self) -> int:
        return len(self.images)