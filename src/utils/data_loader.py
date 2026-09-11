"""VAD Dataset class supporting both VAD (3D) and VA (2D) datasets."""

from __future__ import print_function

import os
import re
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image


class DataLoader(data.Dataset):
    """VAD / VA Dataset.

    Args:
        dataset: Name of the dataset (e.g., "fer", "afew").
        split: Dataset split to use ("Train", "Test", or "Val").
        include_V: Whether to include Valence labels (default: True).
        include_A: Whether to include Arousal labels (default: True).
        include_D: Whether to include Dominance labels (default: True).
        transform: Transform pipeline applied at sample fetch time.
    """

    label_mid: Optional[int] = None
    label_half_range: Optional[float] = None

    image_mean: Optional[torch.Tensor] = None
    image_std: Optional[torch.Tensor] = None

    loaded_dataset: Optional[str] = None
    available_columns: List[str] = []

    data_protocol: str = "small_split"
    split_file_overrides: dict[str, str] = {}
    size: int = 224 # Image dimensions
    num_channels: int = 3  # Number of image channels (1 for pretrained models, 3 for training from scratch)

    @classmethod
    def _repo_root(cls) -> str:
        return os.path.dirname(os.path.abspath(__file__))

    @classmethod
    def set_data_protocol(cls, protocol: str):
        if protocol not in {"small_split"}:
            raise ValueError("Unknown data protocol: {}".format(protocol))
        cls.data_protocol = protocol

    @classmethod
    def set_num_channels(cls, num_channels: int):
        """Set the number of channels for image loading (1 for grayscale, 3 for RGB)."""
        if num_channels not in {1, 3}:
            raise ValueError(f"num_channels must be 1 or 3, got {num_channels}")
        cls.num_channels = num_channels

    @classmethod
    def set_image_size(cls, size: int):
        """Set the image size for loading (assumes square images)."""
        if size <= 0:
            raise ValueError(f"Image size must be positive, got {size}")
        cls.size = size

    @classmethod
    def _get_split_candidates(cls, split: str, dataset: str = "fer") -> List[str]:
        if split in cls.split_file_overrides and cls.split_file_overrides[split]:
            return [cls.split_file_overrides[split]]

        split_key = split.lower()
        return [f"./data/{split_key}-{dataset}.csv"]

    @classmethod
    def _resolve_data_file(cls, candidates: List[str]) -> str:
        for candidate in candidates:
            search_paths = [candidate]
            if not os.path.isabs(candidate):
                search_paths.append(os.path.join(cls._repo_root(), candidate))
                search_paths.append(os.path.join(cls._repo_root(), "data", candidate))

            for path in search_paths:
                if os.path.exists(path):
                    return path
        raise FileNotFoundError("No dataset file found. Tried: {}".format(", ".join(candidates)))

    @classmethod
    def set_split_files(
        cls,
        train_file: Optional[str] = None,
        test_file: Optional[str] = None,
        val_file: Optional[str] = None,
    ) -> None:
        split_map = {"Train": train_file, "Test": test_file, "Val": val_file}
        for split_name, file_path in split_map.items():
            if file_path:
                cls.split_file_overrides[split_name] = file_path
            elif split_name in cls.split_file_overrides:
                del cls.split_file_overrides[split_name]

    @classmethod
    def _ensure_label_stats(cls, dataset: str) -> None:
        # Recompute stats if dataset changes or if stats are not loaded
        if cls.loaded_dataset == dataset and cls.label_mid is not None:
            return

        train_path = cls._resolve_data_file(cls._get_split_candidates("Train", dataset))
        train_df = pd.read_csv(train_path)

        # Detect present columns among Valence, Arousal, Dominance
        all_possible = ["Valence", "Arousal", "Dominance"]
        cls.available_columns = [col for col in all_possible if col in train_df.columns]

        if not cls.available_columns:
            raise ValueError(f"No valid target columns found in dataset file '{train_path}'.")

        label_array = train_df[cls.available_columns].to_numpy(dtype=np.float32)

        # Mask out invalid entries (NaN or out-of-range values)
        valid_mask = np.all(np.isfinite(label_array), axis=1)
        valid_labels = label_array[valid_mask]

        min_vals = valid_labels.min(axis=0)
        max_vals = valid_labels.max(axis=0)
        mid_vals = (min_vals + max_vals) / 2.0
        half_ranges = (max_vals - min_vals) / 2.0

        half_ranges[half_ranges == 0] = 1.0  # Avoid division by zero

        cls.label_mid = torch.tensor(mid_vals, dtype=torch.float32)
        cls.label_half_range = torch.tensor(half_ranges, dtype=torch.float32)

    @classmethod
    def _ensure_image_stats(cls, dataset: str) -> None:
        if cls.loaded_dataset == dataset and cls.image_mean is not None:
            return

        train_path = cls._resolve_data_file(cls._get_split_candidates("Train", dataset))
        train_df = pd.read_csv(train_path)
        pixels = []

        pixels_length = len(train_df["pixels"][0].split())
        cls.size = int(np.sqrt(pixels_length))
        for pixel_entry in train_df["pixels"]:
            pixel_str = str(pixel_entry).strip()
            if not pixel_str or pixel_str.lower() == "nan":
                continue
            try:
                values = np.fromstring(pixel_str, dtype=np.float32, sep=" ")
                if len(values) == pixels_length:
                    pixels.append(values / 255.0)
            except ValueError:
                continue

        if not pixels:
            cls.image_mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
            cls.image_std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        else:
            pixel_array = np.stack(pixels, axis=0)
            mean = float(pixel_array.mean())
            std = float(pixel_array.std())
            if std == 0.0:
                std = 1.0

            cls.image_mean = torch.tensor([mean, mean, mean], dtype=torch.float32)
            cls.image_std = torch.tensor([std, std, std], dtype=torch.float32)

        cls.loaded_dataset = dataset

    def __init__(
        self,
        dataset: str = "fer",
        split: str = "Train",
        include_V: bool = True,
        include_A: bool = True,
        include_D: bool = True,
        transform: Optional[Callable] = None,
        display: bool = True,
    ) -> None:
        super().__init__()
        if split not in {"Train", "Test", "Val"}:
            raise ValueError("Unknown split: {}".format(split))

        self.transform = transform
        self.split = split
        self.dataset = dataset
        self.display = display

        DataLoader._ensure_label_stats(dataset)
        DataLoader._ensure_image_stats(dataset)

        self.include_V = include_V and ("Valence" in DataLoader.available_columns)
        self.include_A = include_A and ("Arousal" in DataLoader.available_columns)
        self.include_D = include_D and ("Dominance" in DataLoader.available_columns)

        active_columns = []
        active_indices = []

        for idx, col in enumerate(DataLoader.available_columns):
            if col == "Valence" and self.include_V:
                active_columns.append(col)
                active_indices.append(idx)
            elif col == "Arousal" and self.include_A:
                active_columns.append(col)
                active_indices.append(idx)
            elif col == "Dominance" and self.include_D:
                active_columns.append(col)
                active_indices.append(idx)

        if not active_columns:
            raise ValueError(
                f"No active columns selected or available for dataset '{dataset}'. "
                f"Available in CSV: {DataLoader.available_columns}"
            )

        self.active_label_mid = DataLoader.label_mid[active_indices]
        self.active_label_half_range = DataLoader.label_half_range[active_indices]

        split_candidates = self._get_split_candidates(self.split, self.dataset)
        split_file = self._resolve_data_file(split_candidates)
        data_df = pd.read_csv(split_file)

        processed_images = []
        processed_labels = []
        dropped_invalid = 0
        total_rows = len(data_df)
        expected_length = self.size * self.size

        for idx, row in data_df.iterrows():
            progress = (idx + 1) / total_rows
            bar = "█" * int(progress * 20) + " " * int(20 - int(progress * 20))
            print(f"Processing {self.split} Data: |{bar}| {idx + 1}/{total_rows} [{progress * 100:.2f}%]", end="\r") if self.display else None

            pixel_str = str(row["pixels"]).strip()
            if not pixel_str or pixel_str.lower() == "nan":
                empty_pixels_count += 1
                continue

            try:
                pixels = np.fromstring(pixel_str, dtype=np.uint8, sep=" ")
            except ValueError:
                dropped_invalid += 1
                continue

            if len(pixels) != expected_length:
                dropped_invalid += 1
                continue

            label_values = row[active_columns].to_numpy(dtype=np.float32)

            # Filter out outlier samples based on dataset limits
            if not np.isfinite(label_values).all():
                dropped_invalid += 1
                continue

            arr_2d = pixels.reshape(self.size, self.size)
            arr_channel = arr_2d if self.num_channels == 1 else np.stack([arr_2d] * 3, axis=-1)
            processed_images.append(arr_channel)
            processed_labels.append(label_values)

        print(" " * 100, end="\r") if self.display else None
        if dropped_invalid > 0 and self.display:
            print(
                f"[Label Warning] '{self.split}' split: Dropped {dropped_invalid} invalid samples."
            )

        self.images = processed_images
        raw_labels = torch.tensor(np.array(processed_labels), dtype=torch.float32)
        self.labels = (raw_labels - self.active_label_mid) / self.active_label_half_range
        raw_labels_np = np.array(processed_labels)
        print(f"Finished processing {self.split} data. Total valid samples: {len(self.images)}") if self.display else None

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = Image.fromarray(self.images[index])
        if self.transform is not None:
            img = self.transform(img)
        target = self.labels[index]
        return img, target

    def __len__(self) -> int:
        return len(self.images)