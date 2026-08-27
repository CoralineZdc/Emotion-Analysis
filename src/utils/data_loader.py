"""VAD Dataset class."""

from __future__ import print_function

import os
from typing import Optional, List, Callable

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image
import re


class DataLoader(data.Dataset):
    """VAD Dataset.

    Args:
        dataset: Name of the dataset (default: "fer").
        split: Dataset split to use ("Train", "Test", or "Val").
        include_V: Whether to include Valence labels (default: True).
        include_A: Whether to include Arousal labels (default: True).
        include_D: Whether to include Dominance labels (default: True).
        transform: Transform pipeline applied at sample fetch time.
    """

    label_mean: Optional[torch.Tensor] = None
    label_std: Optional[torch.Tensor] = None
    image_mean: Optional[torch.Tensor] = None
    image_std: Optional[torch.Tensor] = None

    data_protocol: str = "small_split"
    split_file_overrides: dict[str, str] = {}
    size: int = 48  # Image dimensions


    @classmethod
    def _repo_root(cls):
        return os.path.dirname(os.path.abspath(__file__))


    @classmethod
    def set_data_protocol(cls, protocol: str):
        if protocol not in {"small_split"}:
            raise ValueError("Unknown data protocol: {}".format(protocol))
        cls.data_protocol = protocol

    @classmethod
    def _get_split_candidates(cls, split: str, dataset: str = "fer") -> List[str]:
        if split in cls.split_file_overrides and cls.split_file_overrides[split]:
            return [cls.split_file_overrides[split]]        

        small_split_to_file = {
            "Train": [f"./data/train-{dataset}.csv"],
            "Test": [f"./data/test-{dataset}.csv"],
            "Val": [f"./data/val-{dataset}.csv"],
        }
        return small_split_to_file[split]
    

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
        val_file: Optional[str] = None
    ) -> None:
        split_map = {"Train": train_file, "Test": test_file, "Val": val_file,}
        for split_name, file_path in split_map.items():
            if file_path:
                cls.split_file_overrides[split_name] = file_path
            elif split_name in cls.split_file_overrides:
                del cls.split_file_overrides[split_name]


    @classmethod
    def _ensure_label_stats(cls, dataset: str) -> None:
        if cls.label_mean is not None and cls.label_std is not None: return  # Already computed

        train_path = cls._resolve_data_file(cls._get_split_candidates("Train", dataset))
        train_df = pd.read_csv(train_path)
        label_array = train_df[["Valence", "Arousal", "Dominance"]].to_numpy(dtype=np.float32)

        mean = label_array.mean(axis=0)
        std = label_array.std(axis=0)
        std[std == 0] = 1.0  # Avoid division by zero

        cls.label_mean = torch.tensor(mean, dtype=torch.float32)
        cls.label_std = torch.tensor(std, dtype=torch.float32)


    @classmethod
    def _ensure_image_stats(cls, dataset: str) -> None:
        if cls.image_mean is not None and cls.image_std is not None: return  # Already computed

        train_path = cls._resolve_data_file(cls._get_split_candidates("Train", dataset))
        train_df = pd.read_csv(train_path)
        pixels = []

        expected_length = cls.size * cls.size
        for pixel_entry in train_df["pixels"]:
            pixel_str = str(pixel_entry).strip()
            if not pixel_str or pixel_str.lower() == "nan":
                continue
            try:
                values = np.fromstring(pixel_str, dtype=np.float32, sep=' ')
                if len(values) == expected_length:
                    pixels.append(values / 255.0)
            except ValueError:
                continue

        if not pixels:
            cls.image_mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
            cls.image_std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
            return

        pixel_array = np.stack(pixels, axis=0)
        mean = float(pixel_array.mean())
        std = float(pixel_array.std())
        if std == 0.0: std = 1.0

        cls.image_mean = torch.tensor([mean, mean, mean], dtype=torch.float32)
        cls.image_std = torch.tensor([std, std, std], dtype=torch.float32)


    def __init__(
            self, 
            dataset: str = "fer",
            split: str = "Train", 
            include_V: bool = True, 
            include_A: bool = True, 
            include_D: bool = True, 
            transform: Optional[Callable] = None
        ) -> None:
        super().__init__()
        if split not in {"Train", "Test", "Val"}:
            raise ValueError("Unknown split: {}".format(split))

        self.transform = transform
        self.split = split
        self.dataset = dataset
        self.include_V = include_V
        self.include_A = include_A
        self.include_D = include_D

        active_indices = []
        if self.include_V: active_indices.append(0)
        if self.include_A: active_indices.append(1)
        if self.include_D: active_indices.append(2)

        if not active_indices:
            raise ValueError("At least one of include_V, include_A, or include_D must be True.")

        DataLoader._ensure_label_stats(dataset)
        DataLoader._ensure_image_stats(dataset)

        self.active_label_mean = DataLoader.label_mean[active_indices]
        self.active_label_std = DataLoader.label_std[active_indices]

        split_candidates = self._get_split_candidates(self.split, self.dataset)
        split_file = self._resolve_data_file(split_candidates)
        data_df = pd.read_csv(split_file)

        target_columns = ["Valence", "Arousal", "Dominance"]
        active_columns = [target_columns[i] for i in active_indices]

        processed_images = []
        processed_labels = []
        dropped_outliers = 0
        total_rows = len(data_df)
        expected_length = self.size * self.size







        """
        pixels_series = data_df["pixels"]
        valence_series = data_df["Valence"]
        arousal_series = data_df["Arousal"]
        dominance_series = data_df["Dominance"]
        label_series = valence_series, arousal_series, dominance_series
        label_series = [label_series[i] for i in active_indices]
        """


        for idx, row in data_df.iterrows():
            progress = (idx + 1) / total_rows
            bar = "█" * int(progress * 20) + " " * int(20 - int(progress * 20))
            print(f"Processing {self.split} Data: |{bar}| {idx + 1}/{total_rows} [{progress *100:.2f}%]", end="\r")

            pixel_str = str(row["pixels"]).strip()
            if not pixel_str or pixel_str.lower() == "nan":
                continue

            try:
                pixels = np.fromstring(pixel_str, dtype=np.uint8, sep=' ')
            except ValueError:
                continue

            if len(pixels) != expected_length:
                continue

            label_values = row[active_columns].to_numpy(dtype=np.float32)

            # Filter out rows with labels outside the range [-2.0, 2.0]
            if (label_values < -2.0).any() or (label_values > 2.0).any():
                dropped_outliers += 1
                continue

            # Convert the 1D pixel array to 3-channel RGB matrix (size, size, 3)
            arr_2d = np.array(pixels, dtype=np.uint8).reshape(self.size, self.size)
            arr_3d = np.stack([arr_2d, arr_2d, arr_2d], axis=2)
            processed_images.append(arr_3d)
            processed_labels.append(label_values)

        print(" "*100, end="\r") # Move to the next line after the progress bar
        if dropped_outliers > 0:
            print(f"[Label Warning] '{self.split}' split: Dropped {dropped_outliers} outlier samples with labels outside [-2.0, 2.0].")

        self.images = processed_images
        raw_labels = torch.tensor(processed_labels, dtype=torch.float32)
        self.labels = (raw_labels - self.active_label_mean) / self.active_label_std  # Uncomment this line if you want to normalize the labels

        # Summary statistics for the raw labels (before normalization)
        raw_labels = np.array(processed_labels) # Before normalization
        print(f"[{self.split}] Raw Mean (V,A,D): {raw_labels.mean(axis=0)}")
        print(f"[{self.split}] Raw Std  (V,A,D): {raw_labels.std(axis=0)}")
        print(f"[{self.split}] Normalized Mean (V,A,D): {self.labels.mean(dim=0).numpy()}")
        print(f"[{self.split}] Normalized Std  (V,A,D): {self.labels.std(dim=0).numpy()}")
        print(f"Finished processing {self.split} data. Total valid samples: {len(self.images)}")


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = Image.fromarray(self.images[index])
        if self.transform is not None: img = self.transform(img)
        target = self.labels[index]
        return img, target


    def __len__(self):
        return len(self.images)
