"""VAD Dataset class."""

from __future__ import print_function

import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image
import re


class DataLoader(data.Dataset):
    """VAD Dataset.

    Args:
        split: Dataset split to use ("Train", "Test", or "Val").
        csv_path: Path to the CSV file (train, val, or test).
        transform: Transform pipeline applied at sample fetch time.
    """

    label_mean = None
    label_std = None
    image_mean = None
    image_std = None

    face_detector = None
    data_protocol = "small_split"
    split_file_overrides = {}

    size = 48  # Default size, can be overridden by dataset name (e.g., "fer48" -> size=48)

    @classmethod
    def _repo_root(cls):
        return os.path.dirname(os.path.abspath(__file__))


    @classmethod
    def set_data_protocol(cls, protocol):
        if protocol not in {"small_split"}:
            raise ValueError("Unknown data protocol: {}".format(protocol))
        cls.data_protocol = protocol


    @classmethod
    def _get_split_candidates(cls, split, dataset="fer"):
        if split in cls.split_file_overrides and cls.split_file_overrides[split]:
            return [cls.split_file_overrides[split]]

        

        small_split_to_file = {
            "Train": [f"./data/train-{dataset}.csv"],
            "Test": [f"./data/test-{dataset}.csv"],
            "Val": [f"./data/val-{dataset}.csv"],
        }

        return small_split_to_file[split]
    

    @classmethod
    def _resolve_data_file(cls, candidates):
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
    def set_split_files(cls, train_file=None, test_file=None, val_file=None):
        split_map = {
            "Train": train_file,
            "Test": test_file,
            "Val": val_file,
        }
        for split_name, file_path in split_map.items():
            if file_path:
                cls.split_file_overrides[split_name] = file_path
            elif split_name in cls.split_file_overrides:
                del cls.split_file_overrides[split_name]


    @classmethod
    def _ensure_label_stats(cls, dataset):
        train_path = cls._resolve_data_file(cls._get_split_candidates("Train", dataset))
        train_df = pd.read_csv(train_path)
        label_array = train_df[["Valence", "Arousal", "Dominance"]].to_numpy(dtype=np.float32)
        cls.label_mean = torch.tensor(label_array.mean(axis=0), dtype=torch.float32)
        cls.label_std = torch.tensor(label_array.std(axis=0), dtype=torch.float32)
        cls.label_std[cls.label_std == 0] = 1.0


    @classmethod
    def _ensure_image_stats(cls, dataset):
        train_path = cls._resolve_data_file(cls._get_split_candidates("Train", dataset))
        train_df = pd.read_csv(train_path)
        pixels = []
        for pixel_entry in train_df["pixels"]:
            pixel_str = str(pixel_entry).strip()
            if pixel_str.lower() == "nan" or not pixel_str:
                continue
            try:
                values = list(map(int, pixel_str.split()))
            except ValueError:
                continue
            if len(values) != cls.size * cls.size:
                continue
            pixels.append(np.array(values, dtype=np.float32) / 255.0)

        if not pixels:
            cls.image_mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
            cls.image_std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
            return

        pixel_array = np.stack(pixels, axis=0)
        mean = float(pixel_array.mean())
        std = float(pixel_array.std())
        if std == 0.0:
            std = 1.0

        cls.image_mean = torch.tensor([mean, mean, mean], dtype=torch.float32)
        cls.image_std = torch.tensor([std, std, std], dtype=torch.float32)


    def __init__(self, dataset = "fer",split="Train", transform=None):
        self.transform = transform
        self.split = split
        self.dataset = dataset

        if self.split not in {"Train", "Test", "Val"}:
            raise ValueError("Unknown split: {}".format(self.split))

        split_candidates = self._get_split_candidates(self.split, self.dataset)
        split_file = self._resolve_data_file(split_candidates)
        self.data = pd.read_csv(split_file)
        pixels_series = self.data["pixels"]
        valence_series = self.data["Valence"]
        arousal_series = self.data["Arousal"]
        dominance_series = self.data["Dominance"]

        processed_images = []
        processed_labels = []
        dropped_outliers = 0

        for idx, pixel_entry in enumerate(pixels_series):
            print(f"Processing {self.split} Data: |{'█'*int((idx+1) / len(pixels_series) * 20)}{' '*int(20 - int((idx+1) / len(pixels_series) * 20))}| {idx + 1}/{len(pixels_series)} [{(idx+1) / len(pixels_series) * 100:.2f}%]", end="\r")

            pixel_str = str(pixel_entry).strip()
            if pixel_str.lower() == "nan" or not pixel_str:
                continue

            try:
                pixels = list(map(int, pixel_str.split()))
            except ValueError:
                print("Warning: Could not parse {} pixel string '{}' at index {}. Skipping this entry.".format(self.split, pixel_str, idx))
                continue

            #if len(pixels) != 224 * 224:
            #    print("Warning: {} pixel string has incorrect length ({}) for 224x224 image at index {}. Skipping this entry.".format(self.split, len(pixels), idx))
            #    continue

            label_values = [
                valence_series.iloc[idx],
                arousal_series.iloc[idx],
                dominance_series.iloc[idx]
            ]

            label_array = np.asarray(label_values, dtype=np.float32)
            if (label_array < -2.0).any() or (label_array > 2.0).any():
                dropped_outliers += 1
                continue

            arr_2d = np.array(pixels, dtype=np.uint8).reshape(self.size, self.size)
            arr_3d = np.stack([arr_2d, arr_2d, arr_2d], axis=2)
            processed_images.append(arr_3d)
            processed_labels.append(label_values)

        if dropped_outliers > 0:
            print(
                "[Label Warning] '{}' dropped {} rows with labels outside [-2.0, 2.0].".format(
                    split_file,
                    dropped_outliers,
                )
            )

        self.images = processed_images
        labels = torch.tensor(processed_labels, dtype=torch.float32)
        self.labels = (labels - self.label_mean) / self.label_std

        if self.split == "Train":
            self.train_data = self.images
            self.train_labels = self.labels
        elif self.split == "Test":
            self.test_data = self.images
            self.test_labels = self.labels
        else:
            self.val_data = self.images
            self.val_labels = self.labels

        print("Finished processing {} data. Total valid samples: {}          ".format(self.split, len(self.images)))





    def __getitem__(self, index):
        img = Image.fromarray(self.images[index])
        if self.transform is not None:
            img = self.transform(img)
        target = self.labels[index]
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.float32)

        return img, target


    def __len__(self):
        return len(self.images)
