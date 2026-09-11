import argparse
import glob
import json
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mediapipe as mp
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


@dataclass
class VADSample:
    """Standardized representation of a single processed sample."""
    name: str
    valence: float
    arousal: float
    dominance: Optional[float]
    pixels: str
    group_id: Optional[str] = None  # Used for group-based splitting (e.g., video ID)
    age: Optional[str] = None


class PreprocessDataset(ABC):
    """Abstract Base Class for continuous emotion dataset preprocessing."""

    def __init__(
        self,
        dataset_name: str,
        data_dir: Union[str, Path],
        output_dir: Union[str, Path],
        target_size: Tuple[int, int] = (112, 112),
        split_ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
        min_confidence: float = 0.5,
        margin: float = 0.15,
        seed: int = 42,
        target_age: Optional[str] = None,
        resplit: bool = True
    ):
        self.dataset_name = dataset_name.lower()
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.split_ratios = split_ratios
        self.min_confidence = min_confidence
        self.margin = margin
        self.seed = seed
        self.resplit = resplit
        self.target_age = target_age.strip().lower() if target_age else None

        self._face_detector = None 
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(self.seed)


    @property
    def face_detector(self):
        """Lazily initializes and returns a MediaPipe face detector."""
        if self._face_detector is None:
            self._face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=self.min_confidence
            )
        return self._face_detector


    def extract_face_crop(
            self,
            image: Image.Image,
            fallback_ratio: Optional[float] = 0.35
    ) -> Image.Image:
        """Generic face detection and cropping using MediaPipe, with optional fallback."""
        img_rgb = np.array(image.convert("RGB"))
        h_img, w_img, _ = img_rgb.shape

        results = self.face_detector.process(img_rgb)
        if results.detections:
            detection = max(results.detections, key=lambda d: d.score[0])
            bbox = detection.location_data.relative_bounding_box

            x_min, y_min = int(bbox.xmin * w_img), int(bbox.ymin * h_img)
            w, h = int(bbox.width * w_img), int(bbox.height * h_img)

            dw, dh = int(w * self.margin), int(h * self.margin)
            x1, y1 = max(0, x_min - dw), max(0, y_min - dh)
            x2, y2 = min(w_img, x_min + w + dw), min(h_img, y_min + h + dh)

            if x2 > x1 and y2 > y1:
                return image.crop((x1, y1, x2, y2))
             
        if fallback_ratio is not None:
            head_h = max(1, int(h_img * fallback_ratio))
            return image.crop((0, 0, w_img, head_h))

        return image


    def process_image_to_pixel_string(self, img_input: Union[Image.Image, np.ndarray]) -> str:
        """Converts PIL Image or numpy array to a space-separated 1D grayscale pixel string."""
        if not isinstance(img_input, Image.Image):
            arr = np.asarray(img_input)
            if arr.ndim == 3 and arr.shape[0] in (1, 3):  # Convert (C, H, W) -> (H, W, C)
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
        else:
            img = img_input

        gray_img = img.convert("L")
        if self.target_size is not None:
            gray_img = gray_img.resize(self.target_size, Image.Resampling.BILINEAR)

        pixel_array = np.array(gray_img, dtype=np.uint8).flatten()
        return " ".join(pixel_array.astype(str))

    def predict_age(self, img_input: Union[Image.Image, np.ndarray]) -> Optional[str]:
        """Placeholder for age prediction; can be overridden in subclasses."""
        return None

    # -------------------------------------------------------------------------
    # Abstract Methods to Implement in Subclasses
    # -------------------------------------------------------------------------
    @abstractmethod
    def crop_sample(self, raw_source: Union[Path, Image.Image, np.ndarray], **kwargs) -> Image.Image:
        """Dataset-specific face/subject cropping strategy."""
        pass


    @abstractmethod
    def parse_samples(self) -> Dict[str, List[VADSample]]:
        """Parses raw dataset metadata and returns samples mapped by split name.
        
        Returns:
            Dict[str, List[VADSample]]: e.g., {"all": [...]} or {"train": [...], "val": [...]}
        """
        pass

    # -------------------------------------------------------------------------
    # Generic Splitting & Export Pipeline
    # -------------------------------------------------------------------------
    @staticmethod
    def _compute_range_agnostic_bins(values: np.ndarray, n_bins: int = 4) -> np.ndarray:
        """Computes discrete bin indices in a range-agnostic manner.
        
        Uses quantile binning (percentiles) first; falls back to min-max 
        uniform binning if quantiles fail due to low variance.
        """
        if len(values) == 0:
            return np.array([], dtype=int)
            
        # Strategy 1: Quantile/percentile binning (inherently scale-agnostic)
        try:
            return pd.qcut(values, q=n_bins, labels=False, duplicates='drop')
        except Exception:
            pass

        # Strategy 2: Dynamic Min-Max normalization fallback
        v_min, v_max = values.min(), values.max()
        if np.isclose(v_min, v_max):
            return np.zeros(len(values), dtype=int)

        normalized = (values - v_min) / (v_max - v_min + 1e-7)
        return np.clip((normalized * n_bins).astype(int), 0, n_bins - 1)


    def split_dataset(
            self, samples: List[VADSample]
        ) -> Dict[str, List[VADSample]]:
        """Splits samples into train/val/test with scale-agnostic VAD/VA stratification and group leakage prevention."""
        train_r, val_r, test_r = self.split_ratios
        if not np.isclose(train_r + val_r + test_r, 1.0):
            raise ValueError("Split ratios must sum up to 1.0")

        rng = random.Random(self.seed)

        # 1. Determine if Dominance is available across samples
        has_dominance = any(s.dominance is not None for s in samples)
        n_bins = 3 if has_dominance else 4  # 3^3 = 27 strata vs 4^2 = 16 strata

        # -------------------------------------------------------------
        # CASE A: Group-Based Splitting (Prevents Data Leakage)
        # -------------------------------------------------------------
        if any(s.group_id is not None for s in samples):
            groups: Dict[str, List[VADSample]] = {}
            for s in samples:
                gid = s.group_id if s.group_id is not None else s.name
                groups.setdefault(gid, []).append(s)

            group_ids = list(groups.keys())

            mean_v = np.array([np.mean([s.valence for s in groups[gid]]) for gid in group_ids])
            mean_a = np.array([np.mean([s.arousal for s in groups[gid]]) for gid in group_ids])

            v_bins = self._compute_range_agnostic_bins(mean_v, n_bins=n_bins)
            a_bins = self._compute_range_agnostic_bins(mean_a, n_bins=n_bins)

            if has_dominance:
                mean_d = np.array([
                    np.mean([s.dominance for s in groups[gid] if s.dominance is not None])
                    for gid in group_ids
                ])
                d_bins = self._compute_range_agnostic_bins(mean_d, n_bins=n_bins)
                strata_keys = [f"{vb}_{ab}_{db}" for vb, ab, db in zip(v_bins, a_bins, d_bins)]
            else:
                strata_keys = [f"{vb}_{ab}" for vb, ab in zip(v_bins, a_bins)]

            strata_map: Dict[str, List[Tuple[str, int]]] = {}
            for gid, stratum in zip(group_ids, strata_keys):
                strata_map.setdefault(stratum, []).append((gid, len(groups[gid])))

            splits: Dict[str, List[VADSample]] = {"train": [], "val": [], "test": []}

            for stratum, g_list in strata_map.items():
                rng.shuffle(g_list)
                tot_stratum_samples = sum(count for _, count in g_list)
                target_train = tot_stratum_samples * train_r
                target_val = tot_stratum_samples * val_r

                curr_train, curr_val = 0, 0
                for gid, count in g_list:
                    if curr_train < target_train:
                        splits["train"].extend(groups[gid])
                        curr_train += count
                    elif curr_val < target_val:
                        splits["val"].extend(groups[gid])
                        curr_val += count
                    else:
                        splits["test"].extend(groups[gid])

            return splits

        # -------------------------------------------------------------
        # CASE B: Standard Sample-Level Stratified Split
        # -------------------------------------------------------------
        v_vals = np.array([s.valence for s in samples])
        a_vals = np.array([s.arousal for s in samples])

        v_bins = self._compute_range_agnostic_bins(v_vals, n_bins=n_bins)
        a_bins = self._compute_range_agnostic_bins(a_vals, n_bins=n_bins)

        if has_dominance:
            d_vals = np.array([s.dominance if s.dominance is not None else 0.0 for s in samples])
            d_bins = self._compute_range_agnostic_bins(d_vals, n_bins=n_bins)
            strata_keys = [f"{vb}_{ab}_{db}" for vb, ab, db in zip(v_bins, a_bins, d_bins)]
        else:
            strata_keys = [f"{vb}_{ab}" for vb, ab in zip(v_bins, a_bins)]

        strata_map: Dict[str, List[VADSample]] = {}
        for s, stratum in zip(samples, strata_keys):
            strata_map.setdefault(stratum, []).append(s)

        splits: Dict[str, List[VADSample]] = {"train": [], "val": [], "test": []}

        for stratum, s_list in strata_map.items():
            rng.shuffle(s_list)
            n_tot = len(s_list)
            n_tr = int(n_tot * train_r)
            n_va = int(n_tot * val_r)

            splits["train"].extend(s_list[:n_tr])
            splits["val"].extend(s_list[n_tr : n_tr + n_va])
            splits["test"].extend(s_list[n_tr + n_va :])

        return splits


    def save_split_csv(self, samples: List[VADSample], split_name: str) -> None:
        """Exports processed split samples to CSV."""
        if not samples:
            print(f"[{self.dataset_name.upper()}] Warning: Split '{split_name}' is empty.")
            return

        rows = []
        for idx, sample in enumerate(samples):
            row = {
                "name": sample.name if sample.name else f"{split_name}_{idx}",
                "Valence": sample.valence,
                "Arousal": sample.arousal,
            }
            if sample.dominance is not None:
                row["Dominance"] = sample.dominance
            row["pixels"] = sample.pixels
            rows.append(row)

        df = pd.DataFrame(rows)

        if self.target_age:
            age_suffix = "child" if self.target_age == "child" else self.target_age
            filename = f"{split_name}-{self.dataset_name}-{age_suffix}.csv"
        else:
            filename = f"{split_name}-{self.dataset_name}.csv"

        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        print(f"[{self.dataset_name.upper()}] Saved {split_name:<5} split ({len(df):>6} samples) -> {output_path}")


    def close(self) -> None:
        """Closes any resources (e.g., MediaPipe face detector)."""
        if self._face_detector is not None:
            self._face_detector.close()
            self._face_detector = None


    def process(self) -> None:
        """Template Method executing the end-to-end preprocessing pipeline."""
        print(f"\n--- Starting Processing for {self.dataset_name.upper()} ---")
        try:
            parsed_dict = self.parse_samples()
            total_samples = sum(len(v) for v in parsed_dict.values())
            print(f"[{self.dataset_name.upper()}] Parsed {total_samples} valid samples.")

            if self.resplit or "all" not in parsed_dict:
                all_samples = []
                for sample_list in parsed_dict.values():
                    all_samples.extend(sample_list)
                splits = self.split_dataset(all_samples)
            else:
                splits = parsed_dict  # Already split

            for split_name, split_samples in splits.items():
                self.save_split_csv(split_samples, split_name)
        finally:
            self.close()
            print(f"--- Finished Processing for {self.dataset_name.upper()} ---\n")



class PreprocessAFEW(PreprocessDataset):
    """Preprocessor for the AFEW-VA video dataset."""

    def __init__(
        self,
        data_dir: Union[str, Path] = "../Data/VA/AFEW-VA",
        output_dir: Union[str, Path] = "./data",
        target_size: Tuple[int, int] = (112, 112),
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        min_confidence: float = 0.5,
        margin: float = 0.15,
        seed: int = 42,
        target_age: Optional[str] = None,
        resplit: bool = True
    ):
        super().__init__(
            "afew",
            data_dir=data_dir,
            output_dir=output_dir,
            target_size=target_size,
            split_ratios=split_ratios,
            min_confidence=min_confidence,
            margin=margin,
            seed=seed,
            target_age=target_age,
            resplit=resplit
        )


    def crop_sample(self, raw_source: Image.Image) -> Image.Image:
        """Crops a single frame using MediaPipe face detection with margin and fallback."""
        return self.extract_face_crop(raw_source, fallback_ratio=None)
    

    def parse_samples(self) -> Dict[str, List[VADSample]]:
        video_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        all_samples: List[VADSample] = []

        for video_dir in tqdm(video_dirs, desc="[AFEW] Parsing frames"):
            json_files = list(video_dir.glob("*.json"))
            if not json_files:
                continue

            with open(json_files[0], "r", encoding="utf-8") as f:
                meta = json.load(f)

            for frame_id, frame_info in meta.get("frames", {}).items():
                img_path = video_dir / f"{frame_id}.png"
                if not img_path.exists():
                    img_path = video_dir / f"{frame_id}.jpg"
                    if not img_path.exists():
                        continue

                with Image.open(img_path) as img:

                    # Hook for future age-based filtering; currently a placeholder
                    if self.target_age:
                        age_val = self.predict_age(img)
                        if age_val and age_val.lower() != self.target_age:
                            continue

                    cropped_img = self.crop_sample(img)
                    pixel_str = self.process_image_to_pixel_string(cropped_img)

                all_samples.append(
                    VADSample(
                        name=f"{video_dir.name}_{frame_id}",
                        valence=float(frame_info["valence"]),
                        arousal=float(frame_info["arousal"]),
                        dominance=None,
                        pixels=pixel_str,
                        group_id=video_dir.name,  # Assign video name as group_id
                        age=age_val if self.target_age else None
                    )
                )
        return {"all": all_samples}
    

class PreprocessEMOTIC(PreprocessDataset):
    """Preprocessor for the EMOTIC VAD dataset."""

    def __init__(
        self,
        data_dir: Union[str, Path] = "../Data/VAD/EMOTIC",
        output_dir: Union[str, Path] = "./data",
        split_ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
        target_size: Tuple[int, int] = (112, 112),
        include_extra: bool = False,
        min_confidence: float = 0.5,
        margin: float = 0.15,
        seed: int = 42,
        target_age: Optional[str] = None,
        resplit: bool = True,
    ):
        super().__init__(
            "emotic", 
            data_dir=data_dir, 
            output_dir=output_dir, 
            target_size=target_size, 
            split_ratios=split_ratios, 
            min_confidence=min_confidence,
            margin=margin,
            seed=seed,
            target_age=target_age,
            resplit=resplit
        )
        self.include_extra = include_extra
        self.annots_dir = self.data_dir / "annots_arrs"
        self.img_arrs_dir = self.data_dir / "img_arrs"


    def crop_sample(
        self, 
        raw_source: Union[Image.Image, np.ndarray], 
        bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> Image.Image:
        """Crops the body bounding box from the image and then applies face detection for final cropping."""
        if not isinstance(raw_source, Image.Image):
            img = Image.fromarray(np.asarray(raw_source, dtype=np.uint8))
        else:
            img = raw_source

        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            w, h = img.size
            x1, x2 = max(0, int(x_min)), min(w, int(x_max))
            y1, y2 = max(0, int(y_min)), min(h, int(y_max))

            if x2 > x1 and y2 > y1:
                body_crop = img.crop((x1, y1, x2, y2))
            else:
                body_crop = img
        else:
            body_crop = img

        return self.extract_face_crop(body_crop)


    def parse_samples(self) -> Dict[str, List[VADSample]]:
        """Parses EMOTIC metadata adhering to official split files."""
        splits_map = {
            "train": ["annot_arrs_train.csv"],
            "val": ["annot_arrs_val.csv"],
            "test": ["annot_arrs_test.csv"],
        }
        if self.include_extra:
            splits_map["train"].append("annot_arrs_extra_train.csv")

        parsed_splits: Dict[str, List[VADSample]] = {}

        for split_key, csv_filenames in splits_map.items():
            split_samples = []
            for csv_name in csv_filenames:
                csv_path = self.annots_dir / csv_name
                if csv_path.exists():
                    split_samples.extend(self._parse_emotic_csv(csv_path))
            parsed_splits[split_key] = split_samples

        return parsed_splits


    def _load_array_cache(self, file_path: Path) -> Optional[np.ndarray]:
        """Loads a numpy array from disk with error handling."""
        if not file_path.exists():
            return None
        try:
            return np.load(str(file_path), allow_pickle=True)
        except Exception:
            return None


    def _find_raw_image(self, filename: str) -> Optional[Path]:
        """Searches recursively for a raw image file in the dataset directory."""
        matches = list(self.data_dir.glob(f"**/{filename}"))
        return matches[0] if matches else None


    def _parse_emotic_csv(self, csv_file: Path) -> List[VADSample]:
        """Parse a single EMOTIC CSV file and return a list of VADSample instances."""
        df = pd.read_csv(csv_file)

        if "Crop_name" in df.columns:
            df["crop_file_idx"] = df.groupby("Crop_name").cumcount()
        if "Arr_name" in df.columns:
            df["arr_file_idx"] = df.groupby("Arr_name").cumcount()

        array_cache: Dict[str, Optional[np.ndarray]] = {}
        samples: List[VADSample] = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"[EMOTIC] {csv_file.name}"):

            age_val = str(row["Age"]).strip().lower() if "Age" in row and pd.notna(row["Age"]) else None
            if self.target_age and age_val != ("kid" if self.target_age == "child" else self.target_age):
                continue

            # Validate VAD values
            v_val, a_val, d_val = row["Valence"], row["Arousal"], row["Dominance"]
            if not (np.isfinite(v_val) and np.isfinite(a_val) and np.isfinite(d_val)):
                continue

            pixel_str = None
            bbox = (row["X_min"], row["Y_min"], row["X_max"], row["Y_max"])
            group_id = str(row["Filename"]) if "Filename" in row and pd.notna(row["Filename"]) else None

            # Strategy 1: Pre-cropped numpy array
            if "Crop_name" in row and pd.notna(row["Crop_name"]):
                crop_file = str(row["Crop_name"])
                if crop_file not in array_cache:
                    array_cache[crop_file] = self._load_array_cache(self.img_arrs_dir / crop_file)

                crops = array_cache[crop_file]
                if crops is not None and row["crop_file_idx"] < len(crops):
                    pixel_str = self.process_image_to_pixel_string(crops[row["crop_file_idx"]])

            # Strategy 2: Full image numpy array + Bounding Box
            if pixel_str is None and "Arr_name" in row and pd.notna(row["Arr_name"]):
                arr_file = str(row["Arr_name"])
                if arr_file not in array_cache:
                    array_cache[arr_file] = self._load_array_cache(self.img_arrs_dir / arr_file)

                arrs = array_cache[arr_file]
                if arrs is not None and row["arr_file_idx"] < len(arrs):
                    cropped_img = self.crop_sample(arrs[row["arr_file_idx"]], bbox=bbox)
                    pixel_str = self.process_image_to_pixel_string(cropped_img)

            # Strategy 3: Raw image file on disk + Bounding Box
            if pixel_str is None and "Filename" in row and pd.notna(row["Filename"]):
                raw_path = self._find_raw_image(str(row["Filename"]))
                if raw_path and raw_path.exists():
                    try:
                        with Image.open(raw_path) as raw_img:
                            cropped = self.crop_sample(raw_img, bbox=bbox)
                            pixel_str = self.process_image_to_pixel_string(cropped)
                    except Exception:
                        pass

            if pixel_str is not None:
                samples.append(
                    VADSample(
                        group_id=group_id,
                        name=f"{row['Filename']}_{idx}",
                        valence=float(v_val),
                        arousal=float(a_val),
                        dominance=float(d_val),
                        pixels=pixel_str,
                        age=age_val
                    )
                )

        return samples



class PreprocessHECO(PreprocessDataset):
    """Preprocessor for the HECO dataset with body crop and face detection."""

    def __init__(
        self,
        data_dir: Union[str, Path] = "../Data/VAD/HECO",
        output_dir: Union[str, Path] = "./data",
        split_ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
        target_size: Tuple[int, int] = (112, 112),
        min_confidence: float = 0.4,
        margin: float = 0.15,
        seed: int = 42,
        target_age: Optional[str] = None,
        resplit: bool = True
    ):
        super().__init__(
            "heco",
            data_dir=data_dir,
            output_dir=output_dir,
            target_size=target_size,
            split_ratios=split_ratios,
            min_confidence=min_confidence,
            margin=margin,
            seed=seed,
            target_age=target_age,
            resplit=resplit
        )
        self.images_dir = self.data_dir / "Data"
        self.labels_file = self.data_dir / "Labels" / "HECO_Labels.csv"

    def crop_sample(
        self, 
        raw_source: Union[Image.Image, np.ndarray], 
        bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> Image.Image:
        """Crops the body bounding box, then detects and crops the face inside it via base class."""
        if not isinstance(raw_source, Image.Image):
            img = Image.fromarray(np.asarray(raw_source, dtype=np.uint8))
        else:
            img = raw_source

        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            w, h = img.size
            x1, x2 = max(0, int(x_min)), min(w, int(x_max))
            y1, y2 = max(0, int(y_min)), min(h, int(y_max))

            if x2 > x1 and y2 > y1:
                body_crop = img.crop((x1, y1, x2, y2))
            else:
                body_crop = img
        else:
            body_crop = img

        return self.extract_face_crop(body_crop, fallback_ratio=0.35)

    def parse_samples(self) -> Dict[str, List[VADSample]]:
        """Parses HECO metadata and crops corresponding face regions."""
        if not self.labels_file.exists():
            raise FileNotFoundError(f"Labels file not found at {self.labels_file}")

        df = pd.read_csv(self.labels_file)
        samples: List[VADSample] = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="[HECO] Parsing samples"):
            img_name = str(row["Image"])
            img_path = self.images_dir / img_name

            if not img_path.exists():
                continue

            v_val, a_val, d_val = row["Valence"], row["Arousal"], row["Dominance"]
            if not (np.isfinite(v_val) and np.isfinite(a_val) and np.isfinite(d_val)):
                continue

            bbox = (row["xmin"], row["ymin"], row["xmax"], row["ymax"])

            try:
                with Image.open(img_path) as raw_img:
                    cropped_face = self.crop_sample(raw_img, bbox=bbox)
                    pixel_str = self.process_image_to_pixel_string(cropped_face)

                    # Hook for future age-based filtering; currently a placeholder
                    if self.target_age:
                        age_val = self.predict_age(cropped_face)
                        if age_val and age_val.lower() != self.target_age:
                            continue

                samples.append(
                    VADSample(
                        name=f"{Path(img_name).stem}_{idx}",
                        valence=float(v_val),
                        arousal=float(a_val),
                        dominance=float(d_val),
                        pixels= pixel_str,
                        age=age_val if self.target_age else None
                    )
                )
            except Exception:
                continue

        return {"all": samples}



def main():
    parser = argparse.ArgumentParser(description="Preprocess continuous VAD emotion datasets.")
    parser.add_argument("--dataset", type=str, required=True, choices=["afew", "emotic", "heco", "all"], help="Dataset to preprocess")
    parser.add_argument("--data_dir", type=str, default=None, help="Root path to input dataset directory")
    parser.add_argument("--output_dir", type=str, default="./data", help="Path to output processed CSVs")
    parser.add_argument("--image_size", type=int, default=112, help="Output image size (width & height)")
    parser.add_argument("--target_age", type=str, default=None, choices=["child", "adult"], help="Filter samples by target age group")
    parser.add_argument("--include_extra", action="store_true", help="Include EMOTIC extra training data")
    parser.add_argument("--no_resplit", action="store_true", help="Disable forced-resplitting of dataset and retain original splits if available")
    args = parser.parse_args()

    size = (args.image_size, args.image_size)

    if args.dataset in ["afew", "all"]:
        afew_dir = args.data_dir if args.data_dir else "../Data/VA/AFEW-VA"
        afew_processor = PreprocessAFEW(
            data_dir=afew_dir, 
            output_dir=args.output_dir, 
            target_size=size, 
            target_age=args.target_age, 
            resplit=not args.no_resplit
        )
        afew_processor.process()

    if args.dataset in ["emotic", "all"]:
        emotic_dir = args.data_dir if args.data_dir else "../Data/VAD/EMOTIC"
        emotic_processor = PreprocessEMOTIC(
            data_dir=emotic_dir, 
            output_dir=args.output_dir, 
            target_size=size, 
            target_age=args.target_age,
            include_extra=args.include_extra,
            resplit=not args.no_resplit
        )
        emotic_processor.process()


    if args.dataset in ["heco", "all"]:
            heco_dir = args.data_dir if args.data_dir else "../Data/VAD/HECO"
            heco_processor = PreprocessHECO(
                data_dir=heco_dir, 
                output_dir=args.output_dir, 
                target_size=size, 
                target_age=args.target_age,
                resplit=not args.no_resplit
            )
            heco_processor.process()

        

if __name__ == "__main__":
    main()