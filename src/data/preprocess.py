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


class PreprocessDataset(ABC):
    """Abstract Base Class for continuous emotion dataset preprocessing."""

    def __init__(
        self,
        dataset_name: str,
        data_dir: Union[str, Path],
        output_dir: Union[str, Path],
        target_size: Tuple[int, int] = (112, 112),
        split_ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
        seed: int = 42,
    ):
        self.dataset_name = dataset_name.lower()
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.split_ratios = split_ratios
        self.seed = seed

        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(self.seed)

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
    def split_dataset(
        self, samples: List[VADSample]
    ) -> Dict[str, List[VADSample]]:
        """Splits samples into train/val/test using sample-level or group-level partitioning."""
        train_r, val_r, test_r = self.split_ratios
        if not np.isclose(train_r + val_r + test_r, 1.0):
            raise ValueError("Split ratios must sum up to 1.0")

        # Case A: Group-based split (e.g., group frames by video folder to prevent leakage)
        if any(s.group_id is not None for s in samples):
            groups: Dict[str, List[VADSample]] = {}
            for s in samples:
                groups.setdefault(s.group_id, []).append(s)

            group_keys = list(groups.keys())
            random.shuffle(group_keys)

            n_total = len(group_keys)
            n_train = int(n_total * train_r)
            n_val = int(n_total * val_r)

            train_groups = set(group_keys[:n_train])
            val_groups = set(group_keys[n_train : n_train + n_val])

            splits = {"train": [], "val": [], "test": []}
            for group_id, item_list in groups.items():
                if group_id in train_groups:
                    splits["train"].extend(item_list)
                elif group_id in val_groups:
                    splits["val"].extend(item_list)
                else:
                    splits["test"].extend(item_list)
            return splits

        # Case B: Standard sample-level random split
        shuffled = samples.copy()
        random.shuffle(shuffled)
        n_total = len(shuffled)
        n_train = int(n_total * train_r)
        n_val = int(n_total * val_r)

        return {
            "train": shuffled[:n_train],
            "val": shuffled[n_train : n_train + n_val],
            "test": shuffled[n_train + n_val :],
        }


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
        output_path = self.output_dir / f"{split_name}-{self.dataset_name}.csv"
        df.to_csv(output_path, index=False)
        print(f"[{self.dataset_name.upper()}] Saved {split_name:<5} split ({len(df):>6} samples) -> {output_path}")



    def process(self) -> None:
        """Template Method executing the end-to-end preprocessing pipeline."""
        print(f"\n--- Starting Processing for {self.dataset_name.upper()} ---")
        parsed_dict = self.parse_samples()
        print(f"[{self.dataset_name.upper()}] Parsed {sum(len(v) for v in parsed_dict.values())} valid samples.")


        if "all" in parsed_dict:
            splits = self.split_dataset(parsed_dict["all"])
        else:
            splits = parsed_dict  # Already split

        for split_name, split_samples in splits.items():
            self.save_split_csv(split_samples, split_name)


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
    ):
        super().__init__("afew", data_dir=data_dir, output_dir=output_dir, target_size=target_size, split_ratios=split_ratios, seed=seed)
        self.min_confidence = min_confidence
        self.margin = margin
        self.mp_face_detection = mp.solutions.face_detection



    def crop_sample(self, raw_source: Image.Image, face_detector) -> Image.Image:
        """Detects face bounding box using MediaPipe and crops PIL Image."""
        img_rgb = np.array(raw_source.convert("RGB"))
        h_img, w_img, _ = img_rgb.shape

        results = face_detector.process(img_rgb) if face_detector else None
        if not results or not results.detections:
            return raw_source  # Return original if no face detected

        detection = max(results.detections, key=lambda d: d.score[0])
        bbox = detection.location_data.relative_bounding_box

        x_min, y_min = int(bbox.xmin * w_img), int(bbox.ymin * h_img)
        w, h = int(bbox.width * w_img), int(bbox.height * h_img)

        dw, dh = int(w * self.margin), int(h * self.margin)
        x1, y1 = max(0, x_min - dw), max(0, y_min - dh)
        x2, y2 = min(w_img, x_min + w + dw), min(h_img, y_min + h + dh)

        return raw_source.crop((x1, y1, x2, y2))
    

    def parse_samples(self) -> Dict[str, List[VADSample]]:
        video_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        all_samples: List[VADSample] = []

        with self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=self.min_confidence
        ) as face_detector:

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
                        cropped_img = self.crop_sample(img, face_detector=face_detector)
                        pixel_str = self.process_image_to_pixel_string(cropped_img)

                    all_samples.append(
                        VADSample(
                            name=f"{video_dir.name}_{frame_id}",
                            valence=float(frame_info["valence"]),
                            arousal=float(frame_info["arousal"]),
                            dominance=None,
                            pixels=pixel_str,
                            group_id=video_dir.name,  # Assign video name as group_id
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
        seed: int = 42,
    ):
        super().__init__("emotic", data_dir=data_dir, output_dir=output_dir, target_size=target_size, split_ratios=split_ratios, seed=seed)
        self.include_extra = include_extra
        self.annots_dir = self.data_dir / "annots_arrs"
        self.img_arrs_dir = self.data_dir / "img_arrs"


    def crop_sample(
        self, 
        raw_source: Union[Image.Image, np.ndarray], 
        bbox: Tuple[float, float, float, float] = None
    ) -> Image.Image:
        """Overrides crop_sample with coordinate-based bounding box cropping."""
        if bbox is None:
            return raw_source if isinstance(raw_source, Image.Image) else Image.fromarray(raw_source)

        x_min, y_min, x_max, y_max = bbox
        if not isinstance(raw_source, Image.Image):
            img = Image.fromarray(np.asarray(raw_source, dtype=np.uint8))
        else:
            img = raw_source

        width, height = img.size
        x1, x2 = max(0, int(x_min)), min(width, int(x_max))
        y1, y2 = max(0, int(y_min)), min(height, int(y_max))

        if x2 <= x1 or y2 <= y1:
            return img
        return img.crop((x1, y1, x2, y2))


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
            # Validate VAD values
            v_val, a_val, d_val = row["Valence"], row["Arousal"], row["Dominance"]
            if not (np.isfinite(v_val) and np.isfinite(a_val) and np.isfinite(d_val)):
                continue

            pixel_str = None
            bbox = (row["X_min"], row["Y_min"], row["X_max"], row["Y_max"])

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
                        name=f"{row['Filename']}_{idx}",
                        valence=float(v_val),
                        arousal=float(a_val),
                        dominance=float(d_val),
                        pixels=pixel_str,
                    )
                )

        return samples



def main():
    parser = argparse.ArgumentParser(description="Preprocess continuous VAD emotion datasets.")
    parser.add_argument("--dataset", type=str, required=True, choices=["afew", "emotic", "all"], help="Dataset to preprocess")
    parser.add_argument("--data_dir", type=str, default=None, help="Root path to input dataset directory")
    parser.add_argument("--output_dir", type=str, default="./data", help="Path to output processed CSVs")
    parser.add_argument("--image_size", type=int, default=112, help="Output image size (width & height)")
    parser.add_argument("--include_extra", action="store_true", help="Include EMOTIC extra training data")
    args = parser.parse_args()

    size = (args.image_size, args.image_size)

    if args.dataset in ["afew", "all"]:
        afew_dir = args.data_dir if args.data_dir else "../Data/VA/AFEW-VA"
        afew_processor = PreprocessAFEW(data_dir=afew_dir, output_dir=args.output_dir, target_size=size)
        afew_processor.process()

    if args.dataset in ["emotic", "all"]:
        emotic_dir = args.data_dir if args.data_dir else "../Data/VAD/EMOTIC"
        emotic_processor = PreprocessEMOTIC(
            data_dir=emotic_dir, 
            output_dir=args.output_dir, 
            target_size=size, 
            include_extra=args.include_extra
        )
        emotic_processor.process()


if __name__ == "__main__":
    main()