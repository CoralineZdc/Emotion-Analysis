import json
from pathlib import Path
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import cv2
import mediapipe as mp


def crop_face_mediapipe(img_pil: Image.Image, face_detector, margin: float = 0.15) -> Image.Image:
    """Detects face using MediaPipe and returns cropped PIL Image."""
    img_rgb = np.array(img_pil.convert("RGB"))
    h_img, w_img, _ = img_rgb.shape

    results = face_detector.process(img_rgb)

    if not results.detections:
        return img_pil  # Fallback to full image if no face detected

    detection = max(results.detections, key=lambda d: d.score[0])
    bbox = detection.location_data.relative_bounding_box

    x_min = int(bbox.xmin * w_img)
    y_min = int(bbox.ymin * h_img)
    w = int(bbox.width * w_img)
    h = int(bbox.height * h_img)

    dw = int(w * margin)
    dh = int(h * margin)

    x1 = max(0, x_min - dw)
    y1 = max(0, y_min - dh)
    x2 = min(w_img, x_min + w + dw)
    y2 = min(h_img, y_min + h + dh)

    return img_pil.crop((x1, y1, x2, y2))


def preprocess_and_split_afew(
    dataset_dir: str = "../Data/VA/AFEW-VA",
    output_dir: str = "./data",
    target_size: tuple = (112, 112),
    min_confidence: float = 0.5,
    seed: int = 42,
) -> None:
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Collect and shuffle video folders
    video_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    print(f"Found {len(video_dirs)} total video folders in {dataset_path}")

    random.seed(seed)
    random.shuffle(video_dirs)

    # 2. Compute 60 / 20 / 20 folder split
    n_total = len(video_dirs)
    n_train = int(0.60 * n_total)
    n_val = int(0.20 * n_total)

    split_assignment = {}
    for idx, vdir in enumerate(video_dirs):
        if idx < n_train:
            split_assignment[vdir] = "train"
        elif idx < n_train + n_val:
            split_assignment[vdir] = "val"
        else:
            split_assignment[vdir] = "test"

    split_data = {"train": [], "val": [], "test": []}

    # 3. Process frames folder by folder
    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=min_confidence
    ) as face_detector:

        for video_dir in tqdm(video_dirs, desc="Cropping & processing frames"):
            target_split = split_assignment[video_dir]

            json_files = list(video_dir.glob("*.json"))
            if not json_files:
                continue

            with open(json_files[0], "r", encoding="utf-8") as f:
                meta = json.load(f)

            frames_data = meta.get("frames", {})

            for frame_id, frame_info in frames_data.items():
                # Scale from [-10, 10] to [-2, 2]
                valence = float(frame_info["valence"]) / 5.0
                arousal = float(frame_info["arousal"]) / 5.0

                img_path = video_dir / f"{frame_id}.png"
                if not img_path.exists():
                    img_path = video_dir / f"{frame_id}.jpg"
                    if not img_path.exists():
                        continue

                with Image.open(img_path) as img:
                    cropped_img = crop_face_mediapipe(img, face_detector)
                    img_gray = cropped_img.convert("L")
                    
                    if target_size is not None:
                        img_gray = img_gray.resize(target_size, Image.BILINEAR)

                    pixels = list(img_gray.getdata())
                    pixel_str = " ".join(map(str, pixels))

                split_data[target_split].append({
                    "Valence": valence,
                    "Arousal": arousal,
                    "pixels": pixel_str
                })
                
    # 4. Save CSV files for each split
    print("\n--- Summary ---")
    for split_name in ["train", "val", "test"]:
        rows = split_data[split_name]
        df = pd.DataFrame(rows)
        df.insert(0, "index", range(len(df)))
        
        file_path = output_path / f"{split_name}-afew.csv"
        df.to_csv(file_path, index=False)
        print(f"Saved {split_name:<5} split ({len(df):>6} frames) to: {file_path.resolve()}")


if __name__ == "__main__":
    preprocess_and_split_afew()