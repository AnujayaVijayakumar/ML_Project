from __future__ import annotations
import cv2
import mediapipe as mp
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Video


import os
import glob
import numpy as np
from dataclasses import dataclass


#from final_project.process_videos.helpers.video_to_dataframe import video_to_dataframe
#from process_videos.helpers.video_to_dataframe import video_to_dataframe

from final_project.process_videos.helpers.video_to_dataframe import video_to_dataframe

MANDATORY = {"rotate", "swipe_left", "swipe_right"}


def _normalize_labels(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

def load_elan_txt(annotation_path: str) -> pd.DataFrame:
    """
    Tries to auto-detect which columns contain:
      - start time in seconds
      - end time in seconds
      - label string
    Returns columns: start, end, label (normalized).
    """
    raw = pd.read_csv(annotation_path, sep="\t", header=None, dtype=str)

    # Helper: find columns that look numeric seconds
    def _is_seconds_col(col: pd.Series) -> bool:
        x = pd.to_numeric(col, errors="coerce")
        # must have enough numeric values and be non-negative
        return x.notna().mean() > 0.8 and (x.dropna() >= 0).all()

    # Candidate numeric columns
    numeric_cols = [i for i in raw.columns if _is_seconds_col(raw[i])]

    if len(numeric_cols) < 2:
        raise ValueError(f"Cannot find start/end second columns in {annotation_path}")

    # pick first two numeric columns as start/end
    start_col, end_col = numeric_cols[0], numeric_cols[1]

    # label column: pick the rightmost column that has non-numeric text
    label_col = None
    for i in reversed(raw.columns):
        # ignore numeric cols
        if i in (start_col, end_col):
            continue
        # must contain some non-empty strings
        if raw[i].fillna("").str.strip().ne("").any():
            label_col = i
            break

    if label_col is None:
        raise ValueError(f"Cannot find label column in {annotation_path}")

    ann = pd.DataFrame({
        "start": pd.to_numeric(raw[start_col], errors="coerce"),
        "end":   pd.to_numeric(raw[end_col], errors="coerce"),
        "label": raw[label_col],
    }).dropna(subset=["start", "end"])

    ann["start"] = pd.to_timedelta(ann["start"], unit="s")
    ann["end"]   = pd.to_timedelta(ann["end"], unit="s")

    ann["label"] = (
        ann["label"].astype(str)
        .str.strip().str.lower().str.replace(" ", "_", regex=False)
    )

    return ann


def create_and_annotate_dataframe_from_video(
    video_path: str,
    annotation_path: str,
    flip_image: bool = True,
):
    # 1) Extract frames
    frames = video_to_dataframe(video_path, flip_image=flip_image)
    frames.index = pd.to_timedelta(frames.index, unit="ms")
    frames.index.name = "time"

    # 2) Load ELAN (auto-detect columns)
    annotations = load_elan_txt(annotation_path)

    # 3) Assign ground truth
    frames["ground_truth"] = "idle"

    for _, ann in annotations.iterrows():
        mask = (frames.index >= ann["start"]) & (frames.index <= ann["end"])
        frames.loc[mask, "ground_truth"] = ann["label"]

    # 4) Normalize labels (important for M2 consistency)
    frames["ground_truth"] = (
        frames["ground_truth"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    return frames

def estimate_fps(frames: pd.DataFrame) -> float:
    if not isinstance(frames.index, pd.TimedeltaIndex) or len(frames) < 3:
        return np.nan
    dt = frames.index.to_series().diff().dropna().dt.total_seconds()
    med = dt.median()
    if med == 0 or pd.isna(med):
        return np.nan
    return 1.0 / med

def top_changing_features(df: pd.DataFrame, k=12):
    num = df.select_dtypes(include=[np.number])
    score = num.diff().abs().mean().sort_values(ascending=False)
    return score.head(k)

def infer_video_label(frames: pd.DataFrame) -> str:
    """Dominant non-idle label; otherwise 'unknown'."""
    if "ground_truth" not in frames.columns or len(frames) == 0:
        return "unknown"
    gt = _normalize_labels(frames["ground_truth"])
    non_idle = gt[gt != "idle"]
    if len(non_idle) == 0:
        return "unknown"
    return non_idle.mode().iloc[0]


def load_frames_csv(csv_path: str) -> pd.DataFrame:
    """
    Fast path: load frames from CSV saved earlier.
    Assumes CSV index is milliseconds (int-like).
    """
    df = pd.read_csv(csv_path, index_col=0)
    # convert ms index -> TimedeltaIndex
    df.index = pd.to_timedelta(df.index.astype(float), unit="ms")
    df.index.name = "time"

    if "ground_truth" in df.columns:
        df["ground_truth"] = _normalize_labels(df["ground_truth"])

    return df

@dataclass
class VideoSpec:
    video_path: str
    annotation_path: str
    flip_image: bool
    person_id: str

def load_dataset_from_videos(video_specs: list[VideoSpec]) -> list[pd.DataFrame]:
    """
    Builds list_of_frames from (video, annotation, flip, person_id) specs.
    ELAN format is auto-detected inside load_elan_txt().
    """
    out = []
    for spec in video_specs:
        df = create_and_annotate_dataframe_from_video(
            spec.video_path,
            spec.annotation_path,
            flip_image=spec.flip_image,
        )
        df["person_id"] = spec.person_id
        df["video_label"] = infer_video_label(df)
        out.append(df)
    return out


def load_dataset_from_csv_folder(csv_folder: str, pattern: str = "*_frames_with_ground_truth.csv") -> list[pd.DataFrame]:
    """
    Loads all annotated frames CSVs from a folder.
    """
    paths = sorted(glob.glob(os.path.join(csv_folder, pattern)))
    if not paths:
        raise FileNotFoundError(f"No CSVs found in {csv_folder} with pattern {pattern}")

    out = []
    for p in paths:
        df = load_frames_csv(p)
        # optional: infer metadata from filename if you want
        df["video_label"] = infer_video_label(df)

        # crude person_id inference: prefix before first underscore
        base = os.path.basename(p)
        person_guess = base.split("_")[0] if "_" in base else "unknown"
        df["person_id"] = person_guess

        out.append(df)
    return out