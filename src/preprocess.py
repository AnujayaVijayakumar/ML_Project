# src/preprocess.py
from __future__ import annotations

import numpy as np
import pandas as pd


# -------------------------
# Utilities
# -------------------------
def normalize_ground_truth(frames: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize ground_truth labels:
    - strip whitespace
    - lowercase
    - replace spaces with underscores
    """
    out = frames.copy()
    if "ground_truth" in out.columns:
        out["ground_truth"] = (
            out["ground_truth"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
        )
    return out


def ensure_timedelta_index(frames: pd.DataFrame) -> None:
    if not isinstance(frames.index, pd.TimedeltaIndex):
        raise TypeError(
            "frames.index must be a pandas TimedeltaIndex. "
            "Convert with: frames.index = pd.to_timedelta(frames.index, unit='ms')"
        )


def numeric_feature_columns(frames: pd.DataFrame) -> list[str]:
    """
    Numeric columns only (pose features), excluding metadata/labels.
    """
    drop = {"ground_truth", "person_id", "video_label"}
    cols = []
    for c in frames.columns:
        if c in drop:
            continue
        if np.issubdtype(frames[c].dtype, np.number):
            cols.append(c)
    return cols


# -------------------------
# Preprocessing steps
# -------------------------
def resample_data(frames: pd.DataFrame, target_fps: int = 30) -> pd.DataFrame:
    """
    Resample to a fixed FPS using mean aggregation for numeric features,
    and forward-fill for categorical columns like ground_truth.

    Requires TimedeltaIndex.
    """
    ensure_timedelta_index(frames)

    step_ms = int(round(1000 / target_fps))
    rule = f"{step_ms}ms"

    out = frames.copy()

    # numeric -> mean
    num_cols = out.select_dtypes(include=[np.number]).columns
    numeric = out[num_cols].resample(rule).mean()

    # rebuild output with numeric first
    result = numeric

    # ground_truth -> ffill
    if "ground_truth" in out.columns:
        result["ground_truth"] = out["ground_truth"].resample(rule).ffill()

    # keep metadata as constant columns if present
    for meta in ["person_id", "video_label"]:
        if meta in out.columns:
            result[meta] = out[meta].iloc[0]

    return result


def mask_low_confidence(frames: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
    """
    For each joint, if confidence < threshold, set x/y/z to NaN.
    Then interpolate in time and fill edges.
    """
    ensure_timedelta_index(frames)

    out = frames.copy()
    conf_cols = [c for c in out.columns if c.endswith("_confidence") and np.issubdtype(out[c].dtype, np.number)]

    for conf in conf_cols:
        base = conf.replace("_confidence", "")
        coords = [f"{base}_{a}" for a in ("x", "y", "z") if f"{base}_{a}" in out.columns]
        if not coords:
            continue
        bad = out[conf] < threshold
        out.loc[bad, coords] = np.nan

    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].interpolate(method="time").ffill().bfill()
    return out


def smooth(frames: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Rolling median smoothing for numeric columns to reduce jitter.
    """
    ensure_timedelta_index(frames)

    out = frames.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].rolling(window=window, center=True, min_periods=1).median()
    return out


def recenter(frames: pd.DataFrame, left: str = "left_hip", right: str = "right_hip") -> pd.DataFrame:
    """
    Make coordinates relative by subtracting the mid-hip center for each axis.
    This helps remove camera position differences.
    """
    out = frames.copy()

    for axis in ("x", "y", "z"):
        l, r = f"{left}_{axis}", f"{right}_{axis}"
        if l not in out.columns or r not in out.columns:
            continue

        center = (out[l] + out[r]) / 2.0

        axis_cols = [
            c for c in out.columns
            if c.endswith(f"_{axis}") and not c.endswith("_confidence") and np.issubdtype(out[c].dtype, np.number)
        ]
        out[axis_cols] = out[axis_cols].sub(center, axis=0)

    return out


def add_velocity(frames: pd.DataFrame, joints: tuple[str, ...] = ("left_wrist", "right_wrist")) -> pd.DataFrame:
    """
    Add per-axis velocity features for selected joints: d(position)/dt.
    """
    ensure_timedelta_index(frames)

    out = frames.copy()

    t = out.index.to_series().dt.total_seconds()
    dt = t.diff()
    dt = dt.replace(0, np.nan)

    for j in joints:
        for axis in ("x", "y", "z"):
            col = f"{j}_{axis}"
            if col not in out.columns:
                continue
            if not np.issubdtype(out[col].dtype, np.number):
                continue

            out[f"{col}_vel"] = (out[col].diff() / dt).fillna(0.0)

    return out


def trim_to_gestures(frames: pd.DataFrame, margin_s: float = 0.5) -> pd.DataFrame:
    """
    Keep only a time range around gesture frames to reduce idle dominance.
    If no gestures exist, return unchanged.
    """
    if "ground_truth" not in frames.columns:
        return frames

    gt = frames["ground_truth"].astype(str)
    non_idle = gt != "idle"
    if non_idle.sum() == 0:
        return frames

    t0 = frames.index[non_idle].min() - pd.to_timedelta(margin_s, unit="s")
    t1 = frames.index[non_idle].max() + pd.to_timedelta(margin_s, unit="s")

    t0 = max(t0, frames.index.min())
    t1 = min(t1, frames.index.max())

    return frames.loc[(frames.index >= t0) & (frames.index <= t1)]


# -------------------------
# Pipeline
# -------------------------
def preprocess_pipeline(
    frames: pd.DataFrame,
    target_fps: int = 30,
    conf_threshold: float = 0.6,
    smooth_window: int = 7,
    do_trim: bool = True,
    trim_margin_s: float = 0.5,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline (whitelist-only):
    1) normalize labels
    2) resample to fixed FPS
    3) mask low confidence + interpolate
    4) smooth jitter
    5) recenter around hips
    6) add velocity features
    7) optionally trim around gestures
    """
    x = normalize_ground_truth(frames)
    x = resample_data(x, target_fps=target_fps)
    x = normalize_ground_truth(x)  # resampling may ffill strings; keep consistent

    x = mask_low_confidence(x, threshold=conf_threshold)
    x = smooth(x, window=smooth_window)
    x = recenter(x)
    x = add_velocity(x)

    if do_trim:
        x = trim_to_gestures(x, margin_s=trim_margin_s)

    return x