# src/windowing.py
from __future__ import annotations

import numpy as np
import pandas as pd


# Order matters: this is your class index mapping.
LABELS = ["idle_or_other", "rotate", "swipe_left", "swipe_right"]
label_to_idx = {l: i for i, l in enumerate(LABELS)}
idx_to_label = {i: l for l, i in label_to_idx.items()}

#already defined in data_loader()
def normalize_labels(s: pd.Series) -> pd.Series:
    """Normalize labels: strip, lowercase, spaces->underscores."""
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )


def pick_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Select numeric pose/features only.
    Exclude label/metadata columns.
    """
    drop = {"ground_truth", "person_id", "video_label"}
    cols = []
    for c in df.columns:
        if c in drop:
            continue
        if np.issubdtype(df[c].dtype, np.number):
            cols.append(c)
    return cols


def assign_window_label(gt_window: np.ndarray, majority: float = 0.6) -> int:
    """
    Convert a sequence of per-frame labels into one label for the whole window.
    Rule:
      - if rotate OR swipe_left OR swipe_right covers >= majority of frames -> that label
      - else -> idle_or_other
    Any other labels are treated as idle_or_other.
    """
    gts = pd.Series(gt_window)
    counts = gts.value_counts(normalize=True)

    for gesture in ("rotate", "swipe_left", "swipe_right"):
        if counts.get(gesture, 0.0) >= majority:
            return label_to_idx[gesture]

    return label_to_idx["idle_or_other"]


def make_windows(
    df: pd.DataFrame,
    fps: int = 30,
    win_s: float = 1.0,
    hop_s: float = 0.2,
    majority: float = 0.6,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Create windowed dataset from ONE preprocessed dataframe.
    Returns:
      X: (n_windows, win_frames * n_features)
      y: (n_windows,)
      feature_cols_used: list[str]

    Requirements:
      - df.index should be a TimedeltaIndex
      - df must contain 'ground_truth'
      - df should already be resampled to the same fps used here
    """
    if "ground_truth" not in df.columns:
        raise ValueError("Dataframe must contain a 'ground_truth' column.")

    if not isinstance(df.index, pd.TimedeltaIndex):
        raise TypeError("df.index must be a TimedeltaIndex (time-based index).")

    win = int(round(fps * win_s)) #30
    hop = int(round(fps * hop_s)) #6
    if win < 2:
        raise ValueError("win_s too small (window must be >= 2 frames).")
    if hop < 1:
        raise ValueError("hop_s too small (hop must be >= 1 frame).")

    # Normalize labels once
    gt = normalize_labels(df["ground_truth"]).to_numpy()

    # Pick feature columns if not provided
    if feature_cols is None:
        feature_cols = pick_feature_columns(df)

    # Extract features
    Xf = df[feature_cols].to_numpy(dtype=np.float64)

    X_windows = []
    y_windows = []

    # Sliding window
    for start in range(0, len(df) - win + 1, hop):
        end = start + win
        x_win = Xf[start:end]     # (win, n_features)
        gt_win = gt[start:end]    # (win,)

        y_label = assign_window_label(gt_win, majority=majority)

        X_windows.append(x_win.reshape(-1))  # flatten -> (win*n_features,)
        y_windows.append(y_label)

    if len(X_windows) == 0:
        return (
            np.empty((0, win * len(feature_cols)), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
            feature_cols
        )

    return np.vstack(X_windows), np.array(y_windows, dtype=np.int64), feature_cols


def build_dataset(
    dfs: list[pd.DataFrame],
    fps: int = 30,
    win_s: float = 1.0,
    hop_s: float = 0.2,
    majority: float = 0.6,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Create X,y by concatenating windows from multiple videos (dataframes).
    Enforces consistent feature columns across all dfs.

    Returns: X, y, feature_cols_used
    """
    if not dfs:
        raise ValueError("dfs is empty")

    X_all = []
    y_all = []
    cols_used = feature_cols

    for i, df in enumerate(dfs):
        X, y, cols = make_windows(
            df,
            fps=fps,
            win_s=win_s,
            hop_s=hop_s,
            majority=majority,
            feature_cols=cols_used
        )

        # lock feature columns after first df
        if cols_used is None:
            cols_used = cols
        else:
            if cols != cols_used:
                raise ValueError(
                    f"Feature columns mismatch at df index {i}. "
                    "Make sure preprocessing outputs identical columns."
                )

        if len(X) > 0:
            X_all.append(X)
            y_all.append(y)

    if not X_all:
        return (
            np.empty((0, int(round(fps * win_s)) * len(cols_used)), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
            cols_used
        )

    return np.vstack(X_all), np.concatenate(y_all), cols_used


def downsample_idle(
    X: np.ndarray,
    y: np.ndarray,
    idle_idx: int = 0,
    max_idle_ratio: float = 2.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reduce idle_or_other samples so the dataset isn't dominated by idle.
    Keeps at most max_idle_ratio * (#non-idle) idle samples.
    """
    rng = np.random.default_rng(seed)

    idle = np.where(y == idle_idx)[0]
    non  = np.where(y != idle_idx)[0]

    if len(non) == 0 or len(idle) == 0:
        return X, y

    max_idle = int(max_idle_ratio * len(non))
    if len(idle) <= max_idle:
        return X, y

    keep_idle = rng.choice(idle, size=max_idle, replace=False)
    keep = np.concatenate([non, keep_idle])
    rng.shuffle(keep)

    return X[keep], y[keep]