from __future__ import annotations
import numpy as np
import pandas as pd

def _vec_norm(x, y, z):
    return np.sqrt(x*x + y*y + z*z)

def build_compact_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: preprocessed dataframe (already resampled, smoothed, recentered).
    Output: dataframe with a compact set of meaningful features.
    Keeps ground_truth/person_id/video_label if present.
    """
    out = pd.DataFrame(index=df.index)

    # keep labels/metadata if present
    for c in ["ground_truth", "person_id", "video_label"]:
        if c in df.columns:
            out[c] = df[c]

    # joints we use
    joints = [
        "left_wrist", "right_wrist",
        "left_elbow", "right_elbow",
        "left_shoulder", "right_shoulder",
    ]

    # 1) relative positions (assumes you already recentered around hips)
    for j in joints:
        for a in ["x", "y", "z"]:
            col = f"{j}_{a}"
            if col in df.columns:
                out[col] = df[col]

    # 2) wrist velocities (you already add *_vel; if not, compute)
    for j in ["left_wrist", "right_wrist"]:
        for a in ["x", "y", "z"]:
            vcol = f"{j}_{a}_vel"
            pcol = f"{j}_{a}"
            if vcol in df.columns:
                out[vcol] = df[vcol]
            elif pcol in df.columns:
                # fallback: compute velocity from time index
                t = df.index.to_series().dt.total_seconds()
                dt = t.diff().replace(0, np.nan)
                out[vcol] = (df[pcol].diff() / dt).fillna(0.0)

    # 3) wrist-to-shoulder distances (arm extension)
    if all(c in df.columns for c in ["right_wrist_x","right_wrist_y","right_wrist_z","right_shoulder_x","right_shoulder_y","right_shoulder_z"]):
        dx = df["right_wrist_x"] - df["right_shoulder_x"]
        dy = df["right_wrist_y"] - df["right_shoulder_y"]
        dz = df["right_wrist_z"] - df["right_shoulder_z"]
        out["right_arm_len"] = _vec_norm(dx, dy, dz)

    if all(c in df.columns for c in ["left_wrist_x","left_wrist_y","left_wrist_z","left_shoulder_x","left_shoulder_y","left_shoulder_z"]):
        dx = df["left_wrist_x"] - df["left_shoulder_x"]
        dy = df["left_wrist_y"] - df["left_shoulder_y"]
        dz = df["left_wrist_z"] - df["left_shoulder_z"]
        out["left_arm_len"] = _vec_norm(dx, dy, dz)

    # 4) horizontal dominance ratios (swipes are horizontal)
    eps = 1e-6
    if "right_wrist_x_vel" in out.columns and "right_wrist_y_vel" in out.columns:
        out["right_horiz_ratio"] = np.abs(out["right_wrist_x_vel"]) / (np.abs(out["right_wrist_y_vel"]) + eps)

    if "left_wrist_x_vel" in out.columns and "left_wrist_y_vel" in out.columns:
        out["left_horiz_ratio"] = np.abs(out["left_wrist_x_vel"]) / (np.abs(out["left_wrist_y_vel"]) + eps)

    # 5) rotation proxy: acceleration magnitude (curved motion => changing velocity)
    # compute for right wrist only (rotation gesture uses right arm in your spec)
    if "right_wrist_x_vel" in out.columns and "right_wrist_y_vel" in out.columns and "right_wrist_z_vel" in out.columns:
        t = out.index.to_series().dt.total_seconds()
        dt = t.diff().replace(0, np.nan)

        ax = (out["right_wrist_x_vel"].diff() / dt).fillna(0.0)
        ay = (out["right_wrist_y_vel"].diff() / dt).fillna(0.0)
        az = (out["right_wrist_z_vel"].diff() / dt).fillna(0.0)
        out["right_wrist_acc"] = _vec_norm(ax, ay, az)

    # fill any remaining NaNs from diff/dt
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out