from __future__ import annotations

import numpy as np
from pathlib import Path

from final_project.src.data_loader import VideoSpec, load_dataset_from_videos
from final_project.src.preprocess import preprocess_pipeline
from final_project.src.feature_selection import build_compact_features
from final_project.src.windowing import build_dataset, LABELS
from final_project.src.model_mlp import MLP


def main():

    base = Path(__file__).resolve().parents[1]

    model_path = base / "models" / "mlp_gesture.npz"
    norm_path = base / "models" / "mlp_norm.npz"

    # -------------------------
    # Load normalization
    # -------------------------
    norm = np.load(norm_path, allow_pickle=True)

    mean = norm["mean"]
    std = norm["std"]
    feat_cols = list(norm["feature_cols"])

    # -------------------------
    # Load model
    # -------------------------
    model = MLP.load(str(model_path))

    print("Model loaded.")
    print("Feature dimension:", len(feat_cols))

    # -------------------------
    # Video to test
    # -------------------------
    test_video = VideoSpec(
            str(base / "data" / "anujaya_data" / "anu_swipe_left_3.mp4"),
            str(base / "data" / "anujaya_data" / "anu_swipe_left_annotation_3.txt"),
            False,
            "anu",
        )
    # -------------------------
    # Load frames
    # -------------------------
    frames = load_dataset_from_videos([test_video])[0]

    # -------------------------
    # Preprocess
    # -------------------------
    df = preprocess_pipeline(frames, target_fps=30)

    # -------------------------
    # Feature extraction
    # -------------------------
    df = build_compact_features(df)

    # -------------------------
    # Windowing
    # -------------------------
    X, y, _ = build_dataset(
        [df],
        fps=30,
        win_s=0.8,
        hop_s=0.1,
        majority=0.6,
        feature_cols=feat_cols,
    )

    print("Windows:", X.shape)

    # -------------------------
    # Normalize
    # -------------------------
    X = (X - mean) / std

    # -------------------------
    # Predict
    # -------------------------
    pred, prob = model.predict(X)

    labels = [LABELS[p] for p in pred]

    # -------------------------
    # Print results
    # -------------------------
    print("\nWindow predictions:\n")

    for i, lab in enumerate(labels[:30]):  # print first 30 windows
        print(f"{i:03d}  {lab}")


    # count predictions
    counts = np.bincount(pred, minlength=len(LABELS))

    # ignore idle class (index 0)
    gesture_counts = counts.copy()
    gesture_counts[0] = 0

    if gesture_counts.max() == 0:
        gesture = "idle_or_other"
    else:
        gesture = LABELS[gesture_counts.argmax()]

    print("\nPredicted gesture for video:", gesture)


if __name__ == "__main__":
    main()