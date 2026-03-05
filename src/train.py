# final_project/src/train.py
from __future__ import annotations
import numpy as np
from pathlib import Path

from final_project.src.data_loader import VideoSpec, load_dataset_from_videos
from final_project.src.preprocess import preprocess_pipeline
from final_project.src.windowing import build_dataset, downsample_idle, LABELS
from final_project.src.feature_selection import build_compact_features

from final_project.src.model_mlp import MLP


def train_loop(model: MLP, X, y, Xv, yv, lr=1e-3, epochs=30, batch=64, seed=0):
    rng = np.random.default_rng(seed)

    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(X))
        Xs, ys = X[idx], y[idx]

        for i in range(0, len(Xs), batch):
            xb = Xs[i:i + batch]
            yb = ys[i:i + batch]
            p, cache = model.forward(xb)
            grads = model.backward(cache, yb)
            model.step(grads, lr)

        pred_tr, _ = model.predict(X)
        pred_va, pva = model.predict(Xv)

        tr_acc = float((pred_tr == y).mean())
        va_acc = float((pred_va == yv).mean())
        va_loss = model.loss(yv, pva)

        print(f"epoch {ep:02d} | train_acc={tr_acc:.3f} | val_acc={va_acc:.3f} | val_loss={va_loss:.3f}")

    return model


def main():
    BASE = Path(__file__).resolve().parents[1]  # .../final_project

    videos = [
        VideoSpec(
            str(BASE / "data" / "ahad_data" / "ahad_swipe_right.mp4"),
            str(BASE / "data" / "ahad_data" / "ahad_swipe_right_annotation.txt"),
            True,
            "ahad",
        ),
        VideoSpec(
            str(BASE / "data" / "anujaya_data" / "anu_swipe_left.mp4"),
            str(BASE / "data" / "anujaya_data" / "anu_swipe_left_annotation.txt"),
            True,
            "anu",
        ),
        VideoSpec(
            str(BASE / "data" / "shahzaib_data" / "shahzaib_rotate.mp4"),
            str(BASE / "data" / "shahzaib_data" / "shahzaib_rotate_annotation.txt"),
            True,
            "shahzaib",
        ),
    ]
    frames_list = load_dataset_from_videos(videos)
    #processed = [preprocess_pipeline(df, target_fps=30) for df in frames_list]
    processed = [preprocess_pipeline(df, target_fps=30) for df in frames_list]
    processed = [build_compact_features(df) for df in processed]

    X, y, feat_cols = build_dataset(processed, fps=30, win_s=1.0, hop_s=0.2, majority=0.6)
    print("raw counts:", dict(zip(LABELS, np.bincount(y, minlength=len(LABELS)))))

    X, y = downsample_idle(X, y, idle_idx=0, max_idle_ratio=2.0, seed=42)
    print("balanced counts:", dict(zip(LABELS, np.bincount(y, minlength=len(LABELS)))))
    print("X:", X.shape)

    # Standardize features (train-set statistics only)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    X = (X - mean) / std

    # train/val split
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    n_train = int(0.8 * len(X))
    tr, va = idx[:n_train], idx[n_train:]

    Xtr, ytr = X[tr], y[tr]
    Xva, yva = X[va], y[va]

    model = MLP(in_dim=X.shape[1], hidden_dim=128, out_dim=len(LABELS), seed=0)
    model = train_loop(model, Xtr, ytr, Xva, yva, lr=1e-3, epochs=40, batch=64)

    np.savez("final_project/models/mlp_norm.npz", mean=mean, std=std)
    model.save("final_project/models/mlp_gesture.npz", meta={"feat_dim": X.shape[1]})
    print("saved: final_project/models/mlp_gesture.npz")


if __name__ == "__main__":
    main()