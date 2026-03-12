# final_project/src/train.py
from __future__ import annotations

import numpy as np
from pathlib import Path

from final_project.src.data_loader import VideoSpec, load_dataset_from_videos
from final_project.src.preprocess import preprocess_pipeline
from final_project.src.windowing import build_dataset, downsample_idle, LABELS
from final_project.src.feature_selection import build_compact_features
from final_project.src.model_mlp import MLP


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def train_loop(
    model: MLP,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    lr: float = 1e-3,
    epochs: int = 40,
    batch: int = 32,
    seed: int = 0,
    patience: int = 8,
):
    rng = np.random.default_rng(seed)

    best_val_acc = -1.0
    best_val_loss = np.inf
    best_epoch = 0
    wait = 0

    best_state = {
        "W1": model.W1.copy(),
        "b1": model.b1.copy(),
        "W2": model.W2.copy(),
        "b2": model.b2.copy(),
    }

    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(Xtr))
        Xs, ys = Xtr[idx], ytr[idx]

        for i in range(0, len(Xs), batch):
            xb = Xs[i:i + batch]
            yb = ys[i:i + batch]

            p, cache = model.forward(xb)
            grads = model.backward(cache, yb)
            model.step(grads, lr)

        pred_tr, ptr = model.predict(Xtr)
        pred_va, pva = model.predict(Xva)

        tr_acc = float((pred_tr == ytr).mean())
        va_acc = float((pred_va == yva).mean())
        tr_loss = model.loss(ytr, ptr)
        va_loss = model.loss(yva, pva)

        print(
            f"epoch {ep:02d} | "
            f"train_acc={tr_acc:.3f} | val_acc={va_acc:.3f} | "
            f"train_loss={tr_loss:.3f} | val_loss={va_loss:.3f}"
        )

        improved = False

        # primary criterion: better val accuracy
        if va_acc > best_val_acc:
            improved = True
        # tie-breaker: same val_acc but lower val_loss
        elif va_acc == best_val_acc and va_loss < best_val_loss:
            improved = True

        if improved:
            best_val_acc = va_acc
            best_val_loss = va_loss
            best_epoch = ep
            wait = 0

            best_state["W1"] = model.W1.copy()
            best_state["b1"] = model.b1.copy()
            best_state["W2"] = model.W2.copy()
            best_state["b2"] = model.b2.copy()
        else:
            wait += 1

        if wait >= patience:
            print(f"\nEarly stopping at epoch {ep}. Best epoch was {best_epoch}.")
            break

    # restore best weights
    model.W1 = best_state["W1"]
    model.b1 = best_state["b1"]
    model.W2 = best_state["W2"]
    model.b2 = best_state["b2"]

    print(
        f"\nRestored best model from epoch {best_epoch} "
        f"(val_acc={best_val_acc:.3f}, val_loss={best_val_loss:.3f})"
    )

    return model

def main():
    base = Path(__file__).resolve().parents[1]  # .../final_project
    debug_dir = base / "data" / "debug_frames"
    debug_dir.mkdir(exist_ok=True)


    train_videos = [
        VideoSpec(
            str(base / "data" / "ahad_data" / "ahad_swipe_right.mp4"),
            str(base / "data" / "ahad_data" / "ahad_swipe_right_annotation.txt"),
            True,
            "ahad",
        ),
        VideoSpec(
            str(base / "data" / "ahad_data" / "ahad_swipe_right_2.mp4"),
            str(base / "data" / "ahad_data" / "ahad_swipe_right_annotation_2.txt"),
            True,
            "ahad",
        ),
        VideoSpec(
            str(base / "data" / "shahzaib_data" / "shahzaib_swipe_right.mp4"),
            str(base / "data" / "shahzaib_data" / "shahzaib_swipe_right_annotation.txt"),
            True,
            "shahzaib",
        ),


        VideoSpec(
            str(base / "data" / "ahad_data" / "ahad_swipe_left.mp4"),
            str(base / "data" / "ahad_data" / "ahad_swipe_left_annotation.txt"),
            True,
            "ahad",
        ),
        VideoSpec(
            str(base / "data" / "shahzaib_data" / "shahzaib_swipe_left_2.mp4"),
            str(base / "data" / "shahzaib_data" / "shahzaib_swipe_left_annotation_2.txt"),
            True,
            "shahzaib",
        ),
        VideoSpec(
            str(base / "data" / "shahzaib_data" / "shahzaib_swipe_left.mp4"),
            str(base / "data" / "shahzaib_data" / "shahzaib_swipe_left_annotation.txt"),
            True,
            "shahzaib",
        ),

        VideoSpec(
            str(base / "data" / "shahzaib_data" / "shahzaib_rotate.mp4"),
            str(base / "data" / "shahzaib_data" / "shahzaib_rotate_annotation.txt"),
            False,
            "shahzaib",
        ),
        VideoSpec(
            str(base / "data" / "ahad_data" / "ahad_rotate.mp4"),
            str(base / "data" / "ahad_data" / "ahad_rotate_annotation.txt"),
            False,
            "ahad",
        ),

        VideoSpec(
            str(base / "data" / "ahad_data" / "ahad_rotate_2.mp4"),
            str(base / "data" / "ahad_data" / "ahad_rotate_annotation_2.txt"),
            False,
            "ahad",
        ),



    ]

    val_videos = [


        VideoSpec(
            str(base / "data" / "shahzaib_data" / "shahzaib_swipe_right_2.mp4"),
            str(base / "data" / "shahzaib_data" / "shahzaib_swipe_right_annotation_2.txt"),
            True,
            "shahzaib",
        ),

        VideoSpec(
            str(base / "data" / "ahad_data" / "ahad_swipe_left_2.mp4"),
            str(base / "data" / "ahad_data" / "ahad_swipe_left_annotation_2.txt"),
            True,
            "ahad",
        ),

        VideoSpec(
            str(base / "data" / "shahzaib_data" / "shahzaib_rotate_2.mp4"),
            str(base / "data" / "shahzaib_data" / "shahzaib_rotate_annotation_2.txt"),
            False,
            "shahzaib",
        ),
    ]

    # -------------------------
    # Load datasets
    # -------------------------
    train_frames = load_dataset_from_videos(train_videos)
    val_frames = load_dataset_from_videos(val_videos)

    # -------------------------
    # Preprocess + compact features
    # -------------------------
    train_processed = []

    for video_spec, frames in zip(train_videos, train_frames):
        df = preprocess_pipeline(frames, target_fps=30)
        df = build_compact_features(df)

        # ----------------------------
        # SAVE FRAME CSV FOR DEBUG
        # ----------------------------
        video_name = Path(video_spec.video_path).stem
        out_csv = debug_dir / f"{video_name}_frames.csv"
        df.to_csv(out_csv, index=False)

        train_processed.append(df)

    """train_processed = [
        build_compact_features(preprocess_pipeline(df, target_fps=30))
        for df in train_frames
    ]"""

    val_processed = []

    for video_spec, frames in zip(val_videos, val_frames):
        df = preprocess_pipeline(frames, target_fps=30)
        df = build_compact_features(df)

        video_name = Path(video_spec.video_path).stem
        out_csv = debug_dir / f"{video_name}_frames.csv"
        df.to_csv(out_csv, index=False)

        val_processed.append(df)

    """val_processed = [
        build_compact_features(preprocess_pipeline(df, target_fps=30))
        for df in val_frames
    ]"""

    # -------------------------
    # Windowing
    # -------------------------
    Xtr, ytr, feat_cols = build_dataset(
        train_processed,
        fps=30,
        win_s=0.8,
        hop_s=0.1,
        majority=0.6,
    )

    Xva, yva, _ = build_dataset(
        val_processed,
        fps=30,
        win_s=0.8,
        hop_s=0.1,
        majority=0.6,
        feature_cols=feat_cols,
    )

    print("raw train counts:", dict(zip(LABELS, np.bincount(ytr, minlength=len(LABELS)))))
    print("raw val counts:  ", dict(zip(LABELS, np.bincount(yva, minlength=len(LABELS)))))

    # -------------------------
    # Balance training set only
    # -------------------------
    Xtr, ytr = downsample_idle(Xtr, ytr, idle_idx=0, max_idle_ratio=2.0, seed=42)
    print("balanced train counts:", dict(zip(LABELS, np.bincount(ytr, minlength=len(LABELS)))))

    print("Xtr:", Xtr.shape, "Xva:", Xva.shape)

    # -------------------------
    # Standardize using TRAIN stats only
    # -------------------------
    mean = Xtr.mean(axis=0, keepdims=True)
    std = Xtr.std(axis=0, keepdims=True) + 1e-8

    Xtr = (Xtr - mean) / std
    Xva = (Xva - mean) / std

    # sanity checks
    print("NaN in Xtr:", np.isnan(Xtr).sum(), "Inf in Xtr:", np.isinf(Xtr).sum())
    print("NaN in Xva:", np.isnan(Xva).sum(), "Inf in Xva:", np.isinf(Xva).sum())

    # -------------------------
    # Train
    # -------------------------
    model = MLP(
        in_dim=Xtr.shape[1],
        hidden_dim=32,
        out_dim=len(LABELS),
        seed=0,
    )

    model = train_loop(
        model,
        Xtr,
        ytr,
        Xva,
        yva,
        lr=1e-3,
        epochs=40,
        batch=32,
        seed=0,
        patience=8,
    )
    # -------------------------
    # Final evaluation
    # -------------------------
    pred_va, _ = model.predict(Xva)
    cm = confusion_matrix(yva, pred_va, len(LABELS))
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)

    print("\nPer-class validation accuracy:")
    per_class_acc = cm.diagonal() / np.maximum(1, cm.sum(axis=1))
    for i, acc in enumerate(per_class_acc):
        print(f"{LABELS[i]:>12s}: {acc:.3f} (n={cm.sum(axis=1)[i]})")

    # -------------------------
    # Save model + normalization
    # -------------------------
    model_dir = base / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    norm_path = model_dir / "mlp_norm.npz"
    model_path = model_dir / "mlp_gesture.npz"

    np.savez(norm_path, mean=mean, std=std, feature_cols=np.array(feat_cols, dtype=object))
    model.save(str(model_path), meta={"feat_dim": Xtr.shape[1]})

    print(f"\nsaved model: {model_path}")
    print(f"saved norm : {norm_path}")


if __name__ == "__main__":
    main()