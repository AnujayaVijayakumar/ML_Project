from __future__ import annotations

import os
import time
import pathlib
from collections import deque

import cv2
import numpy as np
import pandas as pd
from mediapipe.python.solutions import pose as mp_pose

from final_project.process_videos.helpers.data_to_csv import CSVDataWriter
from final_project.src.preprocess import preprocess_pipeline
from final_project.src.feature_selection import build_compact_features
from final_project.src.windowing import build_dataset, LABELS
from final_project.src.model_mlp import MLP

import json
import urllib.request
import urllib.error

SLIDESHOW_EVENT_URL = "http://127.0.0.1:8800/event"

# optional: reduce some backend log spam
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# -------------------------
# Live settings
# -------------------------
FPS = 30
WIN_S = 0.8
HOP_S = 0.1

CONF_THRESH = 0.65
STABLE_COUNT = 3
COOLDOWN_S = 1.0

SHOW_VIDEO = True
DRAW_LANDMARKS = True
FLIP_IMAGE = False   # set based on your webcam mirroring

# keep a bit more than one window
BUFFER_SECONDS = 1.5


def load_model_and_norm(base: pathlib.Path):
    model_path = base / "models" / "mlp_gesture.npz"
    norm_path = base / "models" / "mlp_norm.npz"

    model = MLP.load(str(model_path))

    norm = np.load(norm_path, allow_pickle=True)
    mean = norm["mean"]
    std = norm["std"]
    feature_cols = list(norm["feature_cols"])

    return model, mean, std, feature_cols


def build_live_dataframe(frame_list, timestamps, column_names) -> pd.DataFrame:
    df = pd.DataFrame(frame_list, columns=column_names, index=timestamps)
    df.index = pd.to_timedelta(df.index, unit="ms")
    df.index.name = "time"

    # required by your pipeline
    df["ground_truth"] = "idle_or_other"
    return df


def trim_buffer(df: pd.DataFrame, seconds: float) -> pd.DataFrame:
    if len(df) == 0:
        return df
    tmax = df.index.max()
    tmin = tmax - pd.to_timedelta(seconds, unit="s")
    return df[df.index >= tmin]


def decide_gesture_from_predictions(pred_idx: np.ndarray, probs: np.ndarray) -> tuple[str, float]:
    """
    Use latest window prediction.
    """
    idx = int(pred_idx[-1])
    conf = float(probs[-1, idx])
    return LABELS[idx], conf

def send_slideshow_command(cmd: str) -> None:
    payload = json.dumps({"command": cmd}).encode("utf-8")
    req = urllib.request.Request(
        SLIDESHOW_EVENT_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            _ = resp.read()
        print(f"[slideshow] sent: {cmd}")
    except urllib.error.URLError as e:
        print(f"[slideshow] send failed: {e}")

def main():
    base = pathlib.Path(__file__).resolve().parents[1]
    yaml_path = base / "process_videos" / "keypoint_mapping.yml"

    model, mean, std, feat_cols = load_model_and_norm(base)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    writer = CSVDataWriter(path=str(yaml_path))

    recent_labels = deque(maxlen=STABLE_COUNT)
    last_predict_time = 0.0
    last_trigger_time = 0.0

    print("Live gesture recognition started.")
    print("Press ESC to quit.\n")

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ok, image = cap.read()
            if not ok:
                break

            if FLIP_IMAGE:
                image = cv2.flip(image, 1)

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)

            # draw preview
            if SHOW_VIDEO:
                display_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if DRAW_LANDMARKS and results.pose_landmarks is not None:
                    import mediapipe as mp
                    mp_drawing = mp.solutions.drawing_utils
                    mp_drawing_styles = mp.solutions.drawing_styles
                    mp_drawing.draw_landmarks(
                        display_img,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )
                cv2.imshow("Live Gesture Recognition", display_img)

            # ESC quits
            if cv2.waitKey(1) & 0xFF == 27:
                break

            # collect landmarks
            if results.pose_landmarks is not None:
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                writer.read_data(results.pose_landmarks, timestamp_ms)

            now = time.time()
            if now - last_predict_time < HOP_S:
                continue
            last_predict_time = now

            # need enough recent frames
            min_frames = int(FPS * WIN_S)
            if len(writer.frame_list) < min_frames:
                continue

            try:
                # build dataframe from recent buffer
                df = build_live_dataframe(writer.frame_list, writer.timestamps, writer.column_names)
                df = trim_buffer(df, BUFFER_SECONDS)

                # pipeline must match training
                dfp = preprocess_pipeline(df, target_fps=FPS, do_trim=False)
                dfp = build_compact_features(dfp)

                X, _, _ = build_dataset(
                    [dfp],
                    fps=FPS,
                    win_s=WIN_S,
                    hop_s=HOP_S,
                    majority=0.6,
                    feature_cols=feat_cols,
                )

                if len(X) == 0:
                    continue

                # take latest window only
                x = X[-1:]
                x = (x - mean) / std

                pred_idx, probs = model.predict(x)
                pred_label, pred_conf = decide_gesture_from_predictions(pred_idx, probs)

                print(f"pred={pred_label:>12s}  conf={pred_conf:.3f}")

                recent_labels.append(pred_label)

                stable = (
                    len(recent_labels) == STABLE_COUNT
                    and all(lbl == pred_label for lbl in recent_labels)
                )

                if (
                    pred_label != "idle_or_other"
                    and pred_conf >= CONF_THRESH
                    and stable
                    and (now - last_trigger_time) >= COOLDOWN_S
                ):
                    print(f"\nTRIGGER >>> {pred_label}\n")
                    send_slideshow_command(pred_label)
                    last_trigger_time = now

            except Exception as e:
                print("prediction error:", e)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()