# final_project/process_videos/helpers/video_to_dataframe.py
import cv2
import pathlib
from mediapipe.python.solutions import pose as mp_pose
from final_project.process_videos.helpers import data_to_csv as dtc

def video_to_dataframe(video_filename: str, flip_image: bool = False):
    # Absolute path to keypoint_mapping.yml (robust: independent of cwd)
    base_dir = pathlib.Path(__file__).resolve().parents[1]  # .../final_project/process_videos
    yaml_path = base_dir / "keypoint_mapping.yml"

    cap = cv2.VideoCapture(video_filename)
    frames = dtc.CSVDataWriter(path=str(yaml_path))

    success = True
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened() and success:
            success, image = cap.read()
            if not success:
                break

            if flip_image:
                image = cv2.flip(image, 1)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            frames.read_data(
                data=results.pose_landmarks,
                timestamp=cap.get(cv2.CAP_PROP_POS_MSEC),
            )

    cap.release()
    return frames.get_frames()
