from pathlib import Path
from final_project.src.data_loader import VideoSpec, load_dataset_from_videos

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

    # IMPORTANT: pick the correct ELAN columns for YOUR export
    list_of_frames = load_dataset_from_videos(videos)
    for df in list_of_frames:
        print(df["ground_truth"].value_counts())

    print("Loaded:", len(list_of_frames))
    print([df["video_label"].iloc[0] for df in list_of_frames])
    print(list_of_frames[0].head())

if __name__ == "__main__":
    main()