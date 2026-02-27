# src/test_loader.py
from final_project.src.data_loader import VideoSpec, load_dataset_from_videos

def main():
    videos = [
        VideoSpec("real_data/video_rotate.mp4",      "real_data/video_rotate_annotation.txt",      True, "anu"),
        VideoSpec("real_data/video_swipe_left.mp4",  "real_data/video_swipe_left_annotation.txt",  True, "anu"),
        VideoSpec("real_data/video_swipe_right.mp4", "real_data/video_swipe_right_annotation.txt", True, "anu"),
    ]

    list_of_frames = load_dataset_from_videos(videos, elan_usecols=(3, 5, 8))
    print("Loaded:", len(list_of_frames))
    print([df["video_label"].iloc[0] for df in list_of_frames])
    print(list_of_frames[0].head())

if __name__ == "__main__":
    main()