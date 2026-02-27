## MidTerm Submission Tasks

This repository contains the work for a small gesture recognition midterm project. The main tasks are:

### Task 1 – Pose extraction and labeling
- Extract pose landmarks from three gesture videos (`swipe_right`, `swipe_left`, `rotate`).
- Attach per-frame `ground_truth` labels together with `person_id` and `video_label`.

### Task 2 – Preprocessing and feature engineering
- Resample all recordings to a constant FPS.
- Center the pose around a reference joint, smooth jitter, and compute wrist velocity features.

### Task 3 – Exploratory analysis
- Plot landmark trajectories over time (especially wrists, elbows, shoulders).
- Inspect how different gestures appear in position and velocity space.

### Task 4 – Gesture classification
- Train simple baseline models to distinguish between the three gestures.
- Evaluate performance using accuracy and confusion matrices.