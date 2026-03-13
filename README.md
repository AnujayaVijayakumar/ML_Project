# Gesture-Controlled Slideshow using Pose Estimation

## Table of Contents

- Overview
- System Architecture
- Project Structure
- Dataset
- Data Processing Pipeline
- Model Architecture
- Training
- Inference
- Live Gesture Recognition
- Slideshow Integration
- Running the Complete System
- Technologies Used
- Challenges
- Future Improvements

## Overview
This project implements a real-time gesture recognition system that allows a user to control a slideshow presentation using hand and body gestures captured by a webcam.

The system detects body keypoints using MediaPipe Pose, extracts motion-based features from the keypoints, and uses a machine learning model (MLP) to classify gestures. The predicted gestures are then used to control a Reveal.js slideshow in real time.

The implemented gestures allow the user to navigate slides and manipulate images during a presentation.

## System Architecture
```
Video Input (Dataset / Webcam)

        │
        ▼
Pose Estimation (MediaPipe)

        │
        ▼
Keypoint Extraction

        │
        ▼
Feature Engineering

        │
        ▼
Sliding Window Generation

        │
        ▼
MLP Gesture Classification Model

        │
        ▼
Live Gesture Prediction

        │
        ▼
Slideshow Control (Reveal.js)
```

## Project Structure

```
MLProject/
│
├── final_project/
│   ├── src/
│   │   ├── data_loader.py
│   │   ├── preprocess.py
│   │   ├── windowing.py
│   │   ├── model_mlp.py
│   │   ├── train.py
│   │   ├── infer.py
│   │   └── live_infer.py
│   │
│   ├── slideshow/
│   │   ├── slideshow_server.py
│   │   ├── control_slideshow_example.py
│   │   └── static/
│   │       ├── index.html
│   │       ├── slides.csv
│   │       ├── images/
│   │       └── js/
│   │
│   ├── data/
│   │   ├── raw_videos/
│   │   ├── annotations/
│   │   └── processed/
│   │
│   └── models/
│
└── README.md
```
## Datase

Dataset consist of videos and respective annotations for three gestures:
1. swipe left
2. swipe right
3. rotate

The annotation files are used to create the frame-level ground truth labels.

## Data Preprocessing

The video data is processed in several steps.

### 1. Pose Keypoint Extraction

Each video frame is processed with MediaPipe Pose, extracting coordinates for body joints such as:

- nose
- shoulders
- elbows
- wrists

These keypoints form the base features.

### 2. Feature Engineering / Selection

From the keypoints we derive motion-based features:

This include:

- wrist velocity

- wrist displacement

- relative joint distances

- temporal movement patterns

These features help distinguish gestures based on movement dynamics rather than static pose.

### 3. Sliding Window Generation

Gesture recognition is performed using temporal windows.

Parameters used:

- Window size: 1.0 seconds
- Hop size: 0.2 seconds
- FPS: 30

Each window becomes one training sample.
The label of the window is determined using a majority threshold:

majority = 0.6
, Meaning at least 60% of frames in the window must contain the gesture label.

## Model Architecture

A Multi-Layer Perceptron (MLP) is used for classification.

structure:

```
Input Layer
↓
Fully Connected Layer
↓
ReLU
↓
Dropout
↓
Fully Connected Layer
↓
Softmax 

```

Input size corresponds to the feature vector of a window.

Output classes are :

1. swipe_left
2. swipe_right
3. rotate
4. idle_or_other

## Training

Training is performed using: python -m final_project.src.train

Training pipeline:

1. Load processed video features

2. Generate sliding windows

3. Split dataset into:
  - Train set
- Validation set

4. Train MLP classifier

5. Save trained model inside models

## Offline Inference

To evaluate the gesture prediction on a test video: python -m final_project.src.infer

Input:
video file
annotation file (optional for evaluation)

The system outputs predicted gestures over time.

## Live Gesture Recognition

Real-time gesture detection using webcam: python -m final_project.src.live_infer

Pipeline:
```
Webcam frame
↓
MediaPipe Pose
↓
Feature extraction
↓
Sliding window buffer
↓
MLP prediction
↓
Gesture trigger
```
To prevent false triggers the system uses:

- confidence threshold

- prediction smoothing

- cooldown period

Sample output:

```
pred = swipe_right  conf = 0.93
TRIGGER >>> swipe_right  
```

## Slideshow Control

The slideshow is implemented using Reveal.js.
Start the server: python final_project/slideshow/slideshow_server.py

Open slideshow in browser: http://127.0.0.1:8800

Gesture predictions are sent as HTTP events:

POST /event
{
  "command": "swipe_right"
}

The browser receives the command and executes the corresponding action.

Sample mapping:

- swipe_right → next slide
- swipe_left → previous slide
- rotate → rotate image

## Challenges

Several challenges were encountered during development:

1. Real-time prediction stability

Gesture predictions can fluctuate frame-to-frame.
To address this we implemented:

- temporal smoothing
- prediction buffering

- trigger cooldown

2. Dataset variability

Different users perform gestures differently.
Adding multiple videos per class improved robustness.

## Possible Improvements

Future improvements could include:

1. more gesture classes

2. transformer-based temporal models

3. improved feature engineering

4. larger dataset

5. mobile deployment







