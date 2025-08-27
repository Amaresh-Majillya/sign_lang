import os
import json
import cv2
import numpy as np

DATASET_PATH = "dataset"  # Folder where videos + json are stored
VIDEOS_PATH = os.path.join(DATASET_PATH, "videos")
ANNOTATION_FILE = os.path.join(DATASET_PATH, "WLASL_v0.3.json")  # From Kaggle
IMG_SIZE = (64, 64)  # Resize to 64x64
NUM_FEATURES = IMG_SIZE[0] * IMG_SIZE[1] * 3
FRAMES_PER_VIDEO = 1  # Take 1 frame per video for quick training (change to more for accuracy)

def preprocess_wlasl():
    # Load annotation file
    with open(ANNOTATION_FILE, "r") as f:
        annotations = json.load(f)

    X, y = [], []
    labels_set = set()

    # First pass: collect all labels
    for ann in annotations:
        labels_set.add(ann["gloss"])  # "gloss" is the sign word

    labels_list = sorted(list(labels_set))
    label_to_idx = {label: idx for idx, label in enumerate(labels_list)}

    # Process each video
    for ann in annotations:
        label = ann["gloss"]
        label_idx = label_to_idx[label]

        for instance in ann["instances"]:
            video_id = instance["video_id"]
            video_path = os.path.join(VIDEOS_PATH, f"{video_id}.mp4")

            if not os.path.exists(video_path):
                continue

            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            while frame_count < FRAMES_PER_VIDEO:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, IMG_SIZE)
                frame = frame / 255.0  # Normalize
                X.append(frame)
                y.append(label_idx)
                frame_count += 1
            cap.release()

    # Save as numpy arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    np.save("X_train.npy", X)
    np.save("y_train.npy", y)
    np.save("labels.npy", np.array(labels_list))

    print(f"✅ Preprocessing complete: {X.shape[0]} samples saved")

if __name__ == "__main__":
    preprocess_wlasl()

'''import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

mp_hands = mp.solutions.hands
NUM_LANDMARKS = 21
NUM_FEATURES = NUM_LANDMARKS * 3

def extract_keypoints_from_image(image):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            keypoints = []
            for lm in results.multi_hand_landmarks[0].landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
            return keypoints
    return None

def build_dataset(dataset_dir="dataset", output_file="dataset_keypoints.npz"):
    X, y = [], []
    dataset_path = Path(dataset_dir)
    labels = sorted([p.name for p in dataset_path.iterdir() if p.is_dir()])
    for label in labels:
        for img_file in (dataset_path / label).glob("*"):
            img = cv2.imread(str(img_file))
            keypoints = extract_keypoints_from_image(img)
            if keypoints:
                X.append(keypoints)
                y.append(label)
    np.savez_compressed(output_file, X=np.array(X), y=np.array(y))
    print(f"Dataset saved to {output_file} with {len(X)} samples.")'''