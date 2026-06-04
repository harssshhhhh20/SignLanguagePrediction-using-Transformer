import os
import json
import cv2
import numpy as np
import mediapipe as mp
import pickle
from tqdm import tqdm

VIDEO_DIR = "videos"
OUTPUT_DIR = "datasetv2"

NUM_CLASSES = 20
SEQUENCE_LENGTH = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open("WLASL_v0.3.json", "r") as f:
    data = json.load(f)

selected_classes = [item["gloss"] for item in data[:NUM_CLASSES]]

label_map = {
    label: idx
    for idx, label in enumerate(selected_classes)
}

with open("label_map.pkl", "wb") as f:
    pickle.dump(label_map, f)

video_to_label = {}

for sign in data:
    if sign["gloss"] not in selected_classes:
        continue

    for instance in sign["instances"]:
        video_to_label[instance["video_id"]] = sign["gloss"]

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

for label in selected_classes:
    os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)

video_files = [
    f for f in os.listdir(VIDEO_DIR)
    if f.endswith(".mp4")
]

saved_count = 0
skipped_count = 0

matched = 0
not_found = 0
too_short = 0
saved = 0

for video_file in tqdm(video_files):

    video_id = video_file.replace(".mp4", "")

    if video_id not in video_to_label:
        not_found += 1
        continue

    matched += 1

    label = video_to_label[video_id]

    cap = cv2.VideoCapture(
        os.path.join(VIDEO_DIR, video_file)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames < 10:
        too_short += 1
        cap.release()
        continue

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        SEQUENCE_LENGTH,
        dtype=int
    )

    sequence = []

    current_frame = 0
    target_idx = 0

    while cap.isOpened() and target_idx < SEQUENCE_LENGTH:

        ret, frame = cap.read()

        if not ret:
            break

        if current_frame == frame_indices[target_idx]:

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            pose_results = pose.process(frame_rgb)
            hand_results = hands.process(frame_rgb)

            features = np.zeros(258)

            if pose_results.pose_landmarks:

                for lm_idx, lm in enumerate(
                    pose_results.pose_landmarks.landmark
                ):

                    base = lm_idx * 4

                    features[base] = lm.x
                    features[base + 1] = lm.y
                    features[base + 2] = lm.z
                    features[base + 3] = lm.visibility

            if hand_results.multi_hand_landmarks:

                for hand_idx, hand_landmarks in enumerate(
                    hand_results.multi_hand_landmarks[:2]
                ):

                    offset = 132 + hand_idx * 63

                    for lm_idx, lm in enumerate(
                        hand_landmarks.landmark
                    ):

                        base = offset + lm_idx * 3

                        features[base] = lm.x
                        features[base + 1] = lm.y
                        features[base + 2] = lm.z

            sequence.append(features)

            target_idx += 1

        current_frame += 1

    cap.release()

    while len(sequence) < SEQUENCE_LENGTH:
        sequence.append(np.zeros(258))

    sequence = np.array(
        sequence,
        dtype=np.float32
    )

    non_zero_ratio = (
        np.count_nonzero(sequence)
        / sequence.size
    )

    if non_zero_ratio < 0.01:
        skipped_count += 1
        continue

    mean = np.mean(sequence, axis=0)
    std = np.std(sequence, axis=0)

    sequence = (
        sequence - mean
    ) / (std + 1e-6)

    save_path = os.path.join(
        OUTPUT_DIR,
        label,
        f"{video_id}.npy"
    )

    np.save(save_path, sequence)

    saved += 1
    saved_count += 1

hands.close()
pose.close()

print("Dataset creation complete")
print("Classes:", len(selected_classes))

print("\nDiagnostic Report")
print("Matched Videos:", matched)
print("Saved Videos:", saved)
print("Too Short:", too_short)
print("Not Found:", not_found)
print("Skipped:", skipped_count)