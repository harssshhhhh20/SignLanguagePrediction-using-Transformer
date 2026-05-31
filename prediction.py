import cv2
import pickle
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque

MODEL_PATH = "transformer_model_v2_20acc.keras"
LABEL_MAP_PATH = "label_map.pkl"

SEQUENCE_LENGTH = 30

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_MAP_PATH, "rb") as f:
    label_map = pickle.load(f)

idx_to_label = {
    v: k for k, v in label_map.items()
}

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

sequence = deque(maxlen=SEQUENCE_LENGTH)

cap = cv2.VideoCapture(0)

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    hand_results = hands.process(rgb)
    pose_results = pose.process(rgb)

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

    prediction_text = "Collecting Frames..."

    if len(sequence) == SEQUENCE_LENGTH:

        input_data = np.expand_dims(
            np.array(sequence, dtype=np.float32),
            axis=0
        )

        prediction = model.predict(
            input_data,
            verbose=0
        )[0]

        class_id = np.argmax(prediction)

        confidence = prediction[class_id]

        prediction_text = (
            f"{idx_to_label[class_id]} "
            f"({confidence:.2f})"
        )

    cv2.putText(
        frame,
        prediction_text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Sign Language Transformer",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()

cv2.destroyAllWindows()

hands.close()
pose.close()