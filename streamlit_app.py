import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import tensorflow as tf
import numpy as np
import mediapipe as mp
import av
import cv2

# Load trained LSTM model
model = tf.keras.models.load_model('modelLSTM.h5', compile=False)
st.success("✅ Model loaded successfully")
actions = np.array(['Ada', 'Ketupat', 'Nasi', 'Himpit', 'Duit', 'Sampul'])

# Mediapipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Function to extract keypoints
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

# Custom video processor class
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.sequence = []
        self.sentence = []
        self.threshold = 0.5
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        
        # Mediapipe detection
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image_rgb)

        # Prediction logic
        keypoints = extract_keypoints(results)
        print("Keypoint shape:", keypoints.shape)
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-30:]
        print("Sequence length:", len(self.sequence))

        if len(self.sequence) == 30:
            res = model.predict(np.expand_dims(self.sequence, axis=0))[0]
            print("Prediction:", res)
            if np.max(res) > self.threshold:
                predicted_action = actions[np.argmax(res)]
                if len(self.sentence) == 0 or predicted_action != self.sentence[-1]:
                    self.sentence.append(predicted_action)
                self.sentence = self.sentence[-1:]

        # Display predicted word
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(self.sentence), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# Streamlit UI
st.title("MSL Recognition System")
st.write("Real-time Malaysian Sign Language detection with Mediapipe + LSTM")

webrtc_streamer(
    key="msl",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=SignLanguageProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
