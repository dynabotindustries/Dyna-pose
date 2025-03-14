import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# Initialize BlazePose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

st.title("🔥 Real-Time Pose Estimation with BlazePose")

class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

webrtc_streamer(
    key="pose",
    video_processor_factory=PoseProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
