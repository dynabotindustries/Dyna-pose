import os
import streamlit as st
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# ✅ Prevent OpenCV & Mediapipe from using GUI-based rendering
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["DISPLAY"] = ""  # Ensures OpenCV doesn't attempt to use a display server

import cv2  # Import AFTER setting environment variables
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

class PoseVideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit UI
st.title("Pose Estimation Demo - DynabotIndustries")
st.write("Real-time pose estimation using MediaPipe and Streamlit.")

# Start webcam
webrtc_streamer(key="pose", video_processor_factory=PoseVideoProcessor)
