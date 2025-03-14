import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

class PoseVideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Convert image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        # Draw pose landmarks if detected
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit UI
st.title("Pose Estimation Demo - DynabotIndustries")
st.write("Real-time pose estimation using MediaPipe and Streamlit.")

# Start webcam
webrtc_streamer(key="pose", video_processor_factory=PoseVideoProcessor)

