import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# Load MoveNet model
movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
model = movenet.signatures['serving_default']

# Function to detect pose
def detect_pose(image):
    img_resized = tf.image.resize_with_pad(image, 192, 192)
    img_resized = tf.cast(img_resized, dtype=tf.int32)
    img_resized = tf.expand_dims(img_resized, axis=0)

    outputs = model(input=img_resized)
    keypoints = outputs['output_0'].numpy()[0, 0, :, :]

    return keypoints

# Draw keypoints on image
def draw_keypoints(image, keypoints):
    h, w, _ = image.shape
    for kp in keypoints:
        y, x, confidence = kp
        if confidence > 0.3:  # Confidence threshold
            cx, cy = int(x * w), int(y * h)
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
    return image

# Custom Video Processor
class PoseVideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        keypoints = detect_pose(img_rgb)
        img = draw_keypoints(img, keypoints)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit UI
st.title("Pose Estimation Demo - DynabotIndustries")
st.write("Live pose estimation using MoveNet model")

webrtc_streamer(key="pose_estimation", video_processor_factory=PoseVideoProcessor)
