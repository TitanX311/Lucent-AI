import streamlit as st
import ultralytics
import cv2
from PIL import Image
import numpy as np

# Load trained model
model = ultralytics.YOLO("runs/detect/yolov8m_custom-2/weights/best.pt")

st.set_page_config(layout="wide")
st.title("Lucent Interface")

# Sidebar options
# input_type = st.sidebar.radio("Select Input Type:", ["Image Upload", "Webcam"])

# if input_type == "Image Upload":
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    # st.image(image, caption="Uploaded Image")
    
    # Run inference
    results = model.predict(image, conf=0.5)
    
    # Display results with bounding boxes
    annotated_frame = results[0].plot()
    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image")
    with col2:
        st.image(annotated_frame_rgb, caption="Detections")
    
    # Show details
    st.write(f"Objects detected: {len(results[0].boxes)}")
    print(type(results[0].boxes))
    for box in results[0].boxes:
        st.write(f"Class: {box.cls[0]}, Confidence: {box.conf[0]:.2f}")

# else:  # Webcam
#     st.write("Webcam Feed:")
#     # Webcam implementation with cv2