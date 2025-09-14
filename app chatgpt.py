import cv2
import numpy as np
import streamlit as st
from pathlib import Path
import tempfile
import time

st.set_page_config(page_title="Plate Blur Stream", layout="wide")

st.title("Vehicle Number-Plate Detection & Blur (Haar Cascade XML)")

# Sidebar controls
cascade_path = st.sidebar.text_input(
    "Cascade XML path",
    value="models/haarcascade_russian_plate_number.xml"
)
source = st.sidebar.selectbox("Video source", ["Webcam", "Upload video file", "Upload image"])

confidence_slider = st.sidebar.slider("Scale Factor (cascade)", 1.05, 1.5, 1.1, 0.01)
min_neighbors = st.sidebar.slider("Min Neighbors (cascade)", 1, 10, 3)
min_size_w = st.sidebar.number_input("Min plate width (px)", value=60, step=10)
min_size_h = st.sidebar.number_input("Min plate height (px)", value=20, step=5)
blur_strength = st.sidebar.slider("Blur kernel size (odd)", 5, 61, 31, 2)

# Validate cascade file
cascade_file = Path(cascade_path)
if not cascade_file.exists():
    st.error(f"Cascade file not found at: {cascade_path}")
    st.stop()

plate_cascade = cv2.CascadeClassifier(str(cascade_file))
if plate_cascade.empty():
    st.error("Failed to load cascade. File exists but not a valid cascade.")
    st.stop()

# File upload widgets
uploaded_video = None
uploaded_image = None
if source == "Upload video file":
    uploaded_video = st.file_uploader("Upload video (mp4, avi...)")
elif source == "Upload image":
    uploaded_image = st.file_uploader("Upload image (jpg, png...)")

# Utility: blur ROI
def blur_roi(frame, x, y, w, h, ksize):
    # Expand ROI slightly to ensure full plate covered
    pad_w = int(0.12 * w)
    pad_h = int(0.18 * h)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(frame.shape[1], x + w + pad_w)
    y2 = min(frame.shape[0], y + h + pad_h)
    roi = frame[y1:y2, x1:x2]
    # ensure kernel is odd
    k = ksize if ksize % 2 == 1 else ksize + 1
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    frame[y1:y2, x1:x2] = blurred
    return frame

# Processing function for a single frame (BGR input)
def process_frame(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(
        gray,
        scaleFactor=float(confidence_slider),
        minNeighbors=int(min_neighbors),
        minSize=(int(min_size_w), int(min_size_h))
    )
    # Blur each detected plate region
    for (x, y, w, h) in plates:
        try:
            frame_bgr = blur_roi(frame_bgr, x, y, w, h, int(blur_strength))
        except Exception:
            pass
        # optional: draw rectangle (comment out if you want pure blur)
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 1)
    return frame_bgr

# Display area
img_placeholder = st.empty()
status_text = st.empty()

# Helper to run capture loop and display frames
def run_video_capture(cap):
    fps_display = "N/A"
    prev = 0.0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (0,0), fx=1.0, fy=1.0)  # resize if needed
        processed = process_frame(frame)
        # Convert to RGB for Streamlit
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        img_placeholder.image(rgb, channels="RGB")
        # fps calc (simple)
        now = time.time()
        fps = 1.0 / (now - prev) if prev else 0.0
        prev = now
        status_text.text(f"FPS ~ {fps:.1f} — Press Stop in browser to end.")
        # small sleep so app remains responsive
        time.sleep(0.01)
    cap.release()
    status_text.text("Stream ended.")

# Branch by source
if source == "Webcam":
    use_device = 0  # default camera
    start = st.button("Start Webcam")
    stop = st.button("Stop")  # non-blocking; user should refresh or click stop to break
    if start:
        cap = cv2.VideoCapture(use_device)
        if not cap.isOpened():
            st.error("Unable to open webcam.")
        else:
            run_video_capture(cap)

elif source == "Upload video file":
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        tfile.flush()
        cap = cv2.VideoCapture(tfile.name)
        run_video_capture(cap)
    else:
        st.info("Upload a video file to start processing.")

else:  # image
    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        processed = process_frame(img)
        st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), channels="RGB")
    else:
        st.info("Upload an image with a visible number plate.")

st.markdown("---")
st.write("Notes: Haar cascades are fast but not perfect. If detection is poor, consider training a custom detector (YOLO/SSD) or pre-processing (increase contrast, better resolution).")
