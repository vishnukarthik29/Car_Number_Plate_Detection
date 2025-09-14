import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
import urllib.request
import os
import tempfile

# URLs for pre-trained Haar cascade XML files
CASCADE_URLS = {
    "russian": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_russian_plate_number.xml",
    "general": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_licence_plate_rus_16stages.xml",
    "alternative": "https://raw.githubusercontent.com/zeusees/HyperLPR/master/cascade.xml"
}

@st.cache_resource
def download_cascade_files():
    """
    Download and cache Haar cascade XML files for license plate detection
    """
    cascade_files = {}
    
    for name, url in CASCADE_URLS.items():
        try:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xml')
            
            # Download the cascade file
            with st.spinner(f"Downloading {name} cascade classifier..."):
                urllib.request.urlretrieve(url, temp_file.name)
            
            # Test if the file is valid
            test_cascade = cv2.CascadeClassifier(temp_file.name)
            if not test_cascade.empty():
                cascade_files[name] = temp_file.name
                st.success(f"✅ {name.title()} cascade loaded successfully")
            else:
                st.warning(f"⚠️ {name.title()} cascade file is invalid")
                os.unlink(temp_file.name)
                
        except Exception as e:
            st.error(f"❌ Failed to download {name} cascade: {str(e)}")
            if 'temp_file' in locals():
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
    
    return cascade_files

def detect_with_xml_cascade(image, cascade_path, scale_factor=1.1, min_neighbors=3, min_size=(30, 30)):
    """
    Detect license plates using XML Haar cascade classifier
    """
    # Load the cascade classifier
    cascade = cv2.CascadeClassifier(cascade_path)
    
    if cascade.empty():
        return []
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization for better detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Detect license plates
    plates = cascade.detectMultiScale(
        enhanced,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    return plates

def refine_detection_bounds(image, x, y, w, h):
    """
    Refine the detected bounds to focus on the actual text area
    """
    # Extract the region
    roi = image[y:y+h, x:x+w]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to isolate text
    _, thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Also try inverse threshold
    _, thresh_inv = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Choose the threshold that gives more text-like regions
    if np.sum(thresh == 255) > np.sum(thresh_inv == 255):
        final_thresh = thresh
    else:
        final_thresh = thresh_inv
    
    # Find contours of text regions
    contours, _ = cv2.findContours(final_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return x, y, w, h
    
    # Find bounding box of all significant text regions
    text_regions = []
    for contour in contours:
        cx, cy, cw, ch = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # Filter for character-like regions
        if (area > 50 and cw > 5 and ch > 8 and 
            cw < w * 0.4 and ch < h * 0.9):
            text_regions.append((cx, cy, cx + cw, cy + ch))
    
    if not text_regions:
        return x, y, w, h
    
    # Calculate overall text bounding box
    min_x = min(region[0] for region in text_regions)
    min_y = min(region[1] for region in text_regions)
    max_x = max(region[2] for region in text_regions)
    max_y = max(region[3] for region in text_regions)
    
    # Add small padding
    padding_x = max(5, (max_x - min_x) * 0.15)
    padding_y = max(3, (max_y - min_y) * 0.2)
    
    # Calculate refined coordinates
    refined_x = max(0, x + min_x - padding_x)
    refined_y = max(0, y + min_y - padding_y)
    refined_w = min(w, max_x - min_x + 2 * padding_x)
    refined_h = min(h, max_y - min_y + 2 * padding_y)
    
    return int(refined_x), int(refined_y), int(refined_w), int(refined_h)

def detect_and_blur_plates_xml(image, cascade_files, refine_bounds=True, blur_intensity=5):
    """
    Detect and blur license plates using XML cascade classifiers
    """
    # Convert PIL image to OpenCV format
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    result_image = opencv_image.copy()
    
    all_detections = []
    detection_info = []
    
    # Try each available cascade classifier
    for cascade_name, cascade_path in cascade_files.items():
        try:
            # Detect license plates with this cascade
            plates = detect_with_xml_cascade(
                opencv_image, 
                cascade_path,
                scale_factor=1.05,
                min_neighbors=3,
                min_size=(50, 15)
            )
            
            detection_info.append(f"**{cascade_name.title()}**: {len(plates)} detection(s)")
            
            for (x, y, w, h) in plates:
                # Validate detection (basic size and aspect ratio checks)
                aspect_ratio = w / h if h > 0 else 0
                if 1.5 <= aspect_ratio <= 6.0 and w >= 40 and h >= 12:
                    confidence = 0.8  # XML cascades are generally reliable
                    all_detections.append((x, y, w, h, confidence, cascade_name))
                    
        except Exception as e:
            detection_info.append(f"**{cascade_name.title()}**: Error - {str(e)}")
    
    # Remove duplicate detections (overlapping regions)
    filtered_detections = []
    for detection in all_detections:
        x1, y1, w1, h1, conf1, name1 = detection
        is_duplicate = False
        
        for existing in filtered_detections:
            x2, y2, w2, h2, conf2, name2 = existing
            
            # Calculate overlap
            overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            overlap_area = overlap_x * overlap_y
            
            area1 = w1 * h1
            area2 = w2 * h2
            
            # If overlap is more than 50% of either detection, consider it duplicate
            if overlap_area > 0.5 * min(area1, area2):
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered_detections.append(detection)
    
    # Process detections
    processed_count = 0
    for x, y, w, h, confidence, cascade_name in filtered_detections:
        try:
            # Refine bounds to focus on text area
            if refine_bounds:
                refined_x, refined_y, refined_w, refined_h = refine_detection_bounds(
                    opencv_image, x, y, w, h
                )
            else:
                refined_x, refined_y, refined_w, refined_h = x, y, w, h
            
            # Ensure bounds are within image
            img_h, img_w = opencv_image.shape[:2]
            refined_x = max(0, min(refined_x, img_w - 1))
            refined_y = max(0, min(refined_y, img_h - 1))
            refined_w = min(refined_w, img_w - refined_x)
            refined_h = min(refined_h, img_h - refined_y)
            
            if refined_w > 10 and refined_h > 5:
                # Extract region and apply blur
                plate_region = result_image[refined_y:refined_y+refined_h, 
                                          refined_x:refined_x+refined_w]
                
                # Calculate blur size based on intensity setting and region size
                base_blur = max(15, min(refined_w//6, refined_h//3))
                blur_size = int(base_blur * (blur_intensity / 5.0))
                if blur_size % 2 == 0:
                    blur_size += 1
                
                # Apply Gaussian blur
                blurred_plate = cv2.GaussianBlur(plate_region, (blur_size, blur_size), 0)
                result_image[refined_y:refined_y+refined_h, 
                           refined_x:refined_x+refined_w] = blurred_plate
                
                processed_count += 1
                
        except Exception as e:
            st.warning(f"Error processing detection: {str(e)}")
    
    # Convert back to RGB
    result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(result_rgb), detection_info, processed_count

def get_image_download_link(img, filename, text):
    """
    Generate a download link for the processed image
    """
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=95)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/jpeg;base64,{img_str}" download="{filename}">{text}</a>'
    return href

def main():
    st.set_page_config(
        page_title="XML License Plate Blur Tool",
        page_icon="🚗",
        layout="wide"
    )
    
    st.title("🚗 XML-Based License Plate Blur Tool")
    st.markdown("Upload a car image and automatically detect & blur license plates using **pre-trained XML cascade classifiers**.")
    
    # Initialize cascade files
    st.sidebar.header("🔧 Detection Setup")
    
    if st.sidebar.button("📥 Download XML Cascade Files", help="Download pre-trained Haar cascade classifiers"):
        cascade_files = download_cascade_files()
        if cascade_files:
            st.session_state.cascade_files = cascade_files
        else:
            st.error("Failed to download cascade files. Please check your internet connection.")
    
    # Settings
    st.sidebar.header("⚙️ Settings")
    blur_intensity = st.sidebar.slider("Blur Intensity", 1, 10, 5, 
                                     help="Higher values = stronger blur")
    refine_bounds = st.sidebar.checkbox("Refine Text Bounds", True, 
                                       help="Focus blur on text area only")
    show_detection_info = st.sidebar.checkbox("Show Detection Details", False)
    
    # Instructions
    st.sidebar.header("📖 How it Works")
    st.sidebar.markdown("""
    **XML Cascade Detection:**
    1. Uses pre-trained Haar cascade classifiers
    2. Multiple cascade models for better coverage
    3. Removes duplicate detections
    4. Optional text area refinement
    5. Adjustable blur intensity
    
    **Advantages:**
    - More accurate than contour-based detection
    - Trained on thousands of license plate images
    - Works with various angles and lighting
    - Fewer false positives
    """)
    
    # Check if cascade files are available
    if 'cascade_files' not in st.session_state or not st.session_state.cascade_files:
        st.warning("⚠️ **XML Cascade files not loaded.** Click 'Download XML Cascade Files' in the sidebar to get started.")
        st.info("💡 **Why XML Cascades?** They use machine learning trained on thousands of license plate images, making them much more accurate than simple contour detection.")
        
        # Show example of what XML detection can do
        st.markdown("### 🎯 XML Cascade Detection Benefits:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**✅ More Accurate:**")
            st.markdown("- Trained on real license plate data")
            st.markdown("- Handles various angles & lighting")
            st.markdown("- Fewer false positives")
            
        with col2:
            st.markdown("**⚡ Better Performance:**") 
            st.markdown("- Fast detection algorithm")
            st.markdown("- Multiple cascade models")
            st.markdown("- Automatic duplicate removal")
        
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image containing cars with license plates"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)
            
            # Image info
            st.write(f"**Size:** {image.size[0]} × {image.size[1]} pixels")
            st.write(f"**File Size:** {len(uploaded_file.getvalue())} bytes")
        
        with col2:
            st.subheader("🎯 Processed Image")
            
            with st.spinner("🔍 Detecting license plates with XML cascades..."):
                try:
                    processed_image, detection_info, processed_count = detect_and_blur_plates_xml(
                        image, 
                        st.session_state.cascade_files,
                        refine_bounds=refine_bounds,
                        blur_intensity=blur_intensity
                    )
                    
                    st.image(processed_image, caption=f"Processed ({processed_count} plates blurred)", 
                           use_column_width=True)
                    
                    # Detection information
                    if show_detection_info:
                        st.markdown("### 🔍 Detection Details")
                        for info in detection_info:
                            st.markdown(info)
                        
                        if processed_count > 0:
                            st.success(f"✅ Successfully processed {processed_count} license plate(s)")
                        else:
                            st.info("ℹ️ No license plates detected in this image")
                    
                    # Download section
                    st.markdown("### 📥 Download")
                    
                    # Generate filename
                    original_filename = uploaded_file.name
                    name_without_ext = original_filename.rsplit('.', 1)[0]
                    processed_filename = f"{name_without_ext}_xml_blurred.jpg"
                    
                    # Download button
                    buffered = io.BytesIO()
                    processed_image.save(buffered, format="JPEG", quality=95)
                    
                    st.download_button(
                        label="📥 Download Processed Image",
                        data=buffered.getvalue(),
                        file_name=processed_filename,
                        mime="image/jpeg",
                        help="Download the image with blurred license plates"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error processing image: {str(e)}")
                    st.info("💡 Try adjusting the settings or use a different image.")
    
    else:
        st.info("👆 **Please upload an image to get started**")
        
        st.markdown("### 🚀 XML Cascade Detection Features")
        
        feature_col1, feature_col2, feature_col3 = st.columns(3)
        
        with feature_col1:
            st.markdown("**🎯 Accurate Detection**")
            st.markdown("Pre-trained on thousands of license plate images")
            
        with feature_col2:
            st.markdown("**⚡ Fast Processing**") 
            st.markdown("Optimized Haar cascade algorithms")
            
        with feature_col3:
            st.markdown("**🔧 Customizable**")
            st.markdown("Adjustable blur intensity and text refinement")

if __name__ == "__main__":
    main()