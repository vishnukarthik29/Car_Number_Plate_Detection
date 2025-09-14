import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import urllib.request

# Page configuration
st.set_page_config(
    page_title="Number Plate Detection & Blur",
    page_icon="🚗",
    layout="wide"
)

@st.cache_resource
def download_cascade():
    """Download the Haar Cascade XML file for number plate detection"""
    cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_russian_plate_number.xml"
    cascade_path = "haarcascade_plate.xml"
    
    if not os.path.exists(cascade_path):
        try:
            urllib.request.urlretrieve(cascade_url, cascade_path)
            st.success("Cascade file downloaded successfully!")
        except:
            st.error("Could not download cascade file. Please provide your own XML file.")
            return None
    
    return cascade_path

def detect_and_blur_plates(image, cascade_path, blur_intensity=50):
    """
    Detect number plates in the image and blur them
    
    Parameters:
    - image: Input image (numpy array)
    - cascade_path: Path to the Haar Cascade XML file
    - blur_intensity: Intensity of the blur effect
    
    Returns:
    - processed_image: Image with blurred number plates
    - num_plates: Number of plates detected
    """
    # Convert to grayscale for detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load the cascade classifier
    plate_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Detect number plates
    plates = plate_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Create a copy of the original image
    processed_image = image.copy()
    
    # Blur each detected plate region
    for (x, y, w, h) in plates:
        # Extract the plate region
        plate_region = processed_image[y:y+h, x:x+w]
        
        # Apply Gaussian blur
        blurred_region = cv2.GaussianBlur(
            plate_region, 
            (blur_intensity | 1, blur_intensity | 1),  # Ensure odd number
            0
        )
        
        # Replace the original region with blurred version
        processed_image[y:y+h, x:x+w] = blurred_region
        
        # Draw a rectangle around the detected plate (optional)
        cv2.rectangle(processed_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return processed_image, len(plates)

def main():
    st.title("🚗 Vehicle Number Plate Detection & Blur")
    st.markdown("Upload an image to detect and blur vehicle number plates automatically")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        blur_intensity = st.slider(
            "Blur Intensity",
            min_value=10,
            max_value=100,
            value=50,
            step=10,
            help="Higher values create stronger blur effect"
        )
        
        show_original = st.checkbox("Show Original Image", value=True)
        
        st.markdown("---")
        st.header("📁 Custom XML File")
        custom_xml = st.file_uploader(
            "Upload custom Haar Cascade XML",
            type=['xml'],
            help="Upload your own XML file for better detection"
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image containing vehicles with visible number plates"
        )
    
    if uploaded_file is not None:
        # Load the image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_np.shape) == 3:
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        
        # Determine which cascade file to use
        if custom_xml is not None:
            # Save custom XML temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xml') as tmp_file:
                tmp_file.write(custom_xml.getvalue())
                cascade_path = tmp_file.name
        else:
            # Download or use default cascade
            cascade_path = download_cascade()
        
        if cascade_path:
            # Process the image
            with st.spinner("Processing image..."):
                processed_image, num_plates = detect_and_blur_plates(
                    image_cv, 
                    cascade_path, 
                    blur_intensity
                )
            
            # Convert back to RGB for display
            processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            
            # Display results
            st.success(f"✅ Detected and blurred {num_plates} number plate(s)")
            
            # Show images
            if show_original:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Image")
                    st.image(image, use_column_width=True)
                with col2:
                    st.subheader("Processed Image")
                    st.image(processed_image_rgb, use_column_width=True)
            else:
                st.subheader("Processed Image")
                st.image(processed_image_rgb, use_column_width=True)
            
            # Download button for processed image
            processed_pil = Image.fromarray(processed_image_rgb)
            buf = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            processed_pil.save(buf.name, format='PNG')
            
            with open(buf.name, 'rb') as f:
                st.download_button(
                    label="📥 Download Processed Image",
                    data=f.read(),
                    file_name="blurred_plate_image.png",
                    mime="image/png"
                )
            
            # Clean up temporary files
            if custom_xml is not None:
                os.unlink(cascade_path)
            os.unlink(buf.name)
    
    else:
        # Instructions when no image is uploaded
        with col2:
            st.info("""
            ### 📋 Instructions:
            1. Upload an image containing vehicles
            2. The app will automatically detect number plates
            3. Detected plates will be blurred for privacy
            4. Adjust blur intensity in the sidebar
            5. Download the processed image
            
            ### 💡 Tips:
            - Works best with clear, front-facing vehicle images
            - Good lighting improves detection accuracy
            - You can upload custom XML cascade files for better results
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🔒 Privacy-focused number plate detection and blurring</p>
        <p style='font-size: 12px; color: gray;'>
            Note: Detection accuracy depends on image quality and cascade file used.
            For better results, consider training a custom cascade or using deep learning models.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()