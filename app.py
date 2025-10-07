import streamlit as st
import numpy as np
from PIL import Image
import os
import requests
import cv2

# Page configuration
st.set_page_config(
    page_title="Oil Spill Detection AI",
    page_icon="🌊",
    layout="wide"
)

# TensorFlow import with better error handling
TENSORFLOW_AVAILABLE = False
CLASSIFICATION_MODEL = None
SEGMENTATION_MODEL = None

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    st.sidebar.success("✅ TensorFlow loaded!")
except ImportError as e:
    st.sidebar.error(f"❌ TensorFlow not available: {str(e)}")
    st.sidebar.info("🔧 Running in demo mode")

def download_model_from_drive(file_id, output_path):
    """Download model from Google Drive"""
    try:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        session = requests.Session()
        response = session.get(url, stream=True)
        
        # Handle large file confirmation
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                url = f"https://drive.google.com/uc?export=download&confirm={value}&id={file_id}"
                response = session.get(url, stream=True)
                break
        
        # Download the file
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return True
        return False
    except Exception as e:
        st.error(f"Download error: {str(e)}")
        return False

def load_models():
    """Load models if TensorFlow is available"""
    global CLASSIFICATION_MODEL, SEGMENTATION_MODEL
    
    if not TENSORFLOW_AVAILABLE:
        return False
        
    try:
        os.makedirs("models", exist_ok=True)
        
        # YOUR GOOGLE DRIVE FILE IDs - REPLACE THESE
        models_to_download = [
            {"file_id": "https://drive.google.com/file/d/1EFYBAaMoLY6SCPO5fqyxPP870fnLbVtr/view?usp=drive_link", "filename": "simple_cnn_classifier.h5"},
            {"file_id": "https://drive.google.com/file/d/1oddiVJOirUYGhUnXHGqOrK8W1N7cUxIi/view?usp=drive_link", "filename": "best_improved_unet_model.h5"}
        ]
        
        for model in models_to_download:
            local_path = f"models/{model['filename']}"
            if not os.path.exists(local_path):
                if download_model_from_drive(model["file_id"], local_path):
                    st.sidebar.success(f"✅ Downloaded {model['filename']}")
        
        # Load models
        if os.path.exists("models/simple_cnn_classifier.h5"):
            CLASSIFICATION_MODEL = tf.keras.models.load_model("models/simple_cnn_classifier.h5")
            
        if os.path.exists("models/best_improved_unet_model.h5"):
            SEGMENTATION_MODEL = tf.keras.models.load_model("models/best_improved_unet_model.h5", compile=False)
            
        return CLASSIFICATION_MODEL is not None or SEGMENTATION_MODEL is not None
        
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return False

def preprocess_image(image):
    """Preprocess image for model input"""
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]

    # Resize to model input size
    image_resized = cv2.resize(image, (256, 256))
    image_normalized = image_resized.astype(np.float32) / 255.0

    return np.expand_dims(image_normalized, axis=0)

def predict_with_ai(image):
    """Use AI models for prediction if available"""
    if not TENSORFLOW_AVAILABLE or (CLASSIFICATION_MODEL is None and SEGMENTATION_MODEL is None):
        return predict_demo(image)
    
    try:
        processed_image = preprocess_image(image)
        
        classification_has_oil = False
        classification_confidence = 0.5
        seg_mask = np.zeros((256, 256))
        oil_percentage = 0.0
        
        # Classification model
        if CLASSIFICATION_MODEL:
            cls_pred = CLASSIFICATION_MODEL.predict(processed_image, verbose=0)[0][0]
            classification_has_oil = cls_pred > 0.5
            classification_confidence = float(cls_pred)
        
        # Segmentation model
        if SEGMENTATION_MODEL:
            seg_pred = SEGMENTATION_MODEL.predict(processed_image, verbose=0)
            
            if len(seg_pred[0].shape) == 3:
                pred_mask = seg_pred[0][:, :, 0]
            else:
                pred_mask = seg_pred[0]
            
            seg_mask = (pred_mask > 0.5).astype(np.uint8)
            oil_pixels = np.sum(seg_mask)
            oil_percentage = (oil_pixels / seg_mask.size) * 100
        
        # Combine results
        segmentation_has_oil = oil_percentage > 0.5
        final_has_oil = classification_has_oil or segmentation_has_oil
        
        if CLASSIFICATION_MODEL and SEGMENTATION_MODEL:
            segmentation_confidence = min(0.95, 0.7 + (oil_percentage / 100) * 0.25)
            confidence = (classification_confidence * 0.3 + segmentation_confidence * 0.7) * 100
        elif CLASSIFICATION_MODEL:
            confidence = classification_confidence * 100
        elif SEGMENTATION_MODEL:
            confidence = min(95, 70 + (oil_percentage / 100) * 25)
        else:
            confidence = 50.0
            
        return final_has_oil, seg_mask, confidence, oil_percentage
        
    except Exception as e:
        st.error(f"AI prediction error: {str(e)}")
        return predict_demo(image)

def predict_demo(image):
    """Demo prediction"""
    img_array = np.array(image.resize((256, 256)))
    mask = np.zeros((256, 256))
    
    # Create simulated detection
    center_y, center_x = 128, 128
    y, x = np.ogrid[:256, :256]
    mask = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (60**2)))
    binary_mask = (mask > 0.3).astype(np.uint8)
    
    oil_pixels = np.sum(binary_mask)
    oil_percentage = (oil_pixels / binary_mask.size) * 100
    confidence = min(85, 60 + oil_percentage)
    has_oil = oil_percentage > 5
    
    return has_oil, binary_mask, confidence, oil_percentage

def create_visualization(original_image, segmentation_mask):
    """Create result visualizations"""
    if isinstance(original_image, Image.Image):
        original_np = np.array(original_image)
    else:
        original_np = original_image

    if original_np.shape[-1] == 4:
        original_np = original_np[..., :3]

    # Resize for display
    display_size = (400, 400)
    display_original = cv2.resize(original_np, display_size)

    # Create colored mask
    mask_resized = cv2.resize(segmentation_mask, display_size, interpolation=cv2.INTER_NEAREST)
    colored_mask = np.zeros((display_size[1], display_size[0], 3), dtype=np.uint8)
    colored_mask[mask_resized > 0] = [255, 0, 124]

    # Create overlay
    overlay = display_original.copy().astype(float)
    oil_areas = mask_resized > 0
    overlay[oil_areas] = overlay[oil_areas] * 0.6 + np.array([255, 0, 124]) * 0.4
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return display_original, colored_mask, overlay

def main():
    st.title("🌊 Oil Spill Detection AI")
    
    # Load models at startup
    models_loaded = load_models() if TENSORFLOW_AVAILABLE else False
    
    st.sidebar.title("📤 Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose satellite image", type=["jpg", "jpeg", "png"])
    
    st.sidebar.markdown("---")
    
    # Display status
    if models_loaded:
        st.sidebar.success("**Status:** ✅ AI MODELS ACTIVE")
    elif TENSORFLOW_AVAILABLE:
        st.sidebar.warning("**Status:** ⚠️ MODELS NOT LOADED")
    else:
        st.sidebar.info("**Status:** 🔧 DEMO MODE")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("🔍 Detect Oil Spills", type="primary"):
            with st.spinner("Analyzing image..."):
                if models_loaded:
                    has_oil, mask, confidence, oil_percent = predict_with_ai(image)
                    source = "AI Models"
                else:
                    has_oil, mask, confidence, oil_percent = predict_demo(image)
                    source = "Demo Mode"
                
                # Display results
                col1, col2, col3 = st.columns(3)
                with col1:
                    if has_oil:
                        st.error("🚨 OIL SPILL DETECTED")
                    else:
                        st.success("✅ NO OIL DETECTED")
                with col2:
                    st.metric("Confidence", f"{confidence:.1f}%")
                with col3:
                    st.metric("Oil Coverage", f"{oil_percent:.2f}%")
                
                risk = "HIGH" if oil_percent > 15 else "MEDIUM" if oil_percent > 5 else "LOW"
                st.info(f"**Risk Level:** {risk}")
                
                # Visualizations
                orig_viz, mask_viz, overlay_viz = create_visualization(image, mask)
                
                viz_col1, viz_col2, viz_col3 = st.columns(3)
                with viz_col1:
                    st.image(orig_viz, caption="Original", use_container_width=True)
                with viz_col2:
                    st.image(mask_viz, caption="Detection", use_container_width=True)
                with viz_col3:
                    st.image(overlay_viz, caption="Overlay", use_container_width=True)
                    
                if not models_loaded:
                    st.warning("💡 **Demo Mode** - Add Google Drive file IDs for real AI detection")
    
    else:
        st.info("👆 Upload a satellite image to start detection")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🎯 Features")
            st.markdown("- AI-Powered Detection")
            st.markdown("- Real-time Analysis")
            st.markdown("- Risk Assessment")
        
        with col2:
            st.markdown("### 🔧 Status")
            if TENSORFLOW_AVAILABLE:
                st.success("TensorFlow: Available")
            else:
                st.error("TensorFlow: Not Available")
            
            if models_loaded:
                st.success("AI Models: Loaded")
            else:
                st.warning("AI Models: Not Loaded")

if __name__ == "__main__":
    main()
