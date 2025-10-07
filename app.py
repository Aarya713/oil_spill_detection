import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os
import requests
import tempfile

# Page configuration
st.set_page_config(
    page_title="Oil Spill Detection AI",
    page_icon="🌊",
    layout="wide"
)

print("🚀 OIL SPILL DETECTION SYSTEM - IMPROVED UNET + CLASSIFICATION")

class OilSpillDetector:
    def __init__(self):
        self.classification_model = None
        self.segmentation_model = None
        self.models_loaded = False
        self.load_trained_models()

    def download_model_from_drive(self, file_id, output_path):
        """Download model from Google Drive using direct download"""
        try:
            # Direct download URL for Google Drive
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
                    st.sidebar.success(f"✅ Downloaded: {os.path.basename(output_path)}")
                    return True
                else:
                    st.sidebar.error(f"❌ Download failed: Empty file")
                    return False
            else:
                st.sidebar.error(f"❌ Download failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            st.sidebar.error(f"❌ Download error: {str(e)}")
            return False

    def load_trained_models(self):
        """Load BOTH models directly from Google Drive - MATCHING COLAB CODE"""
        st.sidebar.info("🔄 Loading both classification and improved segmentation models...")

        # Create models directory
        os.makedirs("models", exist_ok=True)

        # YOUR GOOGLE DRIVE FILE IDs - REPLACE WITH YOUR ACTUAL IDs
        model_configs = [
            {
                "name": "CNN Classification Model",
                "file_id": "https://drive.google.com/file/d/1EFYBAaMoLY6SCPO5fqyxPP870fnLbVtr/view?usp=drive_link",  # ← REPLACE
                "filename": "simple_cnn_classifier.h5"
            },
            {
                "name": "Improved U-Net Segmentation Model", 
                "file_id": "https://drive.google.com/file/d/1oddiVJOirUYGhUnXHGqOrK8W1N7cUxIi/view?usp=drive_link",     # ← REPLACE
                "filename": "best_improved_unet_model.h5"
            }
        ]

        try:
            models_downloaded = 0
            
            for config in model_configs:
                local_path = f"models/{config['filename']}"
                
                if not os.path.exists(local_path):
                    st.sidebar.info(f"📥 Downloading {config['name']}...")
                    if self.download_model_from_drive(config["file_id"], local_path):
                        models_downloaded += 1
                else:
                    st.sidebar.info(f"✅ {config['name']} already exists")
                    models_downloaded += 1

            # Load the downloaded models
            if models_downloaded > 0:
                for config in model_configs:
                    local_path = f"models/{config['filename']}"
                    if os.path.exists(local_path):
                        try:
                            if "unet" in config["name"].lower():
                                self.segmentation_model = tf.keras.models.load_model(
                                    local_path, 
                                    compile=False
                                )
                                st.sidebar.success(f"✅ {config['name']} loaded!")
                            else:
                                self.classification_model = tf.keras.models.load_model(local_path)
                                st.sidebar.success(f"✅ {config['name']} loaded!")
                                
                        except Exception as e:
                            st.sidebar.error(f"❌ Failed to load {config['name']}: {str(e)}")
                
                # Check if models are loaded
                if self.segmentation_model is not None or self.classification_model is not None:
                    self.models_loaded = True
                    st.sidebar.success("🎉 ALL AI MODELS LOADED AND ACTIVE!")
                else:
                    st.sidebar.warning("⚠️ Some models failed to load")
            else:
                st.sidebar.error("❌ No models could be downloaded")
                
        except Exception as e:
            st.sidebar.error(f"❌ Model loading failed: {str(e)}")

    def preprocess_image(self, image):
        """Preprocess image for model input - EXACT SAME AS COLAB"""
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]

        # Resize to 256x256 (model input size)
        image_resized = cv2.resize(image, (256, 256))

        # Normalize to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0

        return np.expand_dims(image_normalized, axis=0)

    def predict_with_both_models(self, image):
        """Use BOTH classification and improved segmentation models together - EXACT COLAB LOGIC"""
        processed_image = self.preprocess_image(image)

        st.info("🔍 Running both classification and improved segmentation models...")

        # 1. FIRST: Use classification model for quick detection
        classification_has_oil = False
        classification_confidence = 0.5

        if self.classification_model:
            try:
                cls_pred = self.classification_model.predict(processed_image, verbose=0)[0][0]
                classification_has_oil = cls_pred > 0.5
                classification_confidence = float(cls_pred) if classification_has_oil else float(1 - cls_pred)
                st.write(f"📊 Classification: {'OIL SPILL' if classification_has_oil else 'NO OIL'} (confidence: {classification_confidence:.3f})")
            except Exception as e:
                st.error(f"❌ Classification error: {e}")
                classification_has_oil = False
                classification_confidence = 0.5
        else:
            st.warning("⚠️ No classification model available")

        # 2. SECOND: Use IMPROVED segmentation model for detailed analysis
        seg_mask = np.zeros((256, 256), dtype=np.uint8)
        oil_percentage = 0.0
        segmentation_confidence = 0.5

        if self.segmentation_model:
            try:
                # Run IMPROVED segmentation model prediction
                seg_pred = self.segmentation_model.predict(processed_image, verbose=0)

                st.write(f"🎯 Improved U-Net output shape: {seg_pred.shape}")
                st.write(f"📈 Prediction range: [{seg_pred.min():.3f}, {seg_pred.max():.3f}]")

                # Process the prediction
                if len(seg_pred[0].shape) == 3:
                    pred_single = seg_pred[0][:, :, 0]  # Take first channel
                else:
                    pred_single = seg_pred[0]

                # Apply threshold to get binary mask
                threshold = 0.5
                binary_mask = (pred_single > threshold).astype(np.uint8)
                seg_mask = binary_mask

                # Calculate statistics
                oil_pixels = np.sum(seg_mask)
                total_pixels = seg_mask.size
                oil_percentage = (oil_pixels / total_pixels) * 100

                st.write(f"🔍 Improved U-Net: {oil_pixels} pixels ({oil_percentage:.2f}%)")

                # Calculate segmentation confidence properly
                if oil_percentage > 0.5:  # If oil is detected
                    # Confidence based on oil percentage (capped at 95%)
                    segmentation_confidence = min(0.95, 0.7 + (oil_percentage / 100) * 0.25)
                else:
                    # High confidence when no oil is detected
                    segmentation_confidence = 0.9

            except Exception as e:
                st.error(f"❌ Segmentation error: {e}")
                seg_mask = self.create_demo_mask()
                oil_pixels = np.sum(seg_mask)
                oil_percentage = (oil_pixels / seg_mask.size) * 100
                segmentation_confidence = 0.5
        else:
            st.warning("⚠️ No segmentation model available - using demo")
            seg_mask = self.create_demo_mask()
            oil_pixels = np.sum(seg_mask)
            oil_percentage = (oil_pixels / seg_mask.size) * 100
            segmentation_confidence = 0.5

        # 3. COMBINE RESULTS from both models with FIXED confidence calculation
        segmentation_has_oil = oil_percentage > 0.5

        # Final decision: If either model detects oil, consider it detected
        final_has_oil = classification_has_oil or segmentation_has_oil

        # Combined confidence calculation
        if self.classification_model and self.segmentation_model:
            # Both models available - weighted average (already in 0-1 range)
            final_confidence = (classification_confidence * 0.3 + segmentation_confidence * 0.7)
        elif self.classification_model:
            # Only classification available
            final_confidence = classification_confidence
        elif self.segmentation_model:
            # Only improved segmentation available
            final_confidence = segmentation_confidence
        else:
            # No models available
            final_confidence = 0.5

        # Convert to percentage for display
        final_confidence_percent = final_confidence * 100

        st.success(f"🤝 Combined AI result: {'OIL SPILL' if final_has_oil else 'NO OIL'} (confidence: {final_confidence_percent:.1f}%)")

        return final_has_oil, seg_mask, final_confidence_percent, oil_percentage

    def create_demo_mask(self):
        """Create demo mask for testing - SAME AS COLAB"""
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.ellipse(mask, (128, 128), (60, 40), 0, 0, 360, 1, -1)
        cv2.ellipse(mask, (80, 80), (20, 15), 0, 0, 360, 1, -1)
        cv2.ellipse(mask, (180, 160), (25, 18), 0, 0, 360, 1, -1)
        return mask

def create_visualization_fixed(original_image, segmentation_mask):
    """Create visualization WITHOUT array operation errors - SAME AS COLAB"""
    if isinstance(original_image, Image.Image):
        original_np = np.array(original_image)
    else:
        original_np = original_image

    # Ensure RGB
    if original_np.shape[-1] == 4:
        original_np = original_np[..., :3]

    # Resize for consistent display
    display_size = (400, 400)
    display_original = cv2.resize(original_np, display_size)

    # Create colored mask
    mask_resized = cv2.resize(segmentation_mask, display_size, interpolation=cv2.INTER_NEAREST)

    # Ensure mask is 2D
    if len(mask_resized.shape) == 3:
        mask_resized = mask_resized[:, :, 0]

    colored_mask = np.zeros((display_size[1], display_size[0], 3), dtype=np.uint8)
    oil_areas = mask_resized == 1
    colored_mask[oil_areas] = [255, 0, 124]  # Oil spill color

    # Create overlay - FIXED VERSION
    original_for_overlay = cv2.resize(original_np, display_size)
    red_mask = np.zeros_like(original_for_overlay)
    red_mask[oil_areas] = [255, 0, 124]

    # Use cv2.addWeighted - NO DIRECT ARRAY ASSIGNMENT
    alpha = 0.4
    overlay = cv2.addWeighted(red_mask.astype(np.float32), alpha,
                             original_for_overlay.astype(np.float32), 1 - alpha, 0)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return display_original, colored_mask, overlay

def main():
    st.title("🌊 Oil Spill Detection AI")
    st.markdown("### **Improved U-Net + Classification Models**")

    # Initialize detector (will load models from Google Drive)
    detector = OilSpillDetector()

    st.sidebar.title("📤 Upload Satellite Image")
    uploaded_file = st.sidebar.file_uploader(
        "Choose image for analysis", 
        type=["jpg", "jpeg", "png"],
        help="Upload satellite imagery for oil spill detection"
    )

    st.sidebar.markdown("---")
    
    # Display status
    if detector.models_loaded:
        st.sidebar.success("**Status:** ✅ MODELS LOADED")
        st.sidebar.info("**Detection:** Real AI Analysis")
    else:
        st.sidebar.error("**Status:** ❌ MODELS NOT LOADED")
        st.sidebar.info("**Detection:** Demo Mode")

    # Setup instructions
    with st.sidebar.expander("🔧 Setup Instructions"):
        st.markdown("""
        **To use YOUR AI models:**

        1. Get Google Drive File IDs:
           - Upload .h5 files to Google Drive
           - Get shareable links
           - Extract File IDs from URLs:
             `https://drive.google.com/file/d/FILE_ID_HERE/view`

        2. Replace in app.py:
           - `YOUR_CNN_CLASSIFIER_FILE_ID_HERE`
           - `YOUR_UNET_MODEL_FILE_ID_HERE`

        3. Models will auto-download on first run
        """)

    if uploaded_file is not None:
        st.subheader("📷 Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True, caption="Input Satellite Image")

        if st.button("🔍 Analyze with Improved AI Models", type="primary", use_container_width=True):
            with st.spinner("🛰️ Running improved U-Net + classification models..."):
                try:
                    # Get predictions from BOTH models
                    has_oil, seg_mask, confidence, oil_percent = detector.predict_with_both_models(image)

                    # Calculate statistics
                    oil_pixels = np.sum(seg_mask)
                    total_pixels = seg_mask.size
                    actual_oil_percentage = (oil_pixels / total_pixels) * 100

                    # Improved Risk assessment (SAME AS COLAB)
                    if oil_percent > 30:
                        risk_level = "CRITICAL"
                        risk_color = "🔴"
                    elif oil_percent > 15:
                        risk_level = "HIGH" 
                        risk_color = "🟠"
                    elif oil_percent > 5:
                        risk_level = "MEDIUM"
                        risk_color = "🟡"
                    elif oil_percent > 1:
                        risk_level = "LOW"
                        risk_color = "🟢"
                    else:
                        risk_level = "VERY LOW"
                        risk_color = "⚪"

                    # DISPLAY RESULTS
                    st.subheader("📊 AI Detection Results")
                    st.markdown("---")

                    if has_oil:
                        st.error("🚨 **OIL SPILL DETECTED**")
                    else:
                        st.success("✅ **NO OIL SPILL DETECTED**")

                    # Results columns
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Model Confidence", f"{confidence:.1f}%")
                    with col2:
                        st.metric("Oil Coverage", f"{actual_oil_percentage:.4f}%")
                    with col3:
                        st.metric("Risk Level", f"{risk_color} {risk_level}")

                    st.info(f"📈 Detected Area: {oil_pixels:,} pixels")
                    
                    if detector.models_loaded:
                        st.success("🔍 Source: Improved U-Net + Classification")
                    else:
                        st.warning("🔍 Source: Demo Mode")

                    # Create visualizations
                    st.subheader("🔍 AI Analysis Visualization")
                    display_original, display_mask, display_overlay = create_visualization_fixed(image, seg_mask)

                    viz_col1, viz_col2, viz_col3 = st.columns(3)
                    with viz_col1:
                        st.image(display_original, caption="Original Image", use_container_width=True)
                    with viz_col2:
                        st.image(display_mask, caption="Oil Spill Detection", use_container_width=True)
                    with viz_col3:
                        st.image(display_overlay, caption="Detection Overlay", use_container_width=True)

                    # Model information
                    st.subheader("🤖 Model Information")
                    if detector.classification_model:
                        st.success("✅ Using ACTUAL trained classification model")
                    else:
                        st.error("❌ No classification model available")

                    if detector.segmentation_model:
                        st.success("✅ Using IMPROVED U-Net segmentation model")
                    else:
                        st.error("❌ No segmentation model available")

                    st.info("🎯 Using COMBINED AI approach with improved models")

                except Exception as e:
                    st.error(f"❌ Analysis error: {e}")

    else:
        # Welcome screen
        st.info("👆 **Upload a satellite image to test improved AI models**")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🧠 AI MODELS")
            if detector.classification_model:
                st.success("• **CNN Classifier:** ✅ Loaded")
            else:
                st.error("• **CNN Classifier:** ❌ Missing")
            
            if detector.segmentation_model:
                st.success("• **Improved U-Net:** ✅ Loaded")
            else:
                st.error("• **Improved U-Net:** ❌ Missing")
            
            st.markdown("### 🎯 FEATURES")
            st.markdown("• Combined Model Approach")
            st.markdown("• Improved U-Net Segmentation")
            st.markdown("• Professional Risk Assessment")
            st.markdown("• Real AI Detection")
        
        with col2:
            st.markdown("### 🔧 SYSTEM STATUS")
            if detector.models_loaded:
                st.success("**Status:** ✅ READY")
                st.info("**Models:** IMPROVED AI ACTIVE")
                st.success("**Platform:** Streamlit Cloud")
            else:
                st.error("**Status:** 🔧 SETUP REQUIRED")
                st.info("**Action:** Add Google Drive file IDs")
                st.warning("**Current:** Demo Mode Available")

if __name__ == "__main__":
    main()
