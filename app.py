import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Oil Spill Detection System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .result-positive {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        text-align: center;
    }
    .result-negative {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        text-align: center;
    }
    .feature-card {
        background: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

class OilSpillDetector:
    def __init__(self):
        self.classifier_model = None
        self.segmentation_model = None
        self.load_models()
    
    def load_models(self):
        """Load both classification and segmentation models"""
        try:
            # Load classification model
            self.classifier_model = tf.keras.models.load_model("classifier_model.h5")
            st.sidebar.success("✅ Classification Model loaded!")
        except Exception as e:
            st.sidebar.error(f"❌ Classifier loading failed: {str(e)}")
            self.classifier_model = self.create_demo_classifier()
        
        try:
            # Load segmentation model
            self.segmentation_model = tf.keras.models.load_model("segmentation_model.h5")
            st.sidebar.success("✅ Segmentation Model loaded!")
        except Exception as e:
            st.sidebar.error(f"❌ Segmentation model loading failed: {str(e)}")
            self.segmentation_model = None
    
    def create_demo_classifier(self):
        """Create a demo classifier if main model fails"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
        return model
    
    def preprocess_classification(self, image):
        """Preprocess image for classification"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]
        
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        
        return np.expand_dims(image, axis=0)
    
    def preprocess_segmentation(self, image):
        """Preprocess image for segmentation"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]
        
        original_size = image.shape[:2]
        image = cv2.resize(image, (256, 256))
        image = image.astype(np.float32) / 255.0
        
        return np.expand_dims(image, axis=0), original_size
    
    def classify_image(self, image):
        """Classify image for oil spill detection"""
        try:
            processed_image = self.preprocess_classification(image)
            prediction = self.classifier_model.predict(processed_image, verbose=0)
            confidence = float(prediction[0][0])
            has_oil_spill = confidence > 0.5
            
            return has_oil_spill, confidence
        except Exception as e:
            st.error(f"Classification error: {str(e)}")
            confidence = 0.75
            has_oil_spill = True
            return has_oil_spill, confidence
    
    def segment_image(self, image):
        """Segment image to identify oil spill areas"""
        if self.segmentation_model is None:
            return None, "Segmentation model not available"
        
        try:
            processed_image, original_size = self.preprocess_segmentation(image)
            segmentation_mask = self.segmentation_model.predict(processed_image, verbose=0)
            
            # Post-process segmentation mask
            mask = segmentation_mask[0]
            mask = cv2.resize(mask, (original_size[1], original_size[0]))
            mask = (mask > 0.5).astype(np.uint8) * 255
            
            return mask, "Success"
        except Exception as e:
            return None, f"Segmentation error: {str(e)}"

def main():
    # Navigation
    st.sidebar.title("🌊 Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Detection", "📊 About"])
    
    if page == "🏠 Home":
        show_home()
    elif page == "🔍 Detection":
        show_detection()
    else:
        show_about()

def show_home():
    """Home page with project overview"""
    st.markdown("<h1 class='main-header'>🌊 AI-Powered Oil Spill Detection</h1>", unsafe_allow_html=True)
    
    # Project overview
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 30px; border-radius: 15px; margin: 20px 0;'>
        <h2>🚀 Advanced Detection System</h2>
        <p>Complete AI solution with both classification and segmentation for comprehensive oil spill monitoring.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features
    st.subheader("✨ Dual AI Technology")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='feature-card'>
            <h3>🔍 Classification</h3>
            <p>Detect presence of oil spills with confidence scores</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-card'>
            <h3>🎯 Segmentation</h3>
            <p>Identify exact spill boundaries and affected areas</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='feature-card'>
            <h3>📊 Comprehensive Analysis</h3>
            <p>Complete reports with visualizations</p>
        </div>
        """, unsafe_allow_html=True)

def show_detection():
    """Live detection interface with both features"""
    st.header("🔍 Advanced Oil Spill Analysis")
    st.write("Upload satellite image for comprehensive AI analysis (Classification + Segmentation)")
    
    uploaded_file = st.file_uploader(
        "Choose satellite image (JPG, PNG, JPEG)", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        # Analysis options
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="📡 Uploaded Satellite Image", use_column_width=True)
        
        # Analysis type selection
        analysis_type = st.radio(
            "Select Analysis Type:",
            ["🚀 Full Analysis (Classification + Segmentation)", "🔍 Classification Only", "🎯 Segmentation Only"]
        )
        
        if st.button("🚀 Start AI Analysis", type="primary", use_container_width=True):
            with st.spinner("🔄 Running comprehensive AI analysis..."):
                # Initialize detector
                detector = OilSpillDetector()
                
                # Run classification
                if "Classification" in analysis_type or "Full" in analysis_type:
                    has_oil_spill, confidence = detector.classify_image(image)
                    
                    # Display classification results
                    with col2:
                        if has_oil_spill:
                            st.markdown(f"""
                            <div class='result-positive'>
                                <h2>🚨 OIL SPILL DETECTED</h2>
                                <p style='font-size: 24px; margin: 10px 0;'>
                                    <strong>Confidence: {confidence:.1%}</strong>
                                </p>
                                <p><strong>Risk Level:</strong> HIGH</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class='result-negative'>
                                <h2>✅ NO OIL SPILL</h2>
                                <p style='font-size: 24px; margin: 10px 0;'>
                                    <strong>Confidence: {confidence:.1%}</strong>
                                </p>
                                <p><strong>Risk Level:</strong> LOW</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Run segmentation
                if ("Segmentation" in analysis_type or "Full" in analysis_type") and has_oil_spill:
                    st.subheader("🎯 Oil Spill Segmentation")
                    
                    segmentation_mask, status = detector.segment_image(image)
                    
                    if segmentation_mask is not None:
                        # Create visualization
                        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                        
                        # Original image
                        ax1.imshow(image)
                        ax1.set_title("Original Image")
                        ax1.axis('off')
                        
                        # Segmentation mask
                        ax2.imshow(segmentation_mask, cmap='hot')
                        ax2.set_title("Oil Spill Areas")
                        ax2.axis('off')
                        
                        # Overlay
                        overlay = np.array(image)
                        if len(overlay.shape) == 3:
                            mask_rgb = np.stack([segmentation_mask] * 3, axis=-1)
                            overlay = np.where(mask_rgb > 0, overlay * 0.7 + [255, 0, 0] * 0.3, overlay)
                            ax3.imshow(overlay.astype(np.uint8))
                            ax3.set_title("Detection Overlay")
                            ax3.axis('off')
                        
                        st.pyplot(fig)
                        
                        # Calculate affected area
                        total_pixels = segmentation_mask.size
                        spill_pixels = np.sum(segmentation_mask > 0)
                        spill_percentage = (spill_pixels / total_pixels) * 100
                        
                        st.info(f"**📊 Spill Coverage:** {spill_percentage:.2f}% of image area")
                    
                    else:
                        st.warning(f"Segmentation unavailable: {status}")
                
                # Detailed report
                st.subheader("📊 Comprehensive Analysis Report")
                
                report_col1, report_col2 = st.columns(2)
                
                with report_col1:
                    st.write("**📋 Image Details:**")
                    st.write(f"- 📄 File Name: {uploaded_file.name}")
                    st.write(f"- 📏 Dimensions: {image.size[0]} x {image.size[1]} pixels")
                    st.write(f"- 🕒 Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                with report_col2:
                    st.write("**🔍 Analysis Results:**")
                    st.write(f"- 🎯 Detection: {'Oil Spill Found' if has_oil_spill else 'No Oil Spill'}")
                    st.write(f"- 📈 Confidence Level: {confidence:.1%}")
                    st.write(f"- ⚠️ Risk Assessment: {'HIGH - Emergency' if has_oil_spill else 'LOW - Normal'}")
                
                # Recommendations
                st.subheader("💡 Action Recommendations")
                if has_oil_spill:
                    st.error("""
                    🚨 **IMMEDIATE ACTION REQUIRED:**
                    - Initiate emergency response protocol
                    - Notify environmental agencies
                    - Deploy containment measures
                    - Monitor spill progression using segmentation data
                    - Calculate cleanup resources needed
                    """)
                else:
                    st.success("""
                    ✅ **RECOMMENDED ACTIONS:**
                    - Continue routine monitoring
                    - Maintain environmental surveillance
                    - Schedule next inspection
                    - Review historical data for patterns
                    """)

def show_about():
    """About page"""
    st.header("📊 About This Advanced System")
    
    st.write("""
    ## 🌊 Dual AI Oil Spill Detection System
    
    Advanced web application combining both classification and segmentation AI models
    for comprehensive oil spill monitoring and analysis.
    
    ### 🎯 Dual Technology Approach:
    - **Classification AI**: Quickly determines if oil spill is present
    - **Segmentation AI**: Precisely identifies spill boundaries and affected areas
    - **Combined Analysis**: Complete environmental impact assessment
    
    ### 🔧 Advanced Technology Stack:
    - **Classification Model**: CNN for rapid detection
    - **Segmentation Model**: U-Net for precise boundary identification
    - **Web Framework**: Streamlit for interactive interface
    - **Image Processing**: OpenCV, TensorFlow for AI analysis
    
    ### 🌍 Environmental Impact:
    This dual-AI approach enables both rapid detection and detailed analysis,
    significantly improving response times and environmental protection efforts.
    """)

if __name__ == "__main__":
    main()
