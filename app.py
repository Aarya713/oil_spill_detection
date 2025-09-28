import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from datetime import datetime
import gdown
import os
import random

# Download models from Google Drive
def download_models():
    """Download models from Google Drive"""
    classifier_id = "1oddiVJOirUYGhUnXHGqOrK8W1N7cUxIi"
    segmenter_id = "1EFYBAaMoLY6SCPO5fqyxPP870fnLbVtr"
    
    if not os.path.exists('classifier_model.h5'):
        try:
            st.sidebar.info("📥 Downloading classifier model...")
            url = f'https://drive.google.com/uc?id={classifier_id}'
            gdown.download(url, 'classifier_model.h5', quiet=False)
            if os.path.exists('classifier_model.h5'):
                st.sidebar.success("✅ Classifier model downloaded!")
        except Exception as e:
            st.sidebar.info("⚠️ Classifier download skipped - using demo mode")
    
    if not os.path.exists('segmentation_model.h5'):
        try:
            st.sidebar.info("📥 Downloading segmentation model...")
            url = f'https://drive.google.com/uc?id={segmenter_id}'
            gdown.download(url, 'segmentation_model.h5', quiet=False)
            if os.path.exists('segmentation_model.h5'):
                st.sidebar.success("✅ Segmentation model downloaded!")
        except Exception as e:
            st.sidebar.info("⚠️ Segmentation download skipped - using demo mode")

# Page configuration
st.set_page_config(
    page_title="Oil Spill Detection System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
        self.demo_mode = True
        st.sidebar.info("🔧 Running in Demo Mode")
    
    def classify_image(self, image):
        """Demo classification - simulates AI detection"""
        # Simple demo logic
        img_array = np.array(image)
        
        # Simulate AI decision
        has_oil_spill = random.random() > 0.3  # 70% chance of detecting spill
        confidence = random.uniform(0.7, 0.95) if has_oil_spill else random.uniform(0.1, 0.4)
        
        return has_oil_spill, confidence
    
    def segment_image(self, image):
        """Demo segmentation - creates simulated oil spill areas"""
        img_array = np.array(image)
        height, width = img_array.shape[0], img_array.shape[1]
        
        # Create a simulated segmentation mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Generate random oil spill shapes
        num_spills = random.randint(1, 3)
        for _ in range(num_spills):
            center_x = random.randint(width//4, 3*width//4)
            center_y = random.randint(height//4, 3*height//4)
            radius = random.randint(min(width, height)//8, min(width, height)//4)
            
            # Create circular mask
            y, x = np.ogrid[:height, :width]
            dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            spill_area = dist_from_center <= radius
            mask[spill_area] = 255
        
        return mask, "Demo segmentation completed"

def main():
    # Download models
    download_models()
    
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
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 30px; border-radius: 15px; margin: 20px 0;'>
        <h2>🚀 Advanced Detection System</h2>
        <p>Complete AI solution with both classification and segmentation for comprehensive oil spill monitoring.</p>
    </div>
    """, unsafe_allow_html=True)
    
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
    """Live detection interface"""
    st.header("🔍 Advanced Oil Spill Analysis")
    st.info("🔧 **Demo Mode** - Upload any image to see how the system works!")
    
    uploaded_file = st.file_uploader(
        "Choose satellite image (JPG, PNG, JPEG)", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="📡 Uploaded Satellite Image", use_column_width=True)
        
        if st.button("🚀 Start AI Analysis", type="primary", use_container_width=True):
            with st.spinner("🔄 Running comprehensive AI analysis..."):
                detector = OilSpillDetector()
                
                # Run classification
                has_oil_spill, confidence = detector.classify_image(image)
                
                # Display results
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
                
                # Run segmentation if oil spill detected
                if has_oil_spill:
                    st.subheader("🎯 Oil Spill Segmentation")
                    segmentation_mask, status = detector.segment_image(image)
                    
                    if segmentation_mask is not None:
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
                    """)
                else:
                    st.success("""
                    ✅ **RECOMMENDED ACTIONS:**
                    - Continue routine monitoring
                    - Maintain environmental surveillance
                    - Schedule next inspection
                    """)

def show_about():
    """About page"""
    st.header("📊 About This System")
    
    st.write("""
    ## 🌊 Oil Spill Detection System
    
    Advanced web application for oil spill monitoring and analysis.
    
    ### 🎯 Technology Stack:
    - **Streamlit**: Web interface
    - **Image Processing**: OpenCV, PIL
    - **Visualization**: Matplotlib
    - **Google Drive**: Model storage
    
    ### 🌍 Environmental Impact:
    This system enables rapid detection and detailed analysis,
    significantly improving response times and environmental protection efforts.
    """)

if __name__ == "__main__":
    main()
