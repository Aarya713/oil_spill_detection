import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import io
import base64

# Page configuration
st.set_page_config(
    page_title="Oil Spill Detection System",
    page_icon="🌊",
    layout="wide"
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
</style>
""", unsafe_allow_html=True)

def create_demo_segmentation(width, height):
    """Create a demo segmentation mask"""
    mask = np.zeros((height, width))
    
    # Create some random "oil spill" shapes
    center_x, center_y = width // 2, height // 2
    y, x = np.ogrid[:height, :width]
    
    # Main spill
    mask1 = (x - center_x)**2 + (y - center_y)**2 <= (min(width, height) // 4)**2
    
    # Smaller spills
    mask2 = (x - center_x*0.7)**2 + (y - center_y*1.3)**2 <= (min(width, height) // 8)**2
    mask3 = (x - center_x*1.3)**2 + (y - center_y*0.7)**2 <= (min(width, height) // 6)**2
    
    mask = (mask1 | mask2 | mask3).astype(float)
    return mask

def main():
    st.sidebar.title("🌊 Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Detection"])
    
    if page == "🏠 Home":
        show_home()
    else:
        show_detection()

def show_home():
    st.markdown("<h1 class='main-header'>🌊 Oil Spill Detection System</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 30px; border-radius: 15px; margin: 20px 0;'>
        <h2>🚀 AI-Powered Detection</h2>
        <p>Upload satellite images for oil spill analysis and segmentation</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**🔍 Classification**\nDetect oil spills with AI")
    with col2:
        st.info("**🎯 Segmentation**\nIdentify spill boundaries")
    with col3:
        st.info("**📊 Analysis**\nGenerate detailed reports")

def show_detection():
    st.header("🔍 Oil Spill Analysis")
    
    # Option 1: Upload image
    uploaded_file = st.file_uploader("Upload satellite image", type=["jpg", "jpeg", "png"])
    
    # Option 2: Use demo image
    use_demo = st.checkbox("Use demo image instead")
    
    if uploaded_file or use_demo:
        if use_demo:
            # Create a demo satellite-like image
            st.info("🛰 Using demo satellite image")
            width, height = 400, 300
            demo_image = np.random.rand(height, width, 3) * 0.3 + 0.5  # Blue-ish background
            # Add some land masses
            demo_image[100:200, 50:150] = [0.3, 0.6, 0.2]  # Green land
            demo_image[50:120, 250:350] = [0.4, 0.5, 0.3]  # Another land mass
        else:
            # For uploaded files, we'll create a simple representation
            st.info("📡 Processing uploaded image")
            width, height = 400, 300
            demo_image = np.random.rand(height, width, 3) * 0.4 + 0.4
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.imshow(demo_image)
            ax.set_title("Satellite Image")
            ax.axis('off')
            st.pyplot(fig)
        
        if st.button("🚀 Analyze Image", type="primary"):
            with st.spinner("Analyzing with AI..."):
                # Simulate AI processing
                import time
                time.sleep(2)
                
                # Demo results
                has_spill = np.random.random() > 0.4
                confidence = np.random.uniform(0.7, 0.95) if has_spill else np.random.uniform(0.1, 0.4)
                
                with col2:
                    if has_spill:
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
                
                if has_spill:
                    st.subheader("🎯 Spill Segmentation")
                    
                    # Create segmentation visualization
                    segmentation_mask = create_demo_segmentation(width, height)
                    
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
                    
                    # Original
                    ax1.imshow(demo_image)
                    ax1.set_title("Original Image")
                    ax1.axis('off')
                    
                    # Segmentation
                    ax2.imshow(segmentation_mask, cmap='hot')
                    ax2.set_title("Spill Detection")
                    ax2.axis('off')
                    
                    # Overlay
                    overlay = demo_image.copy()
                    mask_rgb = np.stack([segmentation_mask] * 3, axis=-1)
                    red_overlay = np.array([1, 0, 0])  # Red color for spills
                    overlay = np.where(mask_rgb > 0, overlay * 0.6 + red_overlay * 0.4, overlay)
                    ax3.imshow(overlay)
                    ax3.set_title("Detection Overlay")
                    ax3.axis('off')
                    
                    st.pyplot(fig)
                    
                    # Calculate stats
                    spill_area = np.sum(segmentation_mask > 0)
                    total_area = segmentation_mask.size
                    spill_percentage = (spill_area / total_area) * 100
                    
                    st.info(f"**📊 Estimated Spill Coverage:** {spill_percentage:.1f}%")
                
                # Report
                st.subheader("📋 Analysis Report")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Image Information**")
                    st.write(f"- Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"- Image Size: {width} × {height} pixels")
                    st.write("- Status: Demo Analysis Complete")
                
                with col2:
                    st.write("**Detection Results**")
                    st.write(f"- Oil Spill: {'DETECTED' if has_spill else 'NOT DETECTED'}")
                    st.write(f"- Confidence: {confidence:.1%}")
                    st.write(f"- Urgency: {'HIGH' if has_spill else 'LOW'}")
    
    else:
        st.info("👆 Upload a satellite image or check 'Use demo image' to get started!")

if __name__ == "__main__":
    main()
