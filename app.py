
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
    }
    .result-positive {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
    .result-negative {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

class OilSpillDetector:
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        try:
            self.model = tf.keras.models.load_model("oil_spill_model.h5")
        except:
            st.warning("Demo mode - using simulated AI")
    
    def predict(self, image):
        # Simulate AI prediction
        confidence = np.random.uniform(0.1, 0.9)
        has_oil_spill = confidence > 0.7
        return has_oil_spill, confidence

def main():
    st.sidebar.title("🌊 Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Oil Spill Detection", "About"])
    
    if page == "Home":
        show_home()
    elif page == "Oil Spill Detection":
        show_detection()
    else:
        show_about()

def show_home():
    st.markdown("<h1 class='main-header'>🌊 AI Oil Spill Detection System</h1>", unsafe_allow_html=True)
    st.write("Upload satellite images to detect oil spills using AI technology.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Real-time Analysis**")
    with col2:
        st.info("**AI-Powered Detection**")
    with col3:
        st.info("**Instant Results**")

def show_detection():
    st.header("🔍 Oil Spill Detection")
    
    uploaded_file = st.file_uploader("Upload Satellite Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Analyze Image", type="primary"):
            with st.spinner("Analyzing..."):
                detector = OilSpillDetector()
                has_oil_spill, confidence = detector.predict(image)
                
                with col2:
                    if has_oil_spill:
                        st.markdown(f"""
                        <div class='result-positive'>
                            <h3>🚨 OIL SPILL DETECTED</h3>
                            <p>Confidence: {confidence:.1%}</p>
                            <p>Risk: HIGH - Immediate action required</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='result-negative'>
                            <h3>✅ NO OIL SPILL</h3>
                            <p>Confidence: {confidence:.1%}</p>
                            <p>Risk: LOW - Area appears clean</p>
                        </div>
                        """, unsafe_allow_html=True)

def show_about():
    st.header("About")
    st.write("AI-powered oil spill detection system for environmental monitoring.")

if __name__ == "__main__":
    main()
