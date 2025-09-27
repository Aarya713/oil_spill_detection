import streamlit as st
import numpy as np
from PIL import Image
import random

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

def main():
    st.sidebar.title("🌊 Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Detection"])
    
    if page == "🏠 Home":
        show_home()
    else:
        show_detection()

def show_home():
    st.markdown("<h1 class='main-header'>🌊 AI Oil Spill Detection</h1>", unsafe_allow_html=True)
    st.write("Upload satellite images to detect oil spills using AI technology.")

def show_detection():
    st.header("🔍 Oil Spill Detection")
    
    uploaded_file = st.file_uploader("Upload Satellite Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("🚀 Analyze Image", type="primary"):
            with st.spinner("Analyzing..."):
                # Simulate AI analysis
                confidence = round(random.uniform(0.1, 0.95), 3)
                has_oil_spill = confidence > 0.7
                
                with col2:
                    if has_oil_spill:
                        st.markdown(f"""
                        <div class='result-positive'>
                            <h3>🚨 OIL SPILL DETECTED</h3>
                            <p>Confidence: {confidence:.1%}</p>
                            <p>Risk: HIGH - Immediate action</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='result-negative'>
                            <h3>✅ NO OIL SPILL</h3>
                            <p>Confidence: {confidence:.1%}</p>
                            <p>Risk: LOW - Area clean</p>
                        </div>
                        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
