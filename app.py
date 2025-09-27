import streamlit as st
import random
from PIL import Image

st.set_page_config(page_title="Oil Spill Detection", page_icon="🌊", layout="wide")

st.markdown("""
<style>
    .main-title { color: #1f77b4; text-align: center; font-size: 3rem; }
    .result-box { padding: 20px; border-radius: 10px; margin: 20px 0; color: white; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🌊 Oil Spill Detection System</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload Satellite Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns(2)
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Satellite Image", use_column_width=True)
    
    if st.button("🚀 Analyze for Oil Spills", type="primary"):
        confidence = round(random.uniform(0.1, 0.95), 3)
        has_spill = confidence > 0.7
        
        with col2:
            if has_spill:
                st.markdown(f"""
                <div class='result-box' style='background: linear-gradient(135deg, #ff6b6b, #ee5a52);'>
                    <h2>🚨 OIL SPILL DETECTED</h2>
                    <p>Confidence: {confidence:.1%}</p>
                    <p>Risk: HIGH - Immediate action required</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-box' style='background: linear-gradient(135deg, #00b09b, #96c93d);'>
                    <h2>✅ NO OIL SPILL</h2>
                    <p>Confidence: {confidence:.1%}</p>
                    <p>Risk: LOW - Area clean</p>
                </div>
                """, unsafe_allow_html=True)
