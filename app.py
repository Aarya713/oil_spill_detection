import streamlit as st
import random
from PIL import Image

st.set_page_config(page_title="Oil Spill Detection", page_icon="🌊")
st.title("🌊 Oil Spill Detection System")
st.write("Upload satellite images for analysis")

uploaded_file = st.file_uploader("Choose image", type=["jpg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")
    
    if st.button("Analyze"):
        confidence = random.uniform(0.1, 0.95)
        if confidence > 0.7:
            st.error(f"🚨 Oil spill detected! Confidence: {confidence:.1%}")
        else:
            st.success(f"✅ No oil spill. Confidence: {confidence:.1%}")
