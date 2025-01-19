import cv2
import streamlit as st
from PIL import Image
import numpy as np

st.title("Art Digitization Platform")

# Image upload
uploaded_file = st.file_uploader("Upload an Artwork", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Artwork", use_column_width=True)
