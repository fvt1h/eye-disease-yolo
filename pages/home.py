import streamlit as st
from PIL import Image
import os

def show():
    st.title("Deteksi Penyakit Mata")
    st.write("Berikut adalah contoh fundus image dan informasi terkait pencegahan serta pengobatan.")

    images = {
        "Normal": ("assets/sample_images/normal_example.jpg", "assets/explanations/normal.txt"),
        "Glaukoma": ("assets/sample_images/glaucoma_example.jpg", "assets/explanations/glaucoma.txt"),
        "Diabetic Retinopathy": ("assets/sample_images/diabetic_retinopathy_example.jpg", "assets/explanations/diabetic_retinopathy.txt"),
    }

    for condition, (img_path, explanation_path) in images.items():
        st.subheader(condition)
        img = Image.open(img_path)
        st.image(img, caption=f"Gambar contoh {condition}", use_container_width=True)
        with open(explanation_path, "r") as file:
            explanation = file.read()
        st.write(explanation)