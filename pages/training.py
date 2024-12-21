import streamlit as st
from PIL import Image
import os

def show():
    st.title("Model Yolov11 Detection - Custom Data")
    st.write("Berikut adalah Model yang bisa digunakan untuk melakukan proses deteksi.")

    # Dictionary untuk menyimpan informasi setiap model
    images = {
        "YoloV11n": {
            "result": "assets/sample_images/results_yolov11n.png",
            "train_batch": "assets/sample_images/train_batch_yolov11n.jpg",
            "val_batch": "assets/sample_images/val_batch2_pred_yolov11n.jpg",
            "explanation": "assets/explanations/yolov11n.txt",
        },
        "YoloV11s": {
            "result": "assets/sample_images/results_yolov11s.png",
            "train_batch": "assets/sample_images/train_batch_yolov11s.jpg",
            "val_batch": "assets/sample_images/val_batch2_pred_yolov11s.jpg",
            "explanation": "assets/explanations/yolov11s.txt",
        },
        # "YoloV11m": {
        #     "result": "assets/sample_images/results_yolov11m.png",
        #     "train_batch": "assets/sample_images/train_batch_yolov11m.jpg",
        #     "val_batch": "assets/sample_images/val_batch2_pred_yolov11m.jpg",
        #     "explanation": "assets/explanations/yolov11m.txt",
        # },
    }

    # Loop untuk menampilkan informasi setiap model
    for model_name, data in images.items():
        st.subheader(model_name)
        
        # Membuat kolom untuk gambar
        cols = st.columns(3)
        
        # Gambar result/akurasi
        with cols[0]:
            result_img = Image.open(data["result"])
            st.image(result_img, caption=f"Result/Akurasi", use_container_width=True)
        
        # Gambar train batch
        with cols[1]:
            train_img = Image.open(data["train_batch"])
            st.image(train_img, caption=f"Train Batch", use_container_width=True)
        
        # Gambar val batch
        with cols[2]:
            val_img = Image.open(data["val_batch"])
            st.image(val_img, caption=f"Val Batch", use_container_width=True)
        
        # Penjelasan model di bawah gambar
        st.write(f"Penjelasan Model {model_name}:")
        with open(data["explanation"], "r") as file:
            explanation = file.read()
        st.write(explanation)
