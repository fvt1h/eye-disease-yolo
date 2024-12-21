import streamlit as st
from PIL import Image
import os

def show():
    st.title("Model Yolov11 Detection - Custom Data")
    st.write("Berikut adalah Model yang bisa digunakan untuk melakukan proses deteksi.")

    # Dictionary untuk menyimpan informasi setiap model
    images = {
        "YoloV11n": {
            "result": "assets/sample_images/result_yolov11n.png",
            "train_batch": "assets/sample_images/train_batch_yolov11n.png",
            "val_batch": "assets/sample_images/val_batch2_pred_yolov11n.png",
            "explanation": "assets/explanations/yolov11n.txt",
        },
        "YoloV11s": {
            "result": "assets/sample_images/result_yolov11s.png",
            "train_batch": "assets/sample_images/train_batch_yolov11s.png",
            "val_batch": "assets/sample_images/val_batch_pred_yolov11s.png",
            "explanation": "assets/explanations/yolov11s.txt",
        },
        "YoloV11m": {
            "result": "assets/sample_images/result_yolov11m.png",
            "train_batch": "assets/sample_images/train_batch_yolov11m.png",
            "val_batch": "assets/sample_images/val_batch_pred_yolov11m.png",
            "explanation": "assets/explanations/yolov11m.txt",
        },
    }

    # Loop untuk menampilkan informasi setiap model
    for model_name, data in images.items():
        st.subheader(model_name)
        
        # Gambar hasil deteksi/akurasi
        result_img = Image.open(data["result"])
        st.image(result_img, caption=f"Result/Akurasi - {model_name}", use_container_width=True)
        
        # Gambar train batch
        train_img = Image.open(data["train_batch"])
        st.image(train_img, caption=f"Train Batch - {model_name}", use_container_width=True)
        
        # Gambar val batch
        val_img = Image.open(data["val_batch"])
        st.image(val_img, caption=f"Val Batch - {model_name}", use_container_width=True)
        
        # Penjelasan model
        with open(data["explanation"], "r") as file:
            explanation = file.read()
        st.write(f"Penjelasan Model {model_name}:")
        st.write(explanation)
