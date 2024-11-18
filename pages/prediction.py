import streamlit as st
from PIL import Image
import torch
import numpy as np
import pandas as pd
import cv2
import sqlite3
from datetime import datetime
import os

# Fungsi untuk konversi gambar buffer ke format cv2
def create_opencv_image_from_stringio(img_stream):
    img_stream.seek(0)
    img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# Fungsi untuk load model
def load_model(model_option):
    available_models = {
        "YOLOv11n": "./models/best_yoloV11n.pt",
        "YOLOv11s": "./models/best_yoloV11s.pt",
    }
    model_path = available_models.get(model_option)
    if model_path:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        return model
    else:
        st.error("Model tidak ditemukan!")
        return None

# Fungsi prediksi
def predict(image, model):
    results = model(image)
    results.render()
    return results.imgs[0], results.pandas().xyxy[0]

# Fungsi utama untuk halaman prediksi
def show():
    st.title("Halaman Prediksi")

    model_option = st.selectbox("Pilih Model", ["YOLOv11n", "YOLOv11s"])
    uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Gambar yang diunggah", use_container_width=True)
        
        image = create_opencv_image_from_stringio(uploaded_file)
        model = load_model(model_option)
        
        if model:
            pred_image, pred_data = predict(image, model)
            st.image(pred_image, caption="Hasil Prediksi", use_column_width=True)
            
            st.write("Detail Prediksi:")
            st.write(pred_data)
            
            # Simpan hasil ke database
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn = sqlite3.connect('history/prediction_history.db')
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS history
                              (id INTEGER PRIMARY KEY AUTOINCREMENT,
                               file_name TEXT, 
                               prediction TEXT, 
                               model TEXT, 
                               accuracy REAL, 
                               timestamp TEXT)''')
            
            pred_data_str = str(pred_data)
            cursor.execute("INSERT INTO history (file_name, prediction, model, accuracy, timestamp) VALUES (?, ?, ?, ?, ?)",
                           (uploaded_file.name, pred_data_str, model_option, None, now))
            conn.commit()
            conn.close()
            st.success("Hasil prediksi telah disimpan.")