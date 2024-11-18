import streamlit as st
from PIL import Image
import torch
import os
import sqlite3
from datetime import datetime

# Fungsi untuk menyimpan riwayat prediksi
def save_to_history(file_name, model_name, result, confidence):
    conn = sqlite3.connect('history/prediction_history.db')
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            image_path TEXT,
            prediction TEXT,
            model_name TEXT,
            confidence REAL,
            timestamp TEXT
        )
    """)
    cursor.execute("""
        INSERT INTO history (file_name, image_path, prediction, model_name, confidence, timestamp) 
        VALUES (?, ?, ?, ?, ?, ?)
    """, (file_name, f"uploads/{file_name}", result, model_name, confidence, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()

# Fungsi untuk melakukan prediksi menggunakan model YOLO
def predict(model_path, image_path):
    model = torch.hub.load('ultralytics/yolov5', model_path)
    results = model(image_path)
    pred = results.pandas().xyxy[0]  # Output prediksi dalam bentuk DataFrame
    if not pred.empty:
        top_pred = pred.iloc[0]
        return top_pred['name'], top_pred['confidence']
    else:
        return "Tidak terdeteksi", 0.0

def show():
    st.title("Prediksi Penyakit Mata")
    st.write("Unggah gambar fundus mata untuk melakukan deteksi.")

    # Pilihan model
    model_options = {
        "YOLOv11n": "yolov5n",
        "YOLOv11s": "yolov5s",
        "YOLOv11m": "yolov5m"
    }
    model_choice = st.selectbox("Pilih Model", list(model_options.keys()))
    model_path = model_options[model_choice]

    # Upload gambar
    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Simpan gambar yang diunggah
        if not os.path.exists("uploads"):
            os.makedirs("uploads")
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Tampilkan gambar yang diunggah
        st.image(Image.open(file_path), caption="Gambar yang diunggah", use_column_width=True)

        # Prediksi
        if st.button("Prediksi"):
            result, confidence = predict(model_path, file_path)
            st.write(f"**Hasil Prediksi:** {result}")
            st.write(f"**Tingkat Akurasi:** {confidence:.2f}")
            
            # Keterangan hasil prediksi
            explanations = {
                "Normal": "Gambar menunjukkan kondisi mata normal.",
                "Glaukoma": "Terdeteksi adanya indikasi Glaukoma. Segera konsultasikan dengan dokter spesialis mata.",
                "Diabetic Retinopathy": "Terdeteksi adanya indikasi Retinopati Diabetik. Periksa lebih lanjut dengan dokter."
            }
            st.write(f"**Keterangan:** {explanations.get(result, 'Tidak ada keterangan.')}")
            
            # Simpan ke riwayat
            save_to_history(uploaded_file.name, model_choice, result, confidence)