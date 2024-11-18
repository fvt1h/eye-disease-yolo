import streamlit as st
from PIL import Image
import torch
import sqlite3
import os
from datetime import datetime

# Fungsi untuk menyimpan hasil prediksi ke database
def save_prediction_to_history(filename, image, prediction, model_name, accuracy):
    conn = sqlite3.connect('history/prediction_history.db')
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            image_path TEXT,
            prediction TEXT,
            model_name TEXT,
            accuracy REAL,
            timestamp TEXT
        )
    """)
    image_path = os.path.join("history/images", filename)
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    image.save(image_path)
    cursor.execute("""
        INSERT INTO history (filename, image_path, prediction, model_name, accuracy, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (filename, image_path, prediction, model_name, accuracy, datetime.now()))
    conn.commit()
    conn.close()

# Fungsi untuk melakukan prediksi
def predict(image, model):
    results = model(image)
    predictions = results.pandas().xyxy[0]
    if len(predictions) > 0:
        prediction = predictions.iloc[0]["name"]
        confidence = predictions.iloc[0]["confidence"]
    else:
        prediction = "Tidak ada deteksi"
        confidence = 0.0
    return prediction, confidence

def show():
    st.title("Prediksi Penyakit Mata")
    
    # Pilih model
    st.subheader("Pilih Model YOLOv11")
    model_option = st.selectbox("Pilih model untuk digunakan:", ["YOLOv11n", "YOLOv11s", "YOLOv11m"])
    model_path = {
        "YOLOv11n": "models/yolov11n.pt",
        "YOLOv11s": "models/yolov11s.pt",
        "YOLOv11m": "models/yolov11m.pt",
    }[model_option]
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

    # Unggah gambar
    uploaded_file = st.file_uploader("Unggah gambar fundus", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Tampilkan gambar yang diunggah
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)
        
        # Lakukan prediksi
        if st.button("Prediksi"):
            prediction, confidence = predict(image, model)
            confidence_percentage = confidence * 100

            # Tampilkan hasil prediksi
            st.write("### Hasil Prediksi")
            st.write(f"**Nama File:** {uploaded_file.name}")
            st.write(f"**Hasil Prediksi:** {prediction}")
            st.write(f"**Tingkat Akurasi:** {confidence_percentage:.2f}%")

            # Simpan hasil ke riwayat
            save_prediction_to_history(uploaded_file.name, image, prediction, model_option, confidence_percentage)
            st.success("Hasil prediksi telah disimpan ke riwayat.")
