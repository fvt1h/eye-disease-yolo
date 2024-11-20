import streamlit as st
from ultralytics import YOLO
import sqlite3
from PIL import Image
import os
from datetime import datetime

def save_prediction_to_history(filename, image_path, prediction, model_name):
    conn = sqlite3.connect('history/prediction_history.db')
    cursor = conn.cursor()
    
    # Pastikan tabel history ada
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            image_path TEXT, 
            prediction TEXT,
            model_name TEXT,
            prediction_time DATETIME
        )
    ''')
    
    # Simpan prediksi ke database
    cursor.execute('''
        INSERT INTO history 
        (filename, image_path, prediction, model_name, prediction_time) 
        VALUES (?, ?, ?, ?, ?)
    ''', (filename, image_path, prediction, model_name, datetime.now()))
    
    conn.commit()
    conn.close()

def show():
    st.title("Prediksi Penyakit Mata")
    
    # Pilih model
    model_options = {
        "YOLOv11n": "./models/best_yoloV11n.pt",
        "YOLOv11s": "./models/best_yoloV11s.pt", 
        #"YOLOv11m": "./models/best_yoloV11m.pt"
    }
    selected_model = st.selectbox("Pilih Model", list(model_options.keys()))
    
    # Upload gambar
    uploaded_file = st.file_uploader("Unggah Gambar Fundus", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Simpan gambar sementara
        uploads_path = os.path.join("uploads", uploaded_file.name)
        os.makedirs("uploads", exist_ok=True)
        with open(uploads_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Tampilkan gambar yang diunggah
        st.image(uploaded_file, caption="Gambar yang Diunggah", use_column_width=True)
        
        # Tombol prediksi
        if st.button("Lakukan Prediksi"):
            # Muat model
            model = YOLO(model_options[selected_model])
            
            # Lakukan prediksi
            results = model.predict(source=uploads_path, show=True, save=True)
            
            # Dapatkan prediksi
            prediction = results[0].probs.top1
            
            # Simpan ke dalam riwayat
            save_prediction_to_history(
                uploaded_file.name, 
                uploads_path, 
                prediction, 
                selected_model
            )
            
            # Tampilkan hasil prediksi
            st.success(f"Hasil Prediksi: {prediction}")