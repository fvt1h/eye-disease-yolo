import streamlit as st
from PIL import Image
import pandas as pd
from datetime import datetime
import sqlite3
import os
from ultralytics import solutions

# Setup database connection
conn = sqlite3.connect('history/prediction_history.db')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS history
                  (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                   file_name TEXT, 
                   prediction TEXT, 
                   model TEXT, 
                   accuracy REAL,
                   timestamp TEXT)''')

class_names = {0: "Normal", 1: "Glaukoma", 2: "Diabetic Retinopathy"}

def insert_history(file_name, prediction, model, accuracy, timestamp):
    cursor.execute("INSERT INTO history (file_name, prediction, model, accuracy, timestamp) VALUES (?, ?, ?, ?, ?)",
                   (file_name, prediction, model, accuracy, timestamp))
    conn.commit()

def show():
    st.title("Halaman Prediksi")
    
    model_option = st.selectbox("Pilih Model", ["YOLOv11n", "YOLOv11s"])
    uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Gambar yang diunggah", use_container_width=True)
        
        try:
            # Pemuatan model menggunakan Ultralytics Solutions
            available_models = {
                "YOLOv11n": "./models/best_yoloV11n.pt",
                "YOLOv11s": "./models/best_yoloV11s.pt",
            }
            
            model_path = available_models.get(model_option)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model {model_option} tidak ditemukan di {model_path}.")
            
            # Inferensi menggunakan solusi Ultralytics
            results = solutions.inference(model=model_path, source=uploaded_file)
            
            # Ekstraksi hasil deteksi
            if len(results) > 0:  # Jika ada hasil deteksi
                pred_result = class_names.get(results[0].cls, "Kelas Tidak Dikenal")
                accuracy = results[0].conf
            else:
                pred_result = "Tidak ada objek terdeteksi"
                accuracy = 0
            
            # Tampilkan hasil prediksi
            st.write(f"**Hasil Prediksi:** {pred_result}")
            st.write(f"**Tingkat Akurasi:** {accuracy * 100:.2f}%")
            
            # Simpan ke dalam database
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            insert_history(uploaded_file.name, pred_result, model_option, accuracy, now)
        
        except FileNotFoundError as e:
            st.error(e)
        except Exception as e:
            st.error("Terjadi kesalahan saat melakukan prediksi.")
            st.error(str(e))

show()
