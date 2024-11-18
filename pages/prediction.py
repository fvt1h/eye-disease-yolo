import streamlit as st
from PIL import Image
import torch
import pandas as pd
from datetime import datetime
import sqlite3
import os

# Setup database connection
conn = sqlite3.connect('history/prediction_history.db')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS history
                  (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                   file_name TEXT, 
                   prediction TEXT, 
                   model TEXT, 
                   timestamp TEXT)''')

def load_model(model_option):
    # Definisikan path model yang tersedia
    available_models = {
        "YOLOv11n": "./models/best_yoloV11n.pt",
        "YOLOv11s": "./models/best_yoloV11s.pt",
        # "YOLOv11m": "./models/best_yolov11m.pt"  # Tambahkan ini ketika model sudah tersedia
    }
    
    # Cek apakah model ada dalam available_models
    if model_option not in available_models:
        raise FileNotFoundError(f"Model {model_option} belum tersedia. Silakan pilih model lain.")
    
    model_path = available_models[model_option]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File model untuk {model_option} tidak ditemukan di {model_path}.")
    
    model = torch.load(model_path, map_location="cpu")
    return model

def insert_history(file_name, prediction, model, timestamp):
    cursor.execute("INSERT INTO history (file_name, prediction, model, timestamp) VALUES (?, ?, ?, ?)",
                   (file_name, prediction, model, timestamp))
    conn.commit()

def show():
    st.title("Halaman Prediksi")
    
    # Pilihan model yang tersedia
    model_option = st.selectbox("Pilih Model", ["YOLOv11n", "YOLOv11s"])  # Tambah "YOLOv11m" nanti jika sudah siap

    uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Gambar yang diunggah", use_column_width=True)
        
        # Load dan jalankan prediksi
        try:
            model = load_model(model_option)
            
            # Konversi gambar untuk prediksi
            image = Image.open(uploaded_file)
            image_tensor = torch.tensor([image]).float()  # Sesuaikan preprocessing sesuai model Anda
            
            # Prediksi menggunakan model
            with torch.no_grad():
                output = model(image_tensor)
                pred_result = output['predicted_class']  # Ubah dengan metode akses sesuai model
                accuracy = output['confidence_score']     # Misal confidence score dari prediksi
            
            # Tampilkan hasil prediksi dan akurasi
            st.write(f"Hasil Prediksi: {pred_result}")
            st.write(f"Tingkat Akurasi: {accuracy * 100:.2f}%")

            # Simpan ke dalam database
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            insert_history(uploaded_file.name, pred_result, model_option, accuracy, now)
        
        except FileNotFoundError as e:
            st.error(e)
        except Exception as e:
            st.error("Error dalam melakukan prediksi. Pastikan model dan gambar telah diunggah dengan benar.")
            st.error(str(e))