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
            # Tambahkan kode prediksi di sini
            st.write(f"Prediksi berhasil menggunakan model {model_option}.")
        except FileNotFoundError as e:
            st.error(e)