import streamlit as st
from PIL import Image
import torch
import pandas as pd
from datetime import datetime
import sqlite3
import os
import torchvision.transforms as transforms

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

def load_model(model_option):
    available_models = {
        "YOLOv11n": "./models/best_yoloV11n.pt",
        "YOLOv11s": "./models/best_yoloV11s.pt",
        # "YOLOv11m": "./models/best_yoloV11m.pt"  # Uncomment jika sudah ada model YOLOv11m
    }
    
    if model_option not in available_models:
        raise FileNotFoundError(f"Model {model_option} belum tersedia. Silakan pilih model lain.")
    
    model_path = available_models[model_option]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File model untuk {model_option} tidak ditemukan di {model_path}.")
    
    model = torch.load(model_path, map_location="cpu")
    model.eval()  # Pastikan model dalam mode evaluasi
    return model

def insert_history(file_name, prediction, model, accuracy, timestamp):
    cursor.execute("INSERT INTO history (file_name, prediction, model, accuracy, timestamp) VALUES (?, ?, ?, ?, ?)",
                   (file_name, prediction, model, accuracy, timestamp))
    conn.commit()

def preprocess_image(image):
    # Transformasi gambar (sesuaikan sesuai dengan model YOLO yang Anda gunakan)
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Sesuaikan ukuran ini dengan input model
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # Tambahkan dimensi batch

def show():
    st.title("Halaman Prediksi")
    
    model_option = st.selectbox("Pilih Model", ["YOLOv11n", "YOLOv11s"])  # Tambahkan YOLOv11m jika sudah siap

    uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Gambar yang diunggah", use_container_width=True)
        
        try:
            model = load_model(model_option)
            image = Image.open(uploaded_file)
            image_tensor = preprocess_image(image)
            
            # Prediksi dengan model
            with torch.no_grad():
                output = model(image_tensor)
                # Asumsi output model mengandung 'labels' dan 'scores'
                pred_result = output[0]['labels'][0].item()  # Ganti dengan akses yang sesuai
                accuracy = output[0]['scores'][0].item()
            
            # Tampilkan hasil prediksi
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

show()