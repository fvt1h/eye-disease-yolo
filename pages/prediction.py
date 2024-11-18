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
                   accuracy REAL,
                   timestamp TEXT)''')

def load_model(model_option):
    available_models = {
        "YOLOv11n": "./models/best_yoloV11n.pt",
        "YOLOv11s": "./models/best_yoloV11s.pt",
        # "YOLOv11m": "./models/best_yolov11m.pt"
    }
    
    if model_option not in available_models:
        raise FileNotFoundError(f"Model {model_option} belum tersedia. Silakan pilih model lain.")
    
    model_path = available_models[model_option]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File model untuk {model_option} tidak ditemukan di {model_path}.")
    
    model = torch.load(model_path, map_location="cpu")
    return model

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
            model = load_model(model_option)
            
            image = Image.open(uploaded_file).convert('RGB')
            image_tensor = torch.tensor([torchvision.transforms.ToTensor()(image)])  # Preprocessing sesuai model
            
            with torch.no_grad():
                predictions = model(image_tensor)
                pred_label = predictions[0].get('label', "Tidak Diketahui")  # Ubah sesuai format model
                accuracy = predictions[0].get('confidence', 0.0)
            
            st.write(f"Hasil Prediksi: {pred_label}")
            st.write(f"Tingkat Akurasi: {accuracy * 100:.2f}%")

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            insert_history(uploaded_file.name, pred_label, model_option, accuracy, now)
        
        except FileNotFoundError as e:
            st.error(e)
        except Exception as e:
            st.error("Error dalam melakukan prediksi. Pastikan model dan gambar telah diunggah dengan benar.")
            st.error(str(e))

show()
