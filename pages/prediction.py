import streamlit as st
from PIL import Image
import torch
import pandas as pd
from datetime import datetime
import sqlite3
import os
import torchvision.transforms as transforms
from ultralytics import YOLO  # Untuk solusi menggunakan ultralytics

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

def load_model(model_option):
    available_models = {
        "YOLOv11n": "./models/best_yoloV11n.pt",
        "YOLOv11s": "./models/best_yoloV11s.pt",
    }
    
    if model_option not in available_models:
        raise FileNotFoundError(f"Model {model_option} belum tersedia. Silakan pilih model lain.")
    
    model_path = available_models[model_option]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File model untuk {model_option} tidak ditemukan di {model_path}.")
    
    # Solusi: Memuat model menggunakan ultralytics
    return YOLO(model_path)

def insert_history(file_name, prediction, model, accuracy, timestamp):
    cursor.execute("INSERT INTO history (file_name, prediction, model, accuracy, timestamp) VALUES (?, ?, ?, ?, ?)",
                   (file_name, prediction, model, accuracy, timestamp))
    conn.commit()

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Resolusi input model
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Tambahkan dimensi batch

def show():
    st.title("Halaman Prediksi")
    
    model_option = st.selectbox("Pilih Model", ["YOLOv11n", "YOLOv11s"])
    uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Gambar yang diunggah", use_container_width=True)
        
        try:
            model = load_model(model_option)
            image = Image.open(uploaded_file).convert("RGB")
            image_tensor = preprocess_image(image)
            
            # Prediksi dengan model
            results = model(image_tensor)
            output = results[0]

            if len(output.boxes) > 0:
                pred_class_id = int(output.boxes[0].cls)  # Kelas prediksi pertama
                accuracy = float(output.boxes[0].conf)   # Confidence pertama
                pred_result = class_names.get(pred_class_id, "Kelas Tidak Dikenal")
            else:
                pred_result = "Tidak ada objek terdeteksi"
                accuracy = 0
            
            # Tampilkan hasil prediksi
            st.write(f"**Hasil Prediksi:** {pred_result}")
            st.write(f"**Tingkat Akurasi:** {accuracy * 100:.2f}%")
            
            # Simpan hasil ke database
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            insert_history(uploaded_file.name, pred_result, model_option, accuracy, now)
        
        except FileNotFoundError as e:
            st.error(e)
        except Exception as e:
            st.error("Terjadi kesalahan saat melakukan prediksi.")
            st.error(str(e))

show()