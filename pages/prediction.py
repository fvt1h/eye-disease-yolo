import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import sqlite3
from datetime import datetime

# Fungsi untuk menyimpan riwayat prediksi
def save_history(file_name, result_image_path, model_name, date_time):
    conn = sqlite3.connect('history/prediction_history.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS history (
        file_name TEXT,
        result_image_path TEXT,
        model_name TEXT,
        date_time TEXT
    )''')
    conn.commit()
    cursor.execute("INSERT INTO history VALUES (?, ?, ?, ?)",
                   (file_name, result_image_path, model_name, date_time))
    conn.commit()
    conn.close()

def show():
    st.title("Prediksi Penyakit Mata")
    st.write("Upload gambar fundus dan pilih model untuk mendeteksi penyakit mata.")

    # Pilihan model
    models = {
        "YOLOv11n": ".models/best_yoloV11n.pt",
        "YOLOv11s": ".models/best_yoloV11s.pt",
        #"YOLOv11m": ".models/best_yoloV11m.pt"
    }
    model_name = st.selectbox("Pilih Model", list(models.keys()))
    
    # Upload gambar
    uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        file_path = f"uploads/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.image(file_path, caption="Gambar yang diupload", use_column_width=True)
        
        if st.button("Deteksi"):
            # Memuat model YOLO
            model_path = models[model_name]
            model = YOLO(model_path)
            
            # Melakukan prediksi
            st.write("Melakukan prediksi...")
            result = model.predict(source=file_path, show=True, save=True)
            
            # Mendapatkan path hasil prediksi
            result_image_path = os.path.join("runs", "detect", "predict", uploaded_file.name)
            st.image(result_image_path, caption="Hasil Prediksi", use_column_width=True)
            
            # Simpan ke riwayat
            save_history(uploaded_file.name, result_image_path, model_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            st.success("Prediksi selesai dan disimpan ke riwayat.")
