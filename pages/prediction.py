import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import sqlite3
from datetime import datetime

# Fungsi untuk menyimpan riwayat prediksi ke database
def save_to_history(file_name, prediction_result, model_name):
    conn = sqlite3.connect('history/prediction_history.db')
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            image_path TEXT,
            prediction_result TEXT,
            model_name TEXT,
            prediction_time TEXT
        )
    """)
    prediction_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO history (file_name, image_path, prediction_result, model_name, prediction_time) VALUES (?, ?, ?, ?, ?)", 
                   (file_name, f"uploads/{file_name}", prediction_result, model_name, prediction_time))
    conn.commit()
    conn.close()

# Fungsi untuk melakukan prediksi
def predict_image(model_path, image_file):
    model = YOLO(model_path)
    results = model.predict(source=image_file, show=False, save=True)
    result_labels = [result.names[pred.cls] for pred in results[0].boxes]  # Ambil label hasil prediksi
    return ', '.join(set(result_labels))  # Gabungkan hasil prediksi unik

def show():
    st.title("Prediksi Penyakit Mata")

    # Pilihan model
    models = {
        "YOLOv11n": "models/best_yoloV11n.pt",
        "YOLOv11s": "models/best_yoloV11s.pt",
        #"YOLOv11m": "models/best_yolov11m.pt"
    }
    selected_model = st.selectbox("Pilih Model", list(models.keys()))

    # Upload gambar
    uploaded_file = st.file_uploader("Unggah gambar fundus mata", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Simpan gambar sementara
        save_path = os.path.join("uploads", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.image(save_path, caption="Gambar yang diunggah", use_column_width=True)

        # Prediksi
        if st.button("Prediksi"):
            with st.spinner("Model sedang melakukan prediksi..."):
                prediction_result = predict_image(models[selected_model], save_path)

            st.success("Prediksi selesai!")
            st.write(f"**Hasil Prediksi:** {prediction_result}")
            st.write(f"**Nama File:** {uploaded_file.name}")
            st.write(f"**Model yang Digunakan:** {selected_model}")

            # Simpan ke riwayat
            save_to_history(uploaded_file.name, prediction_result, selected_model)