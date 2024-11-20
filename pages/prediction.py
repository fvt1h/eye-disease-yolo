import streamlit as st
from PIL import Image
import os
from datetime import datetime
import sqlite3
from ultralytics import YOLO

# Fungsi untuk menyimpan riwayat prediksi
def save_to_history(file_name, result_label, model_used, confidence, image_path):
    conn = sqlite3.connect('history/prediction_history.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_name TEXT,
                        image_path TEXT,
                        result_label TEXT,
                        model_used TEXT,
                        confidence REAL,
                        prediction_time TEXT)''')
    prediction_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO history (file_name, image_path, result_label, model_used, confidence, prediction_time) VALUES (?, ?, ?, ?, ?, ?)",
                   (file_name, image_path, result_label, model_used, confidence, prediction_time))
    conn.commit()
    conn.close()

def show():
    st.title("Prediksi Penyakit Mata")

    # Pilih model YOLO
    models = {
        "YOLOv11n": "best_yoloV11n.pt",
        "YOLOv11s": "best_yoloV11s.pt",
        "YOLOv11m": "best_yoloV11m.pt",
    }
    model_name = st.selectbox("Pilih Model", list(models.keys()))
    selected_model_path = models[model_name]

    # Upload gambar
    uploaded_file = st.file_uploader("Unggah Gambar Fundus", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)

        # Simpan sementara gambar yang diunggah
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        image.save(temp_path)

        # Prediksi jika tombol diklik
        if st.button("Prediksi"):
            # Load model dan lakukan prediksi
            model = YOLO(selected_model_path)
            results = model.predict(source=temp_path, save=True)

            # Ambil hasil prediksi
            result = results[0]
            result_label = result.names[result.boxes.data[0][-1].int()]
            confidence = result.boxes.data[0][-2].item() * 100

            # Tampilkan hasil
            st.write("Hasil Prediksi:")
            st.image(result.plot(), caption=f"Hasil Prediksi {result_label}", use_column_width=True)
            st.write(f"Nama file: {uploaded_file.name}")
            st.write(f"Hasil: {result_label}")
            st.write(f"Tingkat akurasi: {confidence:.2f}%")

            # Simpan riwayat prediksi
            save_to_history(uploaded_file.name, result_label, model_name, confidence, temp_path)