import streamlit as st
from ultralytics import YOLO
import os
import tempfile
from datetime import datetime
import sqlite3

# Fungsi untuk menyimpan hasil prediksi ke database
def save_history(file_name, result_path, model_name):
    conn = sqlite3.connect("history/prediction_history.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            result_path TEXT,
            model_name TEXT,
            timestamp TEXT
        )
    """)
    cursor.execute("""
        INSERT INTO history (file_name, result_path, model_name, timestamp)
        VALUES (?, ?, ?, ?)
    """, (file_name, result_path, model_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

# Fungsi untuk menampilkan halaman Prediksi
def show():
    st.title("Halaman Prediksi")
    st.write("Unggah gambar fundus mata untuk mendeteksi penyakit.")

    # Pilihan model
    model_options = {
        "YOLOv11n": "./models/best_yoloV11n.pt",
        "YOLOv11s": "./models/best_yoloV11s.pt",
        #"YOLOv11m": "./models/best_yoloV11m.pt"
    }
    model_name = st.selectbox("Pilih Model", list(model_options.keys()))
    model_path = model_options[model_name]

    # Upload file
    uploaded_file = st.file_uploader("Unggah Gambar (format .jpg atau .png)", type=["jpg", "png"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        st.image(temp_file_path, caption="Gambar yang diunggah", use_column_width=True)

        # Tombol prediksi
        if st.button("Prediksi"):
            st.write("Melakukan prediksi, mohon tunggu...")

            # Load model
            model = YOLO(model_path)

            # Jalankan prediksi
            with tempfile.TemporaryDirectory() as temp_dir:
                result = model.predict(source=temp_file_path, project=temp_dir, name="predict", save=True)

                # Ambil path hasil prediksi
                output_dir = os.path.join(temp_dir, "predict")
                predicted_files = [f for f in os.listdir(output_dir) if f.endswith(".jpg")]

                if predicted_files:
                    result_image_path = os.path.join(output_dir, predicted_files[0])
                    st.image(result_image_path, caption="Hasil Prediksi", use_column_width=True)

                    # Simpan ke riwayat
                    save_history(uploaded_file.name, result_image_path, model_name)
                else:
                    st.error("Gagal memuat hasil prediksi. Silakan coba lagi.")
