import streamlit as st
from ultralytics import YOLO
import os
import tempfile
from datetime import datetime
import sqlite3

# Fungsi untuk menampilkan halaman Deteksi
def show():
    st.title("Halaman Deteksi")
    st.write("Unggah gambar fundus mata untuk mendeteksi penyakit.")

    # Pilihan model
    model_options = {
        "YOLOv11n": "./models/best_yoloV11n.pt",
        "YOLOv11s": "./models/best_yoloV11s.pt",
        "YOLOv11m": "./models/best_yoloV11m.pt",
        #"YOLOv11l": "./models/best_yoloV11l.pt",
        #"YOLOv11x": "./models/best_yoloV11x.pt",
    }
    model_name = st.selectbox("Pilih Model", list(model_options.keys()))
    model_path = model_options[model_name]

    # Upload file
    uploaded_file = st.file_uploader("Unggah Gambar (format .jpg atau .png)", type=["jpg", "png"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        st.image(temp_file_path, caption="Gambar yang diunggah", use_container_width=True)

        # Tombol Deteksi
        if st.button("Deteksi"):
            st.write("Melakukan Deteksi, mohon tunggu...")

            # Load model
            model = YOLO(model_path)

            # Jalankan Deteksi
            with tempfile.TemporaryDirectory() as temp_dir:
                result = model.predict(source=temp_file_path, project=temp_dir, name="predict", save=True)

                # Ambil path hasil Deteksi
                output_dir = os.path.join(temp_dir, "predict")
                predicted_files = [f for f in os.listdir(output_dir) if f.endswith(".jpg")]

                if predicted_files:
                    result_image_path = os.path.join(output_dir, predicted_files[0])
                    st.image(result_image_path, caption="Hasil Deteksi", use_column_width=True)
                    
                else:
                    st.error("Gagal memuat hasil Deteksi. Silakan coba lagi.")
