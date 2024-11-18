import streamlit as st
from PIL import Image
import torch
import pandas as pd
from datetime import datetime
import sqlite3

# Setup database connection
conn = sqlite3.connect('history/prediction_history.db')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS history
                  (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                   file_name TEXT, 
                   prediction TEXT, 
                   model TEXT, 
                   timestamp TEXT)''')

def load_model(model_name):
    model_path = f"models/{model_name}.pt"
    model = torch.load(model_path, map_location="cpu")
    model.eval()
    return model

def insert_history(file_name, prediction, model, timestamp):
    cursor.execute("INSERT INTO history (file_name, prediction, model, timestamp) VALUES (?, ?, ?, ?)",
                   (file_name, prediction, model, timestamp))
    conn.commit()

def show():
    st.title("Prediksi Penyakit Mata")
    uploaded_file = st.file_uploader("Upload gambar fundus", type=["jpg", "png"])
    model_option = st.selectbox("Pilih model", ["yolov11n", "yolov11s", "yolov11m"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang diupload", use_column_width=True)

        model = load_model(model_option)
        # Asumsi prediksi dan akurasi
        pred_result = "Diabetic Retinopathy"
        accuracy = 0.92

        st.write(f"Hasil Prediksi: {pred_result}")
        st.write(f"Tingkat Akurasi: {accuracy * 100:.2f}%")

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        insert_history(uploaded_file.name, pred_result, model_option, now)