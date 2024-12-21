import streamlit as st
from PIL import Image
import nbformat

def show():
    st.title("Model Yolov11 Detection - Custom Data")
    
    st.header("Kode Training Model")
    st.write("Berikut adalah potongan kode dari notebook yang digunakan untuk melatih model, lengkap dengan penjelasannya.")

    # Header untuk kode training
    st.header("Kode Training Model")
    st.write("Berikut adalah potongan kode dari training model, lengkap dengan penjelasannya.")

    # Kode dan penjelasan
    training_codes = [
        {
            "code": """
# Mengambil dataset yang ada pada github
!git clone https://github.com/fvt1h/dataset.git
""",
            "explanation": "Kode ini digunakan untuk dataset yang sudah dibuat"
        },
        {
            "code": """
# Menginstall ultralytics
!pip install ultralytics
""",
            "explanation": "Kode ini digunakan untuk install library ultralytics dimana didalamnya terdapat model yolo"
        },
        {
            "code": """
# Mengimpor library torch
import torch
torch.cuda.is_available()
""",
            "explanation": "Kode ini digunakan untuk mengimpor library torch dan melakukan pengecekan apakah sudah ada atau belum"
        },
        {
            "code": """
# Mengimport library ultralytics
import ultralytics
ultralytics.checks()
""",
            "explanation": "Kode ini digunakan untuk import library ultralytics yang sudah diinstall tadi melakukan pengecekan apakah library tersebut sudah ada atau belum "
        },
        {
            "code": """
# Mengimport Yolo dan Python image display
from ultralytics import YOLO
""",
            "explanation": "Bagian ini digunakan untuk mengimport model yolo dari library ultralytics"
        },
        {
            "code": """
# Melatih model
!yolo task=detect mode=train data=/content/dataset/dataset.yaml model=yolov11m.pt imgsz=640 epochs=100
""",
            "explanation": "Bagian ini melakukan proses training model YOLOv11, sebagai contohnya adalah yolov11m dan mentrainingnya dengan dataset yang telah dimuat sesuai dengan path yang ada pada dataset.yaml yang akan di training sebanyak 100 epoch."
        },
        {
            "code": """
# Menyimpan model
model.save('best_yoloV11m.pt')
""",
            "explanation": "Setelah proses training selesai, model disimpan ke file bernama `yolov11_custom.pt`."
        }
    ]

    # Loop untuk menampilkan kode dan penjelasannya
    for block in training_codes:
        st.code(block["code"], language="python")  # Menampilkan kode
        st.write(f"**Penjelasan:** {block['explanation']}")  # Menampilkan penjelasan
                
    st.write("Berikut adalah Model yang bisa digunakan untuk melakukan proses deteksi.")
    
    images = {
        "YoloV11n": {
            "result": "assets/sample_images/results_yolov11n.png",
            "train_batch": "assets/sample_images/train_batch_yolov11n.jpg",
            "val_batch": "assets/sample_images/val_batch2_pred_yolov11n.jpg",
            "explanation": "assets/explanations/yolov11n.txt",
        },
        "YoloV11s": {
            "result": "assets/sample_images/results_yolov11s.png",
            "train_batch": "assets/sample_images/train_batch_yolov11s.jpg",
            "val_batch": "assets/sample_images/val_batch2_pred_yolov11s.jpg",
            "explanation": "assets/explanations/yolov11s.txt",
        },
        "YoloV11m": {
            "result": "assets/sample_images/results_yolov11m.png",
            "train_batch": "assets/sample_images/train_batch_yolov11m.jpg",
            "val_batch": "assets/sample_images/val_batch2_pred_yolov11m.jpg",
            "explanation": "assets/explanations/yolov11m.txt",
        },
    }

    # Loop untuk menampilkan informasi setiap model
    for model_name, data in images.items():
        st.subheader(model_name)
        
        # Membuat kolom untuk gambar
        cols = st.columns(3)
        
        # Gambar result/akurasi
        with cols[0]:
            result_img = Image.open(data["result"])
            st.image(result_img, caption=f"Result/Akurasi", use_container_width=True)
        
        # Gambar train batch
        with cols[1]:
            train_img = Image.open(data["train_batch"])
            st.image(train_img, caption=f"Train Batch", use_container_width=True)
        
        # Gambar val batch
        with cols[2]:
            val_img = Image.open(data["val_batch"])
            st.image(val_img, caption=f"Val Batch", use_container_width=True)
        
        # Penjelasan model di bawah gambar
        st.write(f"Penjelasan Model {model_name}:")
        with open(data["explanation"], "r") as file:
            explanation = file.read()
        st.write(explanation)
