import streamlit as st
from pages import home, training, prediction

st.sidebar.title("Menu")
page = st.sidebar.selectbox("Pilih Halaman", ("Beranda", "Train","Deteksi"))

if page == "Beranda":
    home.show()
elif page == "Train":
    training.show()
elif page == "Deteksi":
    prediction.show()
