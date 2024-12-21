import streamlit as st
from pages import home, prediction

st.sidebar.title("Menu")
page = st.sidebar.selectbox("Pilih Halaman", ("Beranda", "Deteksi"))

if page == "Beranda":
    home.show()
elif page == "Train":
    training.show()
elif page == "Deteksi":
    prediction.show()
