import streamlit as st
from pages import home, prediction

st.sidebar.title("Menu")
page = st.sidebar.selectbox("Pilih Halaman", ("Beranda", "Prediksi"))

if page == "Beranda":
    home.show()
elif page == "Prediksi":
    prediction.show()
