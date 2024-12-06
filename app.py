import streamlit as st
from pages import home, prediction #, history

st.sidebar.title("Menu")
page = st.sidebar.selectbox("Pilih Halaman", ("Beranda", "Prediksi"))#, "Riwayat"))

if page == "Beranda":
    home.show()
elif page == "Prediksi":
    prediction.show()
#elif page == "Riwayat":
    #history.show()
