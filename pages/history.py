import streamlit as st
import pandas as pd
import sqlite3

def load_history():
    conn = sqlite3.connect('history/prediction_history.db')
    history_data = pd.read_sql_query("SELECT * FROM history", conn)
    conn.close()
    return history_data

def show():
    st.title("Riwayat Prediksi")
    history_data = load_history()
    st.write(history_data)

    if st.button("Hapus semua riwayat"):
        conn = sqlite3.connect('history/prediction_history.db')
        cursor = conn.cursor()
        cursor.execute("DELETE FROM history")
        conn.commit()
        conn.close()
        st.write("Riwayat prediksi telah dihapus.")