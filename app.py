import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model
from io import BytesIO

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Harga Tanah", layout="wide")

st.title("🌲 Aplikasi Prediksi Harga Tanah (Random Forest)")
st.markdown("Upload file Excel data baru Anda di bawah ini untuk mendapatkan prediksi harga.")

# --- 2. KONFIGURASI MODEL (PATH) ---
# Masukkan path model Anda di sini (tanpa akhiran .pkl)
# Pastikan file .pkl ada di lokasi ini
MODEL_PATH = 'D:/Data Projek/Projek(IDW)/virtual/projek_random_forest/proyek_ml_harga_rf_747'

# --- 3. UPLOAD FILE DATA ---
st.sidebar.header("Panel Input")
uploaded_file = st.sidebar.file_uploader("Upload Data Baru (.xlsx)", type=["xlsx"])

# --- 4. LOGIKA UTAMA ---
if uploaded_file is not None:
    try:
        # A. Membaca Data yang Diupload
        df_baru = pd.read_excel(uploaded_file)
        
        st.subheader("1. Data Awal")
        st.dataframe(df_baru.head())
        
        # Tombol Eksekusi
        if st.button("Mulai Prediksi 🚀"):
            with st.spinner("Sedang memuat model dan melakukan prediksi..."):
                
                # B. Memuat Model
                try:
                    saved_model = load_model(MODEL_PATH)
                except Exception as e:
                    st.error(f"Gagal memuat model! Pastikan path model benar.\nError: {e}")
                    st.stop()

                # C. Data Cleaning (Sesuai kode Anda)
                # Membersihkan koma pada kolom numerik jika terbaca sebagai text/object
                cols_numeric = ['Dist_Fasum', 'Dist_Shop', 'Dist_Faskes', 'Dist_SPBU', 
                                'Dist_Trans', 'Dist_Hotel', 'Dist_Govt', 'Dist_Sekol', 
                                'Dist_Trunk', 'Dist_Sec']

                for col in cols_numeric:
                    if col in df_baru.columns and df_baru[col].dtype == 'object':
                        df_baru[col] = df_baru[col].str.replace(',', '').astype(float)

                # D. Melakukan Prediksi
                try:
                    hasil_lengkap = predict_model(saved_model, data=df_baru)
                    
                    # Rename kolom hasil agar lebih cantik
                    if 'prediction_label' in hasil_lengkap.columns:
                        hasil_lengkap = hasil_lengkap.rename(columns={'prediction_label': 'Harga_Prediksi'})

                    # E. Menampilkan Hasil
                    st.success("Prediksi Selesai!")
                    st.subheader("2. Hasil Prediksi (Preview)")
                    
                    # Tampilkan kolom penting dulu di depan (NOP dan Harga)
                    cols = list(hasil_lengkap.columns)
                    if 'NOP' in cols and 'Harga_Prediksi' in cols:
                        # Pindahkan NOP dan Harga ke depan
                        cols.insert(0, cols.pop(cols.index('Harga_Prediksi')))
                        cols.insert(0, cols.pop(cols.index('NOP')))
                        st.dataframe(hasil_lengkap[cols].head())
                    else:
                        st.dataframe(hasil_lengkap.head())

                    # F. Download Button (Excel)
                    # Menggunakan BytesIO agar tidak perlu save ke harddisk server dulu
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        hasil_lengkap.to_excel(writer, index=False)
                    
                    st.download_button(
                        label="📥 Download Hasil Lengkap (.xlsx)",
                        data=buffer.getvalue(),
                        file_name="hasil_prediksi_final.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memprediksi data: {e}")
                    
    except Exception as e:
        st.error(f"File Excel tidak bisa dibaca. Pastikan formatnya benar. Error: {e}")

else:
    st.info("👈 Silakan upload file Excel (.xlsx) di sidebar sebelah kiri.")