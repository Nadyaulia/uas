import streamlit as st
import pandas as pd
import joblib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Prediksi Obesitas", layout="centered")
st.title("Prediksi Kategori Obesitas")
st.write("Silakan lengkapi data berikut untuk mengetahui kategori obesitas Anda.")

# Load pipeline (model + preprocessing)
@st.cache_resource
def load_pipeline():
    return joblib.load("model_pipeline.joblib")  # Pastikan file ini tersedia di folder yang sama

pipeline = load_pipeline()

# Form input
with st.form("form_prediksi"):
    Age = st.number_input("Usia", min_value=1, max_value=120, value=25)
    Gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    Height = st.number_input("Tinggi (m)", min_value=1.0, max_value=2.5, value=1.65)
    Weight = st.number_input("Berat (kg)", min_value=30.0, max_value=200.0, value=65.0)
    FCVC = st.slider("Frekuensi konsumsi sayur (1=sangat jarang, 3=sangat sering)", 1.0, 3.0, 2.0)
    NCP = st.slider("Jumlah makanan utama per hari", 1.0, 4.0, 3.0)
    CH2O = st.slider("Jumlah konsumsi air per hari", 1.0, 3.0, 2.0)
    FAF = st.slider("Aktivitas fisik mingguan (jam)", 0.0, 3.0, 1.0)
    TUE = st.slider("Waktu layar harian (jam)", 0.0, 2.0, 1.0)
    FAVC = st.selectbox("Sering makan makanan tinggi kalori?", ["yes", "no"])
    CALC = st.selectbox("Konsumsi alkohol", ["no", "Sometimes", "Frequently", "Always"])
    SMOKE = st.selectbox("Merokok?", ["yes", "no"])
    SCC = st.selectbox("Mengontrol konsumsi kalori?", ["yes", "no"])
    family_history_with_overweight = st.selectbox("Riwayat keluarga obesitas?", ["yes", "no"])
    CAEC = st.selectbox("Konsumsi makanan antar waktu", ["no", "Sometimes", "Frequently", "Always"])
    MTRANS = st.selectbox("Transportasi harian", ["Public_Transportation", "Walking", "Bike", "Automobile", "Motorbike"])

    submitted = st.form_submit_button("Prediksi")

# Jika tombol ditekan
if submitted:
    # Buat DataFrame dari input
    input_df = pd.DataFrame([{
        "Age": Age,
        "Gender": Gender,
        "Height": Height,
        "Weight": Weight,
        "FCVC": FCVC,
        "NCP": NCP,
        "CH2O": CH2O,
        "FAF": FAF,
        "TUE": TUE,
        "FAVC": FAVC,
        "CALC": CALC,
        "SMOKE": SMOKE,
        "SCC": SCC,
        "family_history_with_overweight": family_history_with_overweight,
        "CAEC": CAEC,
        "MTRANS": MTRANS
    }])

    st.write("Data yang dikirim ke model:")
    st.dataframe(input_df)

    # Prediksi
    prediction = pipeline.predict(input_df)[0]
    st.success(f"Kategori Obesitas Anda: **{prediction}**")
