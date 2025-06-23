import streamlit as st
import pandas as pd
import joblib

# Load model dan scaler
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

kategori = ["Rendah", "Sedang", "Tinggi"]
expected_cols = ["Age", "Sleep_Quality", "Depression_Score", "Anxiety_Score", "Financial_Stress"]

# Mapping teks ke angka
sleep_map = {"Sangat Buruk": 1, "Buruk": 2, "Cukup": 3, "Baik": 4, "Sangat Baik": 5}
stress_map = {"Rendah": 1, "Sedang": 2, "Tinggi": 3}

st.title("Prediksi Tingkat Stres Mahasiswa")

tab1, tab2 = st.tabs(["Form Manual", "Upload CSV"])

# =============================
# TAB 1: INPUT MANUAL
with tab1:
    st.subheader("Masukkan Data Mahasiswa")

    usia = st.radio("Usia", options=[17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], horizontal=True)
    tidur = st.selectbox("Kualitas Tidur", list(sleep_map.keys()))
    depresi = st.slider("Skor Depresi (0–10)", 0, 10, 5)
    cemas = st.slider("Skor Kecemasan (0–10)", 0, 10, 5)
    stres_uang = st.selectbox("Tingkat Stres Finansial", list(stress_map.keys()))

    if st.button("Prediksi"):
        input_df = pd.DataFrame([[usia, sleep_map[tidur], depresi, cemas, stress_map[stres_uang]]],
                                columns=expected_cols)
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        st.success(f"Hasil Prediksi: {kategori[int(pred)]} (label: {pred})")

# =============================
# TAB 2: UPLOAD CSV
with tab2:
    st.subheader("Upload File CSV")

    uploaded = st.file_uploader("Unggah file CSV", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.write("Data Diupload:")
            st.dataframe(df.head())

            # Mapping teks ke angka
            for col in df.columns:
                df[col] = df[col].replace(sleep_map | stress_map)

            df = df[expected_cols]
            df = df.apply(pd.to_numeric, errors='coerce')

            if df.isnull().any().any():
                st.error("Beberapa nilai tidak valid. Pastikan semua kolom lengkap dan sesuai.")
            else:
                X_scaled = scaler.transform(df)
                preds = model.predict(X_scaled)
                df["Prediksi_Tingkat_Stres"] = [kategori[int(p)] for p in preds]

                st.success("Prediksi Berhasil!")
                st.dataframe(df)

                csv_out = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Hasil Prediksi CSV", csv_out, "hasil_prediksi.csv", "text/csv")

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
