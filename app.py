import streamlit as st
import pandas as pd
import joblib

model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

kategori = ["Rendah", "Sedang", "Tinggi"]
expected_cols = ["Age", "Sleep_Quality", "Depression_Score", "Anxiety_Score", "Financial_Stress"]

st.title("Aplikasi Prediksi Tingkat Stres Mahasiswa")

tab1, tab2 = st.tabs(["Form Input Manual", "Upload CSV"])

with tab1:
    usia = st.slider("Usia", 17, 30, 21)
    tidur = st.selectbox("Kualitas Tidur", ["Sangat Buruk", "Buruk", "Cukup", "Baik", "Sangat Baik"])
    depresi = st.select_slider("Skor Depresi", range(0,11), 5)
    cemas = st.select_slider("Skor Kecemasan", range(0,11), 5)
    stres_uang = st.selectbox("Tingkat Stres Finansial", ["Rendah", "Sedang", "Tinggi"])

    sleep_map = {"Sangat Buruk": 1, "Buruk": 2, "Cukup": 3, "Baik": 4, "Sangat Baik": 5}
    stress_map = {"Rendah": 1, "Sedang": 2, "Tinggi": 3}

    if st.button("Prediksi"):
        data = pd.DataFrame([[usia, sleep_map[tidur], depresi, cemas, stress_map[stres_uang]]],
                            columns=expected_cols)
        data_scaled = scaler.transform(data)
        pred = model.predict(data_scaled)[0]
        st.success(f"Hasil Prediksi: {kategori[int(pred)]} (label: {pred})")

with tab2:
    st.subheader("Upload CSV")

    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Data Diupload:")
        st.dataframe(df.head())

        # Mapping
        map_kualitas_tidur = {"Sangat Buruk": 1, "Buruk": 2, "Cukup": 3, "Baik": 4, "Sangat Baik": 5}
        map_stres_uang = {"Rendah": 1, "Sedang": 2, "Tinggi": 3}

        try:
            for col in df.columns:
                if df[col].dtype == "object":
                    df[col] = df[col].replace(map_kualitas_tidur | map_stres_uang)

            df = df[expected_cols]
            df = df.apply(pd.to_numeric, errors='coerce')
            if df.isnull().any().any():
                st.error("Beberapa nilai tidak bisa dikonversi menjadi angka. Periksa isi file.")
            else:
                X_scaled = scaler.transform(df)
                preds = model.predict(X_scaled)
                df["Prediksi_Tingkat_Stres"] = [kategori[int(p)] for p in preds]
                st.success("Prediksi berhasil!")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Hasil CSV", csv, "hasil_prediksi.csv", "text/csv")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")
