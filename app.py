import streamlit as st
import pandas as pd
import joblib

# Load model dan scaler
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

kategori = ["Rendah", "Sedang", "Tinggi"]

st.title("Aplikasi Prediksi Tingkat Stres Mahasiswa")

# Tab untuk dua opsi input
tab1, tab2 = st.tabs(["Form Input Manual", "Upload CSV Dataset"])

# ==============================
# Tab 1: Form Slider
with tab1:
    st.subheader("Input Manual (1 Mahasiswa)")

    usia = st.slider("Usia", 17, 30, 21)
    tidur = st.slider("Jam Tidur", 0, 10, 6)
    depresi = st.slider("Skor Depresi", 0, 10, 5)
    cemas = st.slider("Skor Kecemasan", 0, 10, 5)
    stres_uang = st.slider("Stres Finansial", 0, 10, 5)

    if st.button("Prediksi"):
        data = pd.DataFrame([[usia, tidur, depresi, cemas, stres_uang]],
            columns=["Age", "Sleep_Quality", "Depression_Score", "Anxiety_Score", "Financial_Stress"])
        data_scaled = scaler.transform(data)
        pred = model.predict(data_scaled)[0]
        st.success(f"Hasil Prediksi: {kategori[int(pred)]} (label: {pred})")

# ==============================
# Tab 2: Upload CSV
with tab2:
    st.subheader("Prediksi Massal (Upload CSV)")
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        expected_cols = ["Age", "Sleep_Quality", "Depression_Score", "Anxiety_Score", "Financial_Stress"]

        if all(col in df.columns for col in expected_cols):
            X = df[expected_cols]
            X_scaled = scaler.transform(X)
            preds = model.predict(X_scaled)
            hasil = [kategori[int(p)] for p in preds]
            df["Prediksi_Tingkat_Stres"] = hasil

            st.dataframe(df)

            # Download hasil
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Hasil Prediksi CSV", data=csv, file_name="hasil_prediksi.csv", mime="text/csv")
        else:
            st.error("CSV harus memiliki kolom: " + ", ".join(expected_cols))
