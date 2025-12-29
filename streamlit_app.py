import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Analisis Penggunaan AI Mahasiswa", layout="wide")

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset_Klasifikasi_Pengguna_AI_Mahasiswa_UMMgl.csv", sep=";")
    df.columns = ["Nama", "Jurusan", "Semester", "AI_Tools", "Trust_Level", "Usage_Intensity"]
    return df

df = load_data()

# ===============================
# PREPROCESSING
# ===============================
def preprocess_data(df):
    df = df.copy()

    # Cleaning
    df["Usage_Intensity"] = df["Usage_Intensity"].astype(str)
    df["Usage_Intensity"] = df["Usage_Intensity"].str.replace("+", "", regex=False)
    df["Usage_Intensity"] = df["Usage_Intensity"].astype(int)

    # Klasifikasi Target
    def klasifikasi(x):
        if x <= 4:
            return "Rendah"
        elif x <= 7:
            return "Sedang"
        else:
            return "Tinggi"

    df["Level_AI"] = df["Usage_Intensity"].apply(klasifikasi)

    # Encoding
    encoder = LabelEncoder()
    for col in ["Jurusan", "AI_Tools", "Level_AI"]:
        df[col] = encoder.fit_transform(df[col])

    X = df[["Jurusan", "Semester", "AI_Tools", "Trust_Level", "Usage_Intensity"]]
    y = df["Level_AI"]

    return X, y, df

# ===============================
# LOGIN
# ===============================
st.sidebar.title("Login")

role = st.sidebar.selectbox("Pilih Role", ["Guru", "Siswa"])

# ===============================
# DASHBOARD GURU
# ===============================
if role == "Guru":
    st.title("📊 Dashboard Guru")

    if st.sidebar.button("Proses Data & Training Model"):
        X, y, df_clean = preprocess_data(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        st.subheader("📈 Evaluasi Model")
        st.write("**Akurasi Model:**", accuracy_score(y_test, y_pred))

        st.text("Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y_test, y_pred))

        # ===============================
        # REKOMENDASI OTOMATIS
        # ===============================
        st.subheader("📌 Rekomendasi Otomatis")

        df_result = df.copy()
        df_result["Prediksi_Level"] = model.predict(X)

        rekomendasi_map = {
            0: "🟢 Level Aman - Penggunaan AI masih wajar",
            1: "🟡 Perlu Teguran - Gunakan AI secara bijak",
            2: "🔴 Perlu Pengawasan Lebih Ketat"
        }

        df_result["Rekomendasi"] = df_result["Prediksi_Level"].map(rekomendasi_map)

        st.dataframe(df_result[["Nama", "Prediksi_Level", "Rekomendasi"]])

# ===============================
# DASHBOARD SISWA
# ===============================
else:
    st.title("🎓 Dashboard Siswa")

    nama = st.selectbox("Pilih Nama", df["Nama"].unique())

    X, y, df_clean = preprocess_data(df)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    idx = df[df["Nama"] == nama].index[0]
    pred = model.predict(X.iloc[[idx]])[0]

    rekomendasi_map = {
        0: "🟢 Level Aman – Tetap pertahankan etika akademik",
        1: "🟡 Sedang – Kurangi ketergantungan pada AI",
        2: "🔴 Tinggi – Butuh pengawasan & evaluasi akademik"
    }

    st.subheader("📌 Hasil Analisis")
    st.write("**Nama Mahasiswa:**", nama)
    st.write("**Tingkat Penggunaan AI:**", ["Rendah", "Sedang", "Tinggi"][pred])
    st.success(rekomendasi_map[pred])
