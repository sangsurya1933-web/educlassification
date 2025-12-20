import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# KONFIGURASI HALAMAN
# ==============================
st.set_page_config(
    page_title="Analisis Penggunaan AI terhadap Performa Akademik",
    layout="wide"
)

# ==============================
# LOGIN SYSTEM
# ==============================
def login():
    st.title("🔐 Login Dashboard")

    role = st.selectbox("Login sebagai", ["Siswa", "Guru"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if role == "Guru" and username == "guru" and password == "guru123":
            st.session_state["login"] = True
            st.session_state["role"] = "Guru"
        elif role == "Siswa" and username == "siswa" and password == "siswa123":
            st.session_state["login"] = True
            st.session_state["role"] = "Siswa"
        else:
            st.error("Username atau Password salah!")

# ==============================
# LOAD DATASET
# ==============================
@st.cache_data
def load_data():
    df = pd.read_csv("Students tabel.csv")
    return df

# ==============================
# MAIN PROGRAM
# ==============================
def main():
    if "login" not in st.session_state:
        login()
        return

    st.sidebar.title("📊 MENU APLIKASI")
    menu = st.sidebar.selectbox(
        "Pilih Menu",
        [
            "Dashboard",
            "Preprocessing Data",
            "Data Latih",
            "Data Uji",
            "Klasifikasi Random Forest",
            "Evaluasi Model",
            "Logout"
        ]
    )

    df = load_data()

    # ==============================
    # DASHBOARD
    # ==============================
    if menu == "Dashboard":
        st.title("📈 Dashboard Analisis Akademik")
        st.write(f"Login sebagai: **{st.session_state['role']}**")
        st.dataframe(df.head())

    # ==============================
    # PREPROCESSING
    # ==============================
    elif menu == "Preprocessing Data":
        st.title("🧹 Preprocessing Data")

        st.subheader("Cek Missing Value")
        st.write(df.isnull().sum())

        st.subheader("Encoding Label")
        encoder = LabelEncoder()
        df['Academic_Performance'] = encoder.fit_transform(df['Academic_Performance'])
        st.success("Label Encoding berhasil")

        st.subheader("Dataset Setelah Preprocessing")
        st.dataframe(df.head())

        st.session_state["df_processed"] = df

    # ==============================
    # DATA LATIH
    # ==============================
    elif menu == "Data Latih":
        st.title("📚 Data Latih")

        X = df.drop('Academic_Performance', axis=1)
        y = df['Academic_Performance']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        st.write("Jumlah Data Latih:", X_train.shape)
        st.dataframe(X_train.head())

        st.session_state["X_train"] = X_train
        st.session_state["y_train"] = y_train
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test

    # ==============================
    # DATA UJI
    # ==============================
    elif menu == "Data Uji":
        st.title("🧪 Data Uji")

        X_test = st.session_state.get("X_test")
        y_test = st.session_state.get("y_test")

        st.write("Jumlah Data Uji:", X_test.shape)
        st.dataframe(X_test.head())

    # ==============================
    # KLASIFIKASI RANDOM FOREST
    # ==============================
    elif menu == "Klasifikasi Random Forest":
        st.title("🌳 Klasifikasi Random Forest")

        n_estimators = st.slider("Jumlah Tree", 10, 200, 100)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42
        )

        model.fit(
            st.session_state["X_train"],
            st.session_state["y_train"]
        )

        y_pred = model.predict(st.session_state["X_test"])

        st.success("Model berhasil dilatih!")

        st.session_state["model"] = model
        st.session_state["y_pred"] = y_pred

        st.subheader("Hasil Prediksi")
        st.write(y_pred)

    # ==============================
    # EVALUASI MODEL
    # ==============================
    elif menu == "Evaluasi Model":
        st.title("📊 Evaluasi Model")

        y_test = st.session_state["y_test"]
        y_pred = st.session_state["y_pred"]

        acc = accuracy_score(y_test, y_pred)

        st.metric("Akurasi Model", f"{acc:.2f}")

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

    # ==============================
    # LOGOUT
    # ==============================
    elif menu == "Logout":
        st.session_state.clear()
        st.success("Logout berhasil")

# ==============================
# RUN APP
# ==============================
main()
