# app.py
# Aplikasi Knowledge Base System
# Judul: Analisis Tingkat Penggunaan AI terhadap Performa Akademik Mahasiswa

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==============================
# KONFIGURASI AWAL
# ==============================
st.set_page_config(page_title="Analisis AI & Performa Akademik", layout="wide")

DATASET_PATH = "/mnt/data/Dataset_Klasifikasi_Pengguna_AI_Mahasiswa_UMMgl.csv"

# ==============================
# FUNGSI UTILITAS
# ==============================
@st.cache_data
def load_data():
    return pd.read_csv(DATASET_PATH)


def clean_data(df):
    df = df.drop_duplicates()
    df = df.fillna(df.mode().iloc[0])
    return df


def encode_data(df):
    encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def generate_recommendation(ai_level):
    if ai_level == 0:
        return "Disarankan meningkatkan pemanfaatan AI secara terarah untuk membantu pembelajaran."
    elif ai_level == 1:
        return "Penggunaan AI sudah cukup baik, perlu dikombinasikan dengan strategi belajar mandiri."
    else:
        return "Penggunaan AI sangat baik, pertahankan dengan tetap menjaga integritas akademik."

# ==============================
# SIDEBAR LOGIN
# ==============================
st.sidebar.title("Login Pengguna")
role = st.sidebar.selectbox("Masuk sebagai", ["Guru", "Mahasiswa"])

# ==============================
# DASHBOARD GURU
# ==============================
if role == "Guru":
    st.title("Dashboard Guru")
    st.subheader("Pengolahan Data & Analisis Model")

    df = load_data()

    menu = st.tabs([
        "Dataset",
        "Pre-processing",
        "Training & Testing",
        "Evaluasi",
        "Knowledge Base & Rekomendasi"
    ])

    # ===== TAB DATASET =====
    with menu[0]:
        st.write("### Dataset Mahasiswa")
        st.dataframe(df, use_container_width=True)

    # ===== TAB PREPROCESSING =====
    with menu[1]:
        st.write("### Data Cleaning & Encoding")
        if st.button("Lakukan Pre-processing"):
            df_clean = clean_data(df)
            df_encoded, encoders = encode_data(df_clean)
            st.success("Pre-processing berhasil dilakukan")
            st.dataframe(df_encoded.head(), use_container_width=True)
            st.session_state['df_encoded'] = df_encoded

    # ===== TAB TRAINING =====
    with menu[2]:
        st.write("### Training dan Testing Random Forest")
        if 'df_encoded' in st.session_state:
            df_encoded = st.session_state['df_encoded']
            target_col = st.selectbox("Pilih Kolom Target (Klasifikasi)", df_encoded.columns)

            test_size = st.slider("Proporsi Data Uji", 0.2, 0.4, 0.2)

            if st.button("Latih Model"):
                X = df_encoded.drop(columns=[target_col])
                y = df_encoded[target_col]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )

                model = train_model(X_train, y_train)
                y_pred = model.predict(X_test)

                st.session_state['model'] = model
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['y_pred'] = y_pred

                st.success("Model Random Forest berhasil dilatih")
        else:
            st.warning("Lakukan pre-processing terlebih dahulu")

    # ===== TAB EVALUASI =====
    with menu[3]:
        st.write("### Evaluasi Model")
        if 'y_pred' in st.session_state:
            y_test = st.session_state['y_test']
            y_pred = st.session_state['y_pred']

            acc = accuracy_score(y_test, y_pred)
            st.metric("Akurasi Model", f"{acc:.2f}")

            st.text("Classification Report")
            st.text(classification_report(y_test, y_pred))

            st.text("Confusion Matrix")
            st.write(confusion_matrix(y_test, y_pred))
        else:
            st.info("Model belum dilatih")

    # ===== TAB KNOWLEDGE BASE =====
    with menu[4]:
        st.write("### Aturan Knowledge Base & Rekomendasi")
        st.markdown("""
        **Aturan Sistem Pakar (Knowledge Base):**
        - Kelas 0 : Penggunaan AI Rendah → Risiko performa akademik kurang optimal
        - Kelas 1 : Penggunaan AI Sedang → Performa akademik cukup stabil
        - Kelas 2 : Penggunaan AI Tinggi → Performa akademik optimal
        """)

# ==============================
# DASHBOARD MAHASISWA
# ==============================
else:
    st.title("Dashboard Mahasiswa")
    st.subheader("Hasil Analisis Tingkat Penggunaan AI")

    nama = st.text_input("Masukkan Nama Mahasiswa")

    if st.button("Lihat Hasil"):
        if 'model' in st.session_state:
            model = st.session_state['model']
            df = load_data()

            # Ambil data mahasiswa secara acak (simulasi knowledgebase)
            sample = df.sample(1)
            sample_clean = clean_data(sample)
            sample_encoded, _ = encode_data(sample_clean)

            prediction = model.predict(sample_encoded.drop(columns=[sample_encoded.columns[-1]]))[0]
            rekomendasi = generate_recommendation(prediction)

            st.success(f"Nama Mahasiswa : {nama}")
            st.info(f"Tingkat Penggunaan AI (Klasifikasi): {prediction}")
            st.write(f"**Rekomendasi:** {rekomendasi}")
        else:
            st.warning("Model belum tersedia, silakan hubungi guru")
