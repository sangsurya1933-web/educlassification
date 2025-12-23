import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ======================
# KONFIGURASI
# ======================
st.set_page_config(
    page_title="Analisis Penggunaan AI",
    layout="wide"
)

st.title("📊 Analisis Tingkat Penggunaan AI terhadap Performa Akademik Siswa")
st.caption("Algoritma Random Forest")

# ======================
# SIDEBAR
# ======================
role = st.sidebar.selectbox("Login sebagai", ["Guru", "Siswa"])

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset (CSV)",
    type=["csv"]
)

# ======================
# LOAD CSV
# ======================
if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=";")
    st.sidebar.success("Dataset CSV berhasil diunggah")

# ======================
# PREPROCESSING
# ======================
def preprocessing(df):
    df = df.drop_duplicates()

    # Missing values
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].mean())

    # Outlier (IQR)
    for col in df.select_dtypes(include="number").columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    # Encoding
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])

    # Standarisasi
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df

# ======================
# MENU GURU
# ======================
if role == "Guru" and uploaded_file:
    menu = st.sidebar.selectbox(
        "Menu Guru",
        ["Dataset", "Preprocessing", "Pemodelan", "Evaluasi"]
    )

    if menu == "Dataset":
        st.subheader("📂 Dataset Siswa")
        st.dataframe(df)

    elif menu == "Preprocessing":
        st.subheader("⚙️ Preprocessing Data")
        df_clean = preprocessing(df)
        st.dataframe(df_clean.head())

        os.makedirs("output", exist_ok=True)
        df_clean.to_csv("output/data_clean.csv", index=False)
        st.success("Data preprocessing disimpan (data_clean.csv)")

    elif menu == "Pemodelan":
        st.subheader("🌲 Pemodelan Random Forest")

        df_clean = preprocessing(df)

        X = df_clean.drop(columns=["Impact_on_Grades"])
        y = df_clean["Impact_on_Grades"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        X_train.to_csv("output/data_train.csv", index=False)
        X_test.to_csv("output/data_test.csv", index=False)

        st.success("Model Random Forest berhasil dilatih")

    elif menu == "Evaluasi":
        st.subheader("📈 Evaluasi Model")

        df_clean = preprocessing(df)

        X = df_clean.drop(columns=["Impact_on_Grades"])
        y = df_clean["Impact_on_Grades"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Akurasi", f"{acc:.2%}")
        col2.metric("Presisi", f"{prec:.2%}")
        col3.metric("Recall", f"{rec:.2%}")
        col4.metric("F1-Score", f"{f1:.2%}")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        ax.imshow(cm)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        st.pyplot(fig)

        hasil = pd.DataFrame({
            "Aktual": y_test,
            "Prediksi": y_pred
        })
        hasil.to_csv("output/hasil_prediksi.csv", index=False)

# ======================
# MENU SISWA
# ======================
elif role == "Siswa" and uploaded_file:
    st.subheader("🔍 Analisis Performa Akademik")

    df_clean = preprocessing(df)

    X = df_clean.drop(columns=["Impact_on_Grades"])
    y = df_clean["Impact_on_Grades"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    pred = model.predict(X)
    st.dataframe(pd.DataFrame({"Prediksi": pred}))
