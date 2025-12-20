import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(page_title="Analisis Penggunaan AI", layout="wide")

MODEL_PATH = "model_rf.pkl"
ENCODER_PATH = "encoders.pkl"

# ======================================================
# SESSION STATE
# ======================================================
if "login" not in st.session_state:
    st.session_state.login = False
if "role" not in st.session_state:
    st.session_state.role = ""
if "guru_data" not in st.session_state:
    st.session_state.guru_data = None

# ======================================================
# PREPROCESSING
# ======================================================
def preprocessing(df):
    df = df.copy()

    # Mapping target sesuai dataset
    target_map = {1: "Low", 2: "Low", 3: "Medium", 4: "Medium", 5: "High"}
    df["Impact_Label"] = df["Impact_on_Grades"].map(target_map)

    features = [
        "Stream", "Year_of_Study", "AI_Tools_Used",
        "Daily_Usage_Hours", "Trust_in_AI_Tools",
        "Awareness_Level", "Device_Used", "Internet_Access"
    ]

    X = df[features]
    y = df["Impact_Label"]

    encoders = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    return X, y, encoders

# ======================================================
# TRAIN MODEL (HANYA DARI DATA GURU)
# ======================================================
def train_model(df):
    X, y, encoders = preprocessing(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODER_PATH)

    return X_train, X_test, acc, cm, report

# ======================================================
# LOGIN PAGE (TANPA PASSWORD)
# ======================================================
def login_page():
    st.title("🔐 Login Sistem")

    role = st.selectbox("Masuk sebagai", ["Guru", "Siswa"])

    if st.button("Masuk"):
        st.session_state.login = True
        st.session_state.role = role
        st.rerun()

# ======================================================
# DASHBOARD GURU
# ======================================================
def guru_dashboard():
    st.sidebar.title("📊 Menu Guru")
    menu = st.sidebar.radio(
        "Pilih Menu",
        ["Upload Dataset", "Data Training", "Data Latih & Uji", "Analisis Klasifikasi", "Evaluasi Model"]
    )

    # ---------------- Upload Dataset ----------------
    if menu == "Upload Dataset":
        st.title("📤 Upload Dataset Training (Guru)")
        uploaded = st.file_uploader("Upload CSV Dataset Siswa", type="csv")

        if uploaded:
            df = pd.read_csv(uploaded, sep=";")
            st.session_state.guru_data = df
            st.success("Dataset berhasil diupload")
            st.dataframe(df.head())

    # Validasi dataset
    if st.session_state.guru_data is None and menu != "Upload Dataset":
        st.warning("Silakan upload dataset terlebih dahulu.")
        return

    df = st.session_state.guru_data

    # ---------------- Data Training ----------------
    if menu == "Data Training":
        st.title("📄 Data Training")
        st.dataframe(df)

    # ---------------- Data Latih & Uji ----------------
    elif menu == "Data Latih & Uji":
        st.title("📂 Data Latih & Data Uji")
        X_train, X_test, _, _, _ = train_model(df)
        st.write("Jumlah Data Latih:", X_train.shape[0])
        st.write("Jumlah Data Uji:", X_test.shape[0])

    # ---------------- Analisis Klasifikasi ----------------
    elif menu == "Analisis Klasifikasi":
        st.title("🧠 Analisis Klasifikasi")
        _, _, acc, _, _ = train_model(df)
        st.success(f"Akurasi Model Random Forest: **{acc:.2f}**")

    # ---------------- Evaluasi Model ----------------
    elif menu == "Evaluasi Model":
        st.title("📊 Evaluasi Model")
        _, _, _, cm, report = train_model(df)

        fig, ax = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"],
            ax=ax
        )
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        st.pyplot(fig)

        st.subheader("Classification Report")
        st.text(report)

    if st.sidebar.button("🚪 Logout"):
        st.session_state.login = False
        st.session_state.role = ""
        st.session_state.guru_data = None
        st.rerun()

# ======================================================
# DASHBOARD SISWA
# ======================================================
def siswa_dashboard():
    st.title("🎓 Analisis Tingkat Penggunaan AI (Siswa)")

    uploaded = st.file_uploader("Upload Dataset Analisis (CSV)", type="csv")

    if uploaded:
        data = pd.read_csv(uploaded, sep=";")
        st.dataframe(data.head())

        if not os.path.exists(MODEL_PATH):
            st.error("Model belum tersedia. Guru harus melakukan training terlebih dahulu.")
            return

        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODER_PATH)

        for col, le in encoders.items():
            data[col] = le.transform(data[col])

        prediction = model.predict(data)
        data["Hasil_Klasifikasi_Tingkat_AI"] = prediction

        st.success("Hasil Analisis Klasifikasi")
        st.dataframe(data)

    if st.button("🚪 Logout"):
        st.session_state.login = False
        st.session_state.role = ""
        st.rerun()

# ======================================================
# MAIN
# ======================================================
if not st.session_state.login:
    login_page()
else:
    if st.session_state.role == "Guru":
        guru_dashboard()
    elif st.session_state.role == "Siswa":
        siswa_dashboard()
