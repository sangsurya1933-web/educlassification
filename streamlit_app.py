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
# SESSION STATE INIT
# ======================================================
for key in [
    "login", "role", "guru_data",
    "model", "encoders",
    "X_train", "X_test", "y_test", "y_pred", "accuracy"
]:
    if key not in st.session_state:
        st.session_state[key] = None

# ======================================================
# PREPROCESSING
# ======================================================
def preprocess_and_train(df):
    df = df.copy()

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Simpan ke session
    st.session_state.model = model
    st.session_state.encoders = encoders
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.y_pred = y_pred
    st.session_state.accuracy = accuracy_score(y_test, y_pred)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODER_PATH)

# ======================================================
# LOGIN PAGE
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
        "Menu",
        ["Upload Dataset", "Data Training", "Data Latih & Uji", "Analisis Klasifikasi", "Evaluasi"]
    )

    # ---------------- Upload ----------------
    if menu == "Upload Dataset":
        st.title("📤 Upload Dataset Training")
        file = st.file_uploader("Upload CSV", type="csv")
        if file:
            df = pd.read_csv(file, sep=";")
            st.session_state.guru_data = df
            preprocess_and_train(df)
            st.success("Dataset berhasil diproses & model dilatih")
            st.dataframe(df.head())

    # Validasi
    if st.session_state.guru_data is None and menu != "Upload Dataset":
        st.warning("Silakan upload dataset terlebih dahulu.")
        return

    # ---------------- Data Training ----------------
    if menu == "Data Training":
        st.title("📄 Data Training")
        st.dataframe(st.session_state.guru_data)

    # ---------------- Data Latih & Uji ----------------
    elif menu == "Data Latih & Uji":
        st.title("📂 Data Latih & Data Uji")
        st.write("Data Latih:", st.session_state.X_train.shape[0])
        st.write("Data Uji:", st.session_state.X_test.shape[0])

    # ---------------- Analisis ----------------
    elif menu == "Analisis Klasifikasi":
        st.title("🧠 Analisis Klasifikasi")
        st.success(
            f"Akurasi Random Forest: **{st.session_state.accuracy:.2f}**"
        )

    # ---------------- Evaluasi ----------------
    elif menu == "Evaluasi":
        st.title("📊 Evaluasi Model")

        cm = confusion_matrix(
            st.session_state.y_test,
            st.session_state.y_pred
        )

        fig, ax = plt.subplots()
        sns.heatmap(
            cm, annot=True, fmt="d",
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"],
            cmap="Blues", ax=ax
        )
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        st.pyplot(fig)

        st.text(
            classification_report(
                st.session_state.y_test,
                st.session_state.y_pred
            )
        )

    if st.sidebar.button("🚪 Logout"):
        for key in st.session_state.keys():
            st.session_state[key] = None
        st.rerun()

# ======================================================
# DASHBOARD SISWA
# ======================================================
def siswa_dashboard():
    st.title("🎓 Analisis Tingkat Penggunaan AI")

    file = st.file_uploader("Upload Dataset Analisis CSV", type="csv")
    if file:
        df = pd.read_csv(file, sep=";")

        if not os.path.exists(MODEL_PATH):
            st.error("Model belum tersedia. Guru harus melakukan training.")
            return

        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODER_PATH)

        for col, le in encoders.items():
            df[col] = le.transform(df[col])

        df["Hasil_Klasifikasi_Tingkat_AI"] = model.predict(df)
        st.success("Hasil Klasifikasi")
        st.dataframe(df)

    if st.button("🚪 Logout"):
        st.session_state.login = False
        st.session_state.role = None
        st.rerun()

# ======================================================
# MAIN
# ======================================================
if not st.session_state.login:
    login_page()
elif st.session_state.role == "Guru":
    guru_dashboard()
else:
    siswa_dashboard()
