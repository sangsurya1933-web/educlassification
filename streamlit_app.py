import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(page_title="AI vs Performa Akademik", layout="wide")

DATA_PATH = "Students tabel.csv"
MODEL_PATH = "model_rf.pkl"
ENCODER_PATH = "encoders.pkl"

# =====================================================
# INIT SESSION STATE (WAJIB)
# =====================================================
if "login" not in st.session_state:
    st.session_state.login = False

if "role" not in st.session_state:
    st.session_state.role = None

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, sep=";")

df = load_data()

# =====================================================
# PREPROCESSING
# =====================================================
def preprocessing(df):
    df = df.copy()

    # Mapping target
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

# =====================================================
# TRAIN MODEL
# =====================================================
def train_model():
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

    return acc, cm, report

# =====================================================
# ================= LOGIN =================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "role" not in st.session_state:
    st.session_state.role = ""

def login_page():
    st.title("🔐 Login")

    role = st.selectbox("Role", ["Guru", "Siswa"])
    user = st.text_input("Username")
    pw = st.text_input("Password", type="password")

    if st.button("Login"):
        if role == "Guru" and user == "guru" and pw == "guru123":
            st.session_state.logged_in = True
            st.session_state.role = "Guru"
            st.experimental_rerun()

        elif role == "Siswa" and user == "siswa" and pw == "siswa123":
            st.session_state.logged_in = True
            st.session_state.role = "Siswa"
            st.experimental_rerun()

        else:
            st.error("Username atau Password salah")
# =====================================================
# DASHBOARD GURU
# =====================================================
def guru_dashboard():
    st.title("📊 Dashboard Guru")

    if st.button("🔁 Train Model Random Forest"):
        acc, cm, report = train_model()

        st.success("Model berhasil dilatih dan disimpan")

        st.subheader("📈 Akurasi Model")
        st.write(f"Accuracy: **{acc:.2f}**")

        st.subheader("📉 Confusion Matrix")
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

        st.subheader("📄 Classification Report")
        st.text(report)

    if st.button("🚪 Logout"):
        st.session_state.login = False
        st.session_state.role = None
        st.rerun()

# =====================================================
# DASHBOARD SISWA
# =====================================================
def siswa_dashboard():
    st.title("🎓 Dashboard Siswa")

    if not os.path.exists(MODEL_PATH):
        st.warning("Model belum tersedia. Silakan minta guru melakukan training.")
        return

    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)

    col1, col2 = st.columns(2)

    with col1:
        stream = st.selectbox("Stream", df["Stream"].unique())
        year = st.selectbox("Year of Study", sorted(df["Year_of_Study"].unique()))
        ai_tool = st.selectbox("AI Tools Used", df["AI_Tools_Used"].unique())
        usage = st.slider("Daily Usage Hours", 0, 40, 5)

    with col2:
        trust = st.slider("Trust in AI Tools", 1, 5, 3)
        aware = st.slider("Awareness Level", 1, 10, 5)
        device = st.selectbox("Device Used", df["Device_Used"].unique())
        internet = st.selectbox("Internet Access", df["Internet_Access"].unique())

    if st.button("📌 Prediksi Performa Akademik"):
        input_df = pd.DataFrame([[
            stream, year, ai_tool, usage,
            trust, aware, device, internet
        ]], columns=[
            "Stream", "Year_of_Study", "AI_Tools_Used",
            "Daily_Usage_Hours", "Trust_in_AI_Tools",
            "Awareness_Level", "Device_Used", "Internet_Access"
        ])

        for col, le in encoders.items():
            input_df[col] = le.transform(input_df[col])

        prediction = model.predict(input_df)
        st.success(f"Hasil Klasifikasi Performa Akademik: **{prediction[0]}**")

    if st.button("🚪 Logout"):
        st.session_state.login = False
        st.session_state.role = None
        st.rerun()

# =====================================================
# MAIN FLOW (PALING PENTING)
# =====================================================
if not st.session_state.login:
    login_page()
else:
    if st.session_state.role == "Guru":
        guru_dashboard()
    elif st.session_state.role == "Siswa":
        siswa_dashboard()
