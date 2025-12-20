import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# =====================
# LOAD DATA
# =====================
@st.cache_data
def load_data():
    df = pd.read_csv("Students tabel.csv", sep=";")
    return df

df = load_data()

# =====================
# LOGIN SYSTEM
# =====================
def login():
    st.title("🔐 Login Sistem")

    role = st.selectbox("Login sebagai", ["Guru", "Siswa"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if role == "Guru" and username == "guru" and password == "guru123":
            st.session_state["role"] = "Guru"
        elif role == "Siswa" and username == "siswa" and password == "siswa123":
            st.session_state["role"] = "Siswa"
        else:
            st.error("Username atau Password salah!")

# =====================
# PREPROCESSING
# =====================
def preprocessing(df):
    df_model = df.copy()

    target_map = {
        1: "Low",
        2: "Low",
        3: "Medium",
        4: "Medium",
        5: "High"
    }
    df_model["Impact_Label"] = df_model["Impact_on_Grades"].map(target_map)

    features = [
        "Stream", "Year_of_Study", "AI_Tools_Used",
        "Daily_Usage_Hours", "Trust_in_AI_Tools",
        "Awareness_Level", "Device_Used", "Internet_Access"
    ]

    X = df_model[features]
    y = df_model["Impact_Label"]

    encoders = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    return X, y, encoders

# =====================
# TRAIN MODEL
# =====================
def train_model(X, y):
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

    joblib.dump(model, "model_rf.pkl")

    return acc, cm, report

# =====================
# DASHBOARD GURU
# =====================
def guru_dashboard():
    st.title("📊 Dashboard Guru")

    if st.button("Train Model Random Forest"):
        X, y, encoders = preprocessing(df)
        acc, cm, report = train_model(X, y)

        st.success("Model berhasil dilatih!")

        st.subheader("📈 Akurasi Model")
        st.write(f"Akurasi: **{acc:.2f}**")

        st.subheader("📉 Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        st.subheader("📄 Classification Report")
        st.text(report)

# =====================
# DASHBOARD SISWA
# =====================
def siswa_dashboard():
    st.title("🎓 Dashboard Siswa")

    model = joblib.load("model_rf.pkl")

    stream = st.selectbox("Stream", df["Stream"].unique())
    year = st.selectbox("Year of Study", sorted(df["Year_of_Study"].unique()))
    ai_tool = st.selectbox("AI Tool Used", df["AI_Tools_Used"].unique())
    usage = st.slider("Daily Usage Hours", 0, 40, 5)
    trust = st.slider("Trust in AI", 1, 5, 3)
    aware = st.slider("Awareness Level", 1, 10, 5)
    device = st.selectbox("Device Used", df["Device_Used"].unique())
    internet = st.selectbox("Internet Access", df["Internet_Access"].unique())

    if st.button("Prediksi Performa Akademik"):
        data = pd.DataFrame([[stream, year, ai_tool, usage, trust, aware, device, internet]],
                            columns=[
                                "Stream", "Year_of_Study", "AI_Tools_Used",
                                "Daily_Usage_Hours", "Trust_in_AI_Tools",
                                "Awareness_Level", "Device_Used", "Internet_Access"
                            ])

        for col in data.select_dtypes(include="object").columns:
            le = LabelEncoder()
            le.fit(df[col])
            data[col] = le.transform(data[col])

        result = model.predict(data)
        st.success(f"📌 Prediksi Performa Akademik: **{result[0]}**")

# =====================
# MAIN APP
# =====================
if "role" not in st.session_state:
    login()
else:
    if st.session_state["role"] == "Guru":
        guru_dashboard()
    elif st.session_state["role"] == "Siswa":
        siswa_dashboard()
