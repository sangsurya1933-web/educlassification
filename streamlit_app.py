import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

st.title("Analisis Penggunaan AI terhadap Performa Akademik Siswa")

# ======================
# LOAD DATA
# ======================
@st.cache_data
def load_data():
    return pd.read_csv("Students_tabel.csv")

df = load_data()

# ======================
# LOGIN
# ======================
role = st.sidebar.selectbox("Login Sebagai", ["Guru", "Siswa"])

# ======================
# PREPROCESSING
# ======================
def preprocess(df):
    df = df.dropna().drop_duplicates()

    # Target klasifikasi
    def label_performa(x):
        if x <= -1:
            return 0  # Menurun
        elif x == 0:
            return 1  # Stabil
        else:
            return 2  # Meningkat

    df["Performa"] = df["Impact_on_Grades"].apply(label_performa)

    fitur_kategorik = [
        "Stream",
        "AI_Tools_Used",
        "Do_Professors_Allow_Use",
        "Device_Used",
        "Internet_Access"
    ]

    fitur_numerik = [
        "Daily_Usage_Hours",
        "Trust_in_AI_Tools"
    ]

    df_model = df[fitur_kategorik + fitur_numerik + ["Performa"]]

    encoders = {}
    for col in fitur_kategorik:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        encoders[col] = le

    X = df_model.drop("Performa", axis=1)
    y = df_model["Performa"]

    return X, y, encoders

# ======================
# MENU GURU
# ======================
if role == "Guru":
    st.header("Dashboard Guru")

    menu = st.selectbox("Menu", ["Lihat Data", "Training Model", "Evaluasi Model"])

    if menu == "Lihat Data":
        st.dataframe(df)

    if menu == "Training Model":
        X, y, encoders = preprocess(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)

        st.session_state.model = model
        st.session_state.encoders = encoders
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test

        st.success("Model berhasil dilatih")

    if menu == "Evaluasi Model":
        if "model" not in st.session_state:
            st.warning("Model belum dilatih")
        else:
            model = st.session_state.model
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test

            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            st.write("Akurasi Model:", round(acc, 3))

            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", ax=ax)
            ax.set_xlabel("Prediksi")
            ax.set_ylabel("Aktual")
            st.pyplot(fig)

            st.text("Laporan Klasifikasi")
            st.text(classification_report(
                y_test, y_pred,
                target_names=["Menurun", "Stabil", "Meningkat"]
            ))

# ======================
# MENU SISWA
# ======================
if role == "Siswa":
    st.header("Dashboard Siswa")

    if "model" not in st.session_state:
        st.warning("Model belum tersedia. Silakan hubungi guru")
    else:
        model = st.session_state.model
        encoders = st.session_state.encoders

        stream = st.selectbox("Jurusan", encoders["Stream"].classes_)
        ai = st.selectbox("Tools AI", encoders["AI_Tools_Used"].classes_)
        izin = st.selectbox("Izin Dosen", encoders["Do_Professors_Allow_Use"].classes_)
        device = st.selectbox("Perangkat", encoders["Device_Used"].classes_)
        internet = st.selectbox("Akses Internet", encoders["Internet_Access"].classes_)

        jam = st.slider("Jam Penggunaan AI per Hari", 0, 10, 3)
        trust = st.slider("Tingkat Kepercayaan terhadap AI", 1, 10, 5)

        if st.button("Analisa"):
            input_data = pd.DataFrame({
                "Stream": [encoders["Stream"].transform([stream])[0]],
                "AI_Tools_Used": [encoders["AI_Tools_Used"].transform([ai])[0]],
                "Do_Professors_Allow_Use": [encoders["Do_Professors_Allow_Use"].transform([izin])[0]],
                "Device_Used": [encoders["Device_Used"].transform([device])[0]],
                "Internet_Access": [encoders["Internet_Access"].transform([internet])[0]],
                "Daily_Usage_Hours": [jam],
                "Trust_in_AI_Tools": [trust]
            })

            pred = model.predict(input_data)[0]

            label = {
                0: "Performa Akademik Menurun",
                1: "Performa Akademik Stabil",
                2: "Performa Akademik Meningkat"
            }

            st.success("Hasil Analisa")
            st.write(label[pred])

