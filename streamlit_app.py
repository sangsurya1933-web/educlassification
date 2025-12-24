import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Analisis Penggunaan AI", layout="wide")

# ===============================
# LOGIN
# ===============================
st.title("Aplikasi Analisis Penggunaan AI terhadap Performa Akademik")

role = st.sidebar.selectbox("Login Sebagai", ["Guru", "Siswa"])

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("Students_tabel.csv")
    return df

df = load_data()

# ===============================
# PREPROCESSING FUNCTION
# ===============================
def preprocessing(df):
    df = df.dropna()
    df = df.drop_duplicates()

    df["Klasifikasi_Performa"] = df["Impact_on_Grades"].apply(
        lambda x: "Menurun" if x <= -1 else "Stabil" if x == 0 else "Meningkat"
    )

    fitur = [
        "Stream",
        "AI_Tools_Used",
        "Daily_Usage_Hours",
        "Trust_in_AI_Tools",
        "Do_Professors_Allow_Use",
        "Device_Used",
        "Internet_Access"
    ]

    df_model = df[fitur + ["Klasifikasi_Performa"]]

    encoder = LabelEncoder()
    for col in df_model.columns:
        df_model[col] = encoder.fit_transform(df_model[col])

    X = df_model.drop("Klasifikasi_Performa", axis=1)
    y = df_model["Klasifikasi_Performa"]

    return X, y, df_model

# ===============================
# MENU GURU
# ===============================
if role == "Guru":
    st.header("Dashboard Guru")

    menu = st.selectbox("Menu", ["Data", "Preprocessing dan Training", "Evaluasi Model"])

    if menu == "Data":
        st.subheader("Data Dataset")
        st.dataframe(df)

    if menu == "Preprocessing dan Training":
        st.subheader("Preprocessing Data dan Training Model")

        X, y, df_model = preprocessing(df)

        test_size = st.slider("Proporsi Data Uji", 0.1, 0.4, 0.2)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        st.success("Model Random Forest berhasil dilatih")

        st.session_state["model"] = model
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test

    if menu == "Evaluasi Model":
        st.subheader("Evaluasi Model")

        if "model" in st.session_state:
            model = st.session_state["model"]
            X_test = st.session_state["X_test"]
            y_test = st.session_state["y_test"]

            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            st.write("Akurasi Model:", acc)

            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Prediksi")
            ax.set_ylabel("Aktual")
            st.pyplot(fig)

            st.text("Laporan Klasifikasi")
            st.text(classification_report(y_test, y_pred))
        else:
            st.warning("Silakan lakukan training terlebih dahulu")

# ===============================
# MENU SISWA
# ===============================
if role == "Siswa":
    st.header("Dashboard Siswa")
    st.subheader("Analisa Performa Akademik Berdasarkan Penggunaan AI")

    if "model" not in st.session_state:
        st.warning("Model belum tersedia. Silakan hubungi guru")
    else:
        model = st.session_state["model"]

        stream = st.selectbox("Jurusan", df["Stream"].unique())
        ai_tool = st.selectbox("Tools AI", df["AI_Tools_Used"].unique())
        jam = st.slider("Jam Penggunaan AI per Hari", 0, 10, 3)
        trust = st.slider("Tingkat Kepercayaan AI", 1, 10, 5)
        izin = st.selectbox("Dosen Mengizinkan AI", ["Yes", "No"])
        device = st.selectbox("Perangkat", df["Device_Used"].unique())
        internet = st.selectbox("Akses Internet", df["Internet_Access"].unique())

        input_df = pd.DataFrame({
            "Stream": [stream],
            "AI_Tools_Used": [ai_tool],
            "Daily_Usage_Hours": [jam],
            "Trust_in_AI_Tools": [trust],
            "Do_Professors_Allow_Use": [izin],
            "Device_Used": [device],
            "Internet_Access": [internet]
        })

        encoder = LabelEncoder()
        for col in input_df.columns:
            input_df[col] = encoder.fit_transform(input_df[col])

        if st.button("Analisa"):
            hasil = model.predict(input_df)[0]

            label = {0: "Menurun", 1: "Stabil", 2: "Meningkat"}
            st.success("Hasil Klasifikasi Performa Akademik")
            st.write(label[hasil])
