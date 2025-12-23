import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ======================
# KONFIGURASI HALAMAN
# ======================
st.set_page_config(
    page_title="Analisis Penggunaan AI terhadap Performa Akademik",
    layout="wide"
)

st.title("📊 Analisis Tingkat Penggunaan AI terhadap Performa Akademik Siswa")
st.caption("Algoritma Random Forest")

# ======================
# LOAD DATASET
# ======================
@st.cache_data
def load_data():
    return pd.read_csv("Students tabel.csv", sep=";")

df = load_data()

# ======================
# SIDEBAR MENU
# ======================
menu = st.sidebar.selectbox(
    "Menu Navigasi",
    ["Dataset", "Preprocessing", "Pemodelan Random Forest", "Evaluasi Model"]
)

# ======================
# 1. DATASET
# ======================
if menu == "Dataset":
    st.subheader("📂 Dataset Siswa")
    st.dataframe(df)
    st.write("Jumlah Data:", df.shape[0])
    st.write("Jumlah Fitur:", df.shape[1])

# ======================
# 2. PREPROCESSING
# ======================
elif menu == "Preprocessing":
    st.subheader("⚙️ Pra-pemrosesan Data")

    df_clean = df.drop_duplicates()

    df_model = df_clean.drop(columns=["Student_Name", "College_Name", "State"])

    le = LabelEncoder()
    for col in df_model.select_dtypes(include="object").columns:
        df_model[col] = le.fit_transform(df_model[col])

    X = df_model.drop(columns=["Impact_on_Grades"])
    y = df_model["Impact_on_Grades"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.success("Preprocessing selesai.")
    st.write("Data setelah preprocessing:")
    st.dataframe(pd.DataFrame(X_scaled, columns=X.columns).head())

# ======================
# 3. PEMODELAN
# ======================
elif menu == "Pemodelan Random Forest":
    st.subheader("🌲 Pemodelan Random Forest")

    df_model = df.drop(columns=["Student_Name", "College_Name", "State"])
    for col in df_model.select_dtypes(include="object").columns:
        df_model[col] = LabelEncoder().fit_transform(df_model[col])

    X = df_model.drop(columns=["Impact_on_Grades"])
    y = df_model["Impact_on_Grades"]

    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    n_tree = st.slider("Jumlah Pohon (n_estimators)", 50, 300, 100)

    model = RandomForestClassifier(
        n_estimators=n_tree,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.success("Model berhasil dilatih.")

# ======================
# 4. EVALUASI
# ======================
elif menu == "Evaluasi Model":
    st.subheader("📈 Evaluasi Model Random Forest")

    df_model = df.drop(columns=["Student_Name", "College_Name", "State"])
    for col in df_model.select_dtypes(include="object").columns:
        df_model[col] = LabelEncoder().fit_transform(df_model[col])

    X = df_model.drop(columns=["Impact_on_Grades"])
    y = df_model["Impact_on_Grades"]

    X = StandardScaler().fit_transform(X)

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

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    st.pyplot(fig)

    # Feature Importance
    importances = model.feature_importances_
    fig2, ax2 = plt.subplots()
    ax2.barh(X.shape[1]*[""], importances)
    ax2.set_title("Feature Importance")
    st.pyplot(fig2)
