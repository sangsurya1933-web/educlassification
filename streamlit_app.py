import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set halaman
st.set_page_config(page_title="Analisis AI & Performa Akademik", layout="wide")

# Session state untuk login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_role = None
    st.session_state.model = None
    st.session_state.X_train = None
    st.session_state.X_test = None
    st.session_state.y_train = None
    st.session_state.y_test = None
    st.session_state.y_pred = None
    st.session_state.encoders = {}

# Dataset contoh (ganti dengan dataset asli Anda)
def load_sample_data():
    # Data contoh dengan karakteristik yang relevan
    data = {
        'jam_penggunaan_ai_perhari': [2, 5, 1, 7, 3, 4, 6, 2, 8, 1, 3, 5, 2, 6, 4],
        'frekuensi_penggunaan_mingguan': [3, 7, 2, 10, 4, 5, 8, 3, 12, 1, 4, 6, 2, 9, 5],
        'jenis_ai_digunakan': ['Chatbot', 'Tools', 'Chatbot', 'Tools', 'Chatbot', 'Tools', 'Tools', 'Chatbot', 'Tools', 'Chatbot', 'Tools', 'Tools', 'Chatbot', 'Tools', 'Chatbot'],
        'tugas_menggunakan_ai': ['Tidak', 'Ya', 'Tidak', 'Ya', 'Ya', 'Ya', 'Ya', 'Tidak', 'Ya', 'Tidak', 'Ya', 'Ya', 'Tidak', 'Ya', 'Ya'],
        'ipk': [3.2, 3.8, 2.9, 3.9, 3.4, 3.6, 3.7, 3.1, 4.0, 2.8, 3.5, 3.7, 3.0, 3.8, 3.5],
        'nilai_rata_rata': [75, 85, 70, 90, 78, 82, 88, 72, 95, 68, 80, 85, 71, 87, 81],
        'tingkat_penggunaan_ai': ['Rendah', 'Tinggi', 'Rendah', 'Tinggi', 'Sedang', 'Tinggi', 'Tinggi', 'Rendah', 'Tinggi', 'Rendah', 'Sedang', 'Tinggi', 'Rendah', 'Tinggi', 'Sedang']
    }
    return pd.DataFrame(data)

def login_page():
    st.title("Sistem Analisis AI dan Performa Akademik")
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.subheader("Login")
        role = st.selectbox("Pilih Peran", ["Pilih...", "Guru", "Siswa"])
        
        if role == "Guru":
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Login Guru"):
                if username == "guru" and password == "guru123":
                    st.session_state.logged_in = True
                    st.session_state.user_role = "guru"
                    st.success("Login berhasil sebagai Guru")
                    st.rerun()
                else:
                    st.error("Username atau password salah")
        
        elif role == "Siswa":
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Login Siswa"):
                if username == "siswa" and password == "siswa123":
                    st.session_state.logged_in = True
                    st.session_state.user_role = "siswa"
                    st.success("Login berhasil sebagai Siswa")
                    st.rerun()
                else:
                    st.error("Username atau password salah")

def preprocess_data(df):
    """Fungsi untuk preprocessing data"""
    df_processed = df.copy()
    
    # Encoding variabel kategorikal
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if col not in st.session_state.encoders:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            st.session_state.encoders[col] = le
        else:
            df_processed[col] = st.session_state.encoders[col].transform(df_processed[col])
    
    return df_processed

def train_model():
    """Fungsi untuk melatih model Random Forest"""
    df = load_sample_data()
    df_processed = preprocess_data(df)
    
    # Pisahkan fitur dan target
    X = df_processed.drop('tingkat_penggunaan_ai', axis=1)
    y = df_processed['tingkat_penggunaan_ai']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Simpan ke session state
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    
    # Latih model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Prediksi
    y_pred = model.predict(X_test)
    
    st.session_state.model = model
    st.session_state.y_pred = y_pred
    
    return model, X_test, y_test, y_pred

def show_analysis():
    """Fungsi untuk menampilkan analisis"""
    st.header("Analisis Tingkat Penggunaan AI")
    
    df = load_sample_data()
    
    # Tampilkan data
    with st.expander("Lihat Dataset"):
        st.dataframe(df)
    
    # Statistik deskriptif
    with st.expander("Statistik Deskriptif"):
        st.dataframe(df.describe())
    
    # Distribusi tingkat penggunaan AI
    with st.expander("Distribusi Tingkat Penggunaan AI"):
        fig, ax = plt.subplots(figsize=(8, 4))
        df['tingkat_penggunaan_ai'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title('Distribusi Tingkat Penggunaan AI')
        ax.set_xlabel('Kategori')
        ax.set_ylabel('Jumlah Siswa')
        st.pyplot(fig)
    
    # Korelasi dengan performa akademik
    with st.expander("Hubungan AI dengan Performa Akademik"):
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            df.groupby('tingkat_penggunaan_ai')['ipk'].mean().plot(kind='bar', ax=ax)
            ax.set_title('Rata-rata IPK per Tingkat AI')
            ax.set_xlabel('Tingkat Penggunaan AI')
            ax.set_ylabel('IPK Rata-rata')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            df.groupby('tingkat_penggunaan_ai')['nilai_rata_rata'].mean().plot(kind='bar', ax=ax)
            ax.set_title('Rata-rata Nilai per Tingkat AI')
            ax.set_xlabel('Tingkat Penggunaan AI')
            ax.set_ylabel('Nilai Rata-rata')
            st.pyplot(fig)

def show_evaluation():
    """Fungsi untuk menampilkan evaluasi model"""
    st.header("Evaluasi Model Random Forest")
    
    if st.session_state.model is None:
        st.warning("Model belum dilatih. Silakan lakukan training terlebih dahulu.")
        return
    
    y_test = st.session_state.y_test
    y_pred = st.session_state.y_pred
    
    # Akurasi
    accuracy = accuracy_score(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Akurasi Model", f"{accuracy:.2%}")
    with col2:
        st.metric("Jumlah Data Latih", len(st.session_state.X_train))
    with col3:
        st.metric("Jumlah Data Uji", len(st.session_state.X_test))
    
    # Matriks konfusi
    st.subheader("Matriks Konfusi")
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    
    # Ambil nama kelas dari encoder jika ada
    if 'tingkat_penggunaan_ai' in st.session_state.encoders:
        classes = st.session_state.encoders['tingkat_penggunaan_ai'].classes_
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
    
    ax.set_xlabel('Prediksi')
    ax.set_ylabel('Aktual')
    ax.set_title('Matriks Konfusi')
    st.pyplot(fig)
    
    # Laporan klasifikasi
    st.subheader("Laporan Klasifikasi")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format({
        'precision': '{:.2f}',
        'recall': '{:.2f}',
        'f1-score': '{:.2f}',
        'support': '{:.0f}'
    }))
    
    # Feature importance
    st.subheader("Importance Fitur")
    model = st.session_state.model
    feature_importance = pd.DataFrame({
        'Fitur': st.session_state.X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_importance['Fitur'], feature_importance['Importance'])
    ax.set_xlabel('Importance')
    ax.set_title('Importance Fitur dalam Model')
    st.pyplot(fig)

def prediction_interface():
    """Fungsi untuk prediksi data baru (untuk siswa)"""
    st.header("Prediksi Tingkat Penggunaan AI")
    
    if st.session_state.model is None:
        st.warning("Model belum tersedia. Silakan hubungi guru untuk melatih model.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        jam_penggunaan = st.slider("Jam Penggunaan AI per Hari", 0, 10, 3)
        frekuensi = st.slider("Frekuensi Penggunaan per Minggu", 0, 15, 5)
        jenis_ai = st.selectbox("Jenis AI yang Digunakan", ["Chatbot", "Tools", "Lainnya"])
    
    with col2:
        tugas_ai = st.radio("Menggunakan AI untuk Tugas?", ["Ya", "Tidak"])
        ipk = st.slider("IPK", 2.0, 4.0, 3.5)
        nilai_rata = st.slider("Nilai Rata-rata", 60, 100, 80)
    
    if st.button("Prediksi Tingkat Penggunaan AI"):
        # Siapkan data input
        input_data = pd.DataFrame({
            'jam_penggunaan_ai_perhari': [jam_penggunaan],
            'frekuensi_penggunaan_mingguan': [frekuensi],
            'jenis_ai_digunakan': [jenis_ai],
            'tugas_menggunakan_ai': [tugas_ai],
            'ipk': [ipk],
            'nilai_rata_rata': [nilai_rata]
        })
        
        # Encoding untuk prediksi
        for col in ['jenis_ai_digunakan', 'tugas_menggunakan_ai']:
            if col in st.session_state.encoders:
                input_data[col] = st.session_state.encoders[col].transform(input_data[col])
        
        # Pastikan urutan kolom sama dengan training
        model = st.session_state.model
        X_train_cols = st.session_state.X_train.columns
        input_data = input_data[X_train_cols]
        
        # Prediksi
        prediction = model.predict(input_data)[0]
        
        # Decode hasil
        if 'tingkat_penggunaan_ai' in st.session_state.encoders:
            prediction_decoded = st.session_state.encoders['tingkat_penggunaan_ai'].inverse_transform([prediction])[0]
        else:
            prediction_decoded = prediction
        
        # Tampilkan hasil
        st.success(f"Hasil Prediksi: **{prediction_decoded}**")
        
        # Interpretasi
        interpretasi = {
            'Rendah': 'Penggunaan AI masih minim, berpotensi untuk ditingkatkan',
            'Sedang': 'Penggunaan AI cukup baik, pertahankan dan tingkatkan',
            'Tinggi': 'Penggunaan AI optimal, perlu diimbangi dengan pemahaman konsep'
        }
        
        if prediction_decoded in interpretasi:
            st.info(f"Interpretasi: {interpretasi[prediction_decoded]}")

def guru_dashboard():
    """Dashboard untuk guru"""
    st.sidebar.title("Menu Guru")
    menu = st.sidebar.selectbox(
        "Pilih Menu",
        ["Preprocessing Data", "Analisis Data", "Training Model", "Evaluasi Model", "Prediksi"]
    )
    
    st.title("Dashboard Guru - Analisis AI & Performa Akademik")
    
    if menu == "Preprocessing Data":
        st.header("Preprocessing Data")
        
        df = load_sample_data()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Asli")
            st.dataframe(df)
            
            st.subheader("Informasi Data")
            buffer = pd.io.common.StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
        
        with col2:
            st.subheader("Data Setelah Encoding")
            df_processed = preprocess_data(df)
            st.dataframe(df_processed)
            
            st.subheader("Statistik Data")
            st.dataframe(df_processed.describe())
        
        # Tampilkan mapping encoding
        with st.expander("Lihat Mapping Encoding"):
            for col_name, encoder in st.session_state.encoders.items():
                st.write(f"**{col_name}:**")
                for i, class_name in enumerate(encoder.classes_):
                    st.write(f"  {class_name} → {i}")
    
    elif menu == "Analisis Data":
        show_analysis()
    
    elif menu == "Training Model":
        st.header("Training Model Random Forest")
        
        if st.button("Train Model"):
            with st.spinner("Melatih model..."):
                model, X_test, y_test, y_pred = train_model()
                st.success("Model berhasil dilatih!")
                
                # Tampilkan info model
                st.subheader("Informasi Model")
                st.write(f"Jumlah pohon: {model.n_estimators}")
                st.write(f"Kedalaman maksimum: {model.max_depth}")
                st.write(f"Jumlah fitur: {model.n_features_in_}")
    
    elif menu == "Evaluasi Model":
        show_evaluation()
    
    elif menu == "Prediksi":
        prediction_interface()

def siswa_dashboard():
    """Dashboard untuk siswa"""
    st.sidebar.title("Menu Siswa")
    menu = st.sidebar.selectbox(
        "Pilih Menu",
        ["Analisis Data", "Prediksi Pribadi"]
    )
    
    st.title("Dashboard Siswa - Analisis AI & Performa Akademik")
    
    if menu == "Analisis Data":
        show_analysis()
    
    elif menu == "Prediksi Pribadi":
        prediction_interface()

def main():
    if not st.session_state.logged_in:
        login_page()
    else:
        # Tombol logout
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user_role = None
            st.session_state.model = None
            st.rerun()
        
        if st.session_state.user_role == "guru":
            guru_dashboard()
        elif st.session_state.user_role == "siswa":
            siswa_dashboard()

if __name__ == "__main__":
    main()
