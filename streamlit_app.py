import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis AI Mahasiswa",
    page_icon="🎓",
    layout="centered"
)

# Inisialisasi session state
if 'login' not in st.session_state:
    st.session_state.login = False
if 'role' not in st.session_state:
    st.session_state.role = None

# Knowledge Base Sederhana
KNOWLEDGE_BASE = {
    'RENDAH': {
        'warna': 'hijau',
        'level': 'AMAN',
        'rekomendasi': 'Penggunaan AI dalam batas wajar. Pertahankan!'
    },
    'SEDANG': {
        'warna': 'kuning', 
        'level': 'WASPADA',
        'rekomendasi': 'Penggunaan AI mulai sering. Batasi penggunaan!'
    },
    'TINGGI': {
        'warna': 'merah',
        'level': 'BAHAYA',
        'rekomendasi': 'Penggunaan AI berlebihan. Butuh pengawasan!'
    }
}

# Fungsi bantuan
def buat_data_contoh():
    """Membuat data contoh"""
    data = {
        'Nama': ['Andi', 'Budi', 'Cici', 'Dedi', 'Eka', 'Fani', 'Gina', 'Hadi', 'Ira', 'Joko'],
        'IPK': [3.5, 3.2, 3.8, 2.9, 3.0, 3.6, 3.1, 2.8, 3.4, 3.7],
        'Jam_Belajar': [25, 20, 30, 15, 18, 28, 22, 16, 26, 32],
        'AI_Jam': [5, 10, 3, 20, 15, 8, 25, 30, 7, 12],
        'Kategori': ['RENDAH', 'SEDANG', 'RENDAH', 'TINGGI', 'SEDANG', 'RENDAH', 'TINGGI', 'TINGGI', 'RENDAH', 'SEDANG']
    }
    return pd.DataFrame(data)

def proses_data(df):
    """Preprocessing data"""
    df_clean = df.copy()
    
    # Encoding kategori
    le = LabelEncoder()
    df_clean['Kategori_encoded'] = le.fit_transform(df_clean['Kategori'])
    
    return df_clean, le

# Halaman Login
def halaman_login():
    st.title("🎓 Analisis Penggunaan AI")
    st.write("Sistem analisis penggunaan AI terhadap performa akademik")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Login Guru")
        if st.button("Masuk sebagai Guru", use_container_width=True):
            st.session_state.login = True
            st.session_state.role = "guru"
            st.rerun()
    
    with col2:
        st.subheader("Login Mahasiswa")
        if st.button("Masuk sebagai Mahasiswa", use_container_width=True):
            st.session_state.login = True
            st.session_state.role = "mahasiswa"
            st.rerun()
    
    st.divider()
    st.info("**Pilih peran untuk melanjutkan:**")
    st.write("• **Guru**: Analisis data, training model, rekomendasi")
    st.write("• **Mahasiswa**: Lihat hasil analisis pribadi")

# Dashboard Guru
def dashboard_guru():
    st.title("👨‍🏫 Dashboard Guru")
    
    # Sidebar menu
    with st.sidebar:
        st.write("**Menu**")
        menu = st.radio(
            "Pilih menu:",
            ["Data", "Preprocessing", "Model AI", "Hasil", "Keluar"]
        )
    
    if menu == "Keluar":
        st.session_state.login = False
        st.rerun()
    
    # Menu Data
    if menu == "Data":
        st.header("📊 Data Mahasiswa")
        
        # Upload data
        uploaded_file = st.file_uploader("Upload file CSV", type=['csv'])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success("Data berhasil diupload!")
        else:
            df = buat_data_contoh()
            st.info("Menggunakan data contoh")
        
        # Tampilkan data
        st.write("**Preview Data:**")
        st.dataframe(df, use_container_width=True)
        
        # Statistik sederhana
        st.write("**Statistik:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Jumlah Data", len(df))
        with col2:
            st.metric("Rata-rata IPK", f"{df['IPK'].mean():.2f}")
        with col3:
            st.metric("Rata-rata AI", f"{df['AI_Jam'].mean():.1f} jam")
        
        st.session_state.df = df
    
    # Menu Preprocessing
    elif menu == "Preprocessing":
        if 'df' not in st.session_state:
            st.warning("Silakan upload data terlebih dahulu di menu Data")
            return
        
        st.header("🔧 Preprocessing Data")
        
        df = st.session_state.df
        
        # Data cleaning
        st.subheader("1. Data Cleaning")
        if st.checkbox("Hapus data kosong"):
            df = df.dropna()
            st.write(f"Sisa data: {len(df)} baris")
        
        # Encoding
        st.subheader("2. Encoding Data")
        df_clean, encoder = proses_data(df)
        
        st.write("**Mapping kategori:**")
        for i, kategori in enumerate(encoder.classes_):
            st.write(f"- {kategori} → {i}")
        
        # Split data
        st.subheader("3. Split Data")
        test_size = st.slider("Persentase data testing:", 10, 50, 30)
        
        X = df_clean[['IPK', 'Jam_Belajar', 'AI_Jam']]
        y = df_clean['Kategori_encoded']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=42
        )
        
        st.write(f"**Hasil split:**")
        st.write(f"- Training: {len(X_train)} data")
        st.write(f"- Testing: {len(X_test)} data")
        
        # Simpan ke session
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.encoder = encoder
        st.session_state.df_clean = df_clean
        
        st.success("✅ Preprocessing selesai!")
    
    # Menu Model AI
    elif menu == "Model AI":
        if 'X_train' not in st.session_state:
            st.warning("Silakan lakukan preprocessing terlebih dahulu")
            return
        
        st.header("🤖 Model Random Forest")
        
        # Parameter
        st.subheader("Parameter Model")
        n_trees = st.slider("Jumlah pohon:", 10, 200, 100)
        
        # Training
        if st.button("🚀 Training Model", type="primary"):
            with st.spinner("Training..."):
                # Buat model
                model = RandomForestClassifier(
                    n_estimators=n_trees,
                    random_state=42
                )
                
                # Training
                model.fit(st.session_state.X_train, st.session_state.y_train)
                
                # Prediksi
                y_pred = model.predict(st.session_state.X_test)
                
                # Hitung akurasi
                from sklearn.metrics import accuracy_score
                accuracy = accuracy_score(st.session_state.y_test, y_pred)
                
                # Simpan hasil
                st.session_state.model = model
                st.session_state.y_pred = y_pred
                st.session_state.accuracy = accuracy
                
                st.success(f"✅ Model selesai! Akurasi: {accuracy:.0%}")
        
        # Tampilkan hasil jika sudah training
        if 'model' in st.session_state:
            st.subheader("Hasil Model")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Akurasi", f"{st.session_state.accuracy:.0%}")
            with col2:
                st.metric("Jumlah Prediksi", len(st.session_state.y_pred))
    
    # Menu Hasil
    elif menu == "Hasil":
        if 'model' not in st.session_state:
            st.warning("Silakan training model terlebih dahulu")
            return
        
        st.header("📋 Hasil & Rekomendasi")
        
        # Analisis hasil
        prediksi = st.session_state.y_pred
        encoder = st.session_state.encoder
        
        # Hitung jumlah per kategori
        hasil = {}
        for pred in prediksi:
            kategori = encoder.inverse_transform([pred])[0]
            if kategori not in hasil:
                hasil[kategori] = 0
            hasil[kategori] += 1
        
        # Tampilkan hasil
        st.subheader("Distribusi Hasil")
        
        for kategori, jumlah in hasil.items():
            info = KNOWLEDGE_BASE.get(kategori, {})
            
            col1, col2, col3 = st.columns([1, 2, 3])
            
            with col1:
                # Warna berdasarkan kategori
                if info['warna'] == 'hijau':
                    st.markdown(f"<div style='background-color:#d4edda; padding:10px; border-radius:5px; text-align:center;'><b>{kategori}</b></div>", unsafe_allow_html=True)
                elif info['warna'] == 'kuning':
                    st.markdown(f"<div style='background-color:#fff3cd; padding:10px; border-radius:5px; text-align:center;'><b>{kategori}</b></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='background-color:#f8d7da; padding:10px; border-radius:5px; text-align:center;'><b>{kategori}</b></div>", unsafe_allow_html=True)
            
            with col2:
                st.write(f"**{jumlah}** mahasiswa")
            
            with col3:
                st.write(info['rekomendasi'])
        
        # Visualisasi sederhana
        st.subheader("Grafik Distribusi")
        
        fig, ax = plt.subplots(figsize=(8, 4))
        
        kategori_list = list(hasil.keys())
        jumlah_list = list(hasil.values())
        
        # Warna berdasarkan kategori
        colors = []
        for kategori in kategori_list:
            if KNOWLEDGE_BASE[kategori]['warna'] == 'hijau':
                colors.append('#28a745')
            elif KNOWLEDGE_BASE[kategori]['warna'] == 'kuning':
                colors.append('#ffc107')
            else:
                colors.append('#dc3545')
        
        ax.bar(kategori_list, jumlah_list, color=colors)
        ax.set_xlabel('Kategori')
        ax.set_ylabel('Jumlah Mahasiswa')
        
        st.pyplot(fig)
        
        # Rekomendasi umum
        st.subheader("Rekomendasi Umum")
        
        for kategori, info in KNOWLEDGE_BASE.items():
            st.write(f"**{kategori} ({info['level']})**: {info['rekomendasi']}")

# Dashboard Mahasiswa
def dashboard_mahasiswa():
    st.title("👨‍🎓 Dashboard Mahasiswa")
    
    # Input data mahasiswa
    st.header("Profil Mahasiswa")
    
    col1, col2 = st.columns(2)
    
    with col1:
        nama = st.text_input("Nama Mahasiswa")
        fakultas = st.selectbox("Fakultas", ["Teknik", "Sains", "Ekonomi", "Hukum"])
    
    with col2:
        ipk = st.number_input("IPK", min_value=0.0, max_value=4.0, value=3.0, step=0.1)
        ai_jam = st.number_input("Jam penggunaan AI/minggu", min_value=0, max_value=40, value=10)
    
    if st.button("Analisis Profil Saya", type="primary"):
        # Tentukan kategori berdasarkan jam AI
        if ai_jam < 10:
            kategori = 'RENDAH'
        elif ai_jam < 20:
            kategori = 'SEDANG'
        else:
            kategori = 'TINGGI'
        
        info = KNOWLEDGE_BASE[kategori]
        
        # Tampilkan hasil
        st.divider()
        st.header("Hasil Analisis")
        
        # Kartu profil
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Tampilkan warna berdasarkan kategori
            if info['warna'] == 'hijau':
                st.markdown(f"""
                <div style='background-color:#28a745; color:white; padding:20px; border-radius:10px; text-align:center;'>
                    <h2>{kategori}</h2>
                    <h3>{info['level']}</h3>
                </div>
                """, unsafe_allow_html=True)
            elif info['warna'] == 'kuning':
                st.markdown(f"""
                <div style='background-color:#ffc107; color:black; padding:20px; border-radius:10px; text-align:center;'>
                    <h2>{kategori}</h2>
                    <h3>{info['level']}</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background-color:#dc3545; color:white; padding:20px; border-radius:10px; text-align:center;'>
                    <h2>{kategori}</h2>
                    <h3>{info['level']}</h3>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("Detail Profil")
            st.write(f"**Nama:** {nama}")
            st.write(f"**Fakultas:** {fakultas}")
            st.write(f"**IPK:** {ipk}")
            st.write(f"**Penggunaan AI:** {ai_jam} jam/minggu")
        
        # Rekomendasi
        st.divider()
        st.subheader("📝 Rekomendasi untuk Anda")
        st.info(info['rekomendasi'])
        
        # Tips tambahan
        st.subheader("💡 Tips Penggunaan AI")
        
        if kategori == 'RENDAH':
            tips = [
                "✓ Gunakan AI untuk tugas yang sulit",
                "✓ Eksplorasi tools AI untuk belajar",
                "✓ Tetap jaga kemampuan analisis manual"
            ]
        elif kategori == 'SEDANG':
            tips = [
                "✓ Batasi penggunaan AI maksimal 2 jam/hari",
                "✓ Gunakan AI hanya untuk tugas kompleks",
                "✓ Tingkatkan kemampuan problem-solving mandiri"
            ]
        else:
            tips = [
                "✓ Kurangi penggunaan AI secara bertahap",
                "✓ Konsultasi dengan dosen pembimbing",
                "✓ Ikuti workshop belajar tanpa AI",
                "✓ Cari alternatif metode belajar"
            ]
        
        for tip in tips:
            st.write(tip)
    
    # Knowledge Base
    st.divider()
    st.header("ℹ️ Penjelasan Kategori")
    
    for kategori, info in KNOWLEDGE_BASE.items():
        with st.expander(f"{kategori} - {info['level']}"):
            st.write(f"**Deskripsi:** Penggunaan AI dalam kategori {kategori.lower()}")
            st.write(f"**Rekomendasi:** {info['rekomendasi']}")
            
            if kategori == 'RENDAH':
                st.write("**Contoh:** < 10 jam/minggu")
            elif kategori == 'SEDANG':
                st.write("**Contoh:** 10-20 jam/minggu")
            else:
                st.write("**Contoh:** > 20 jam/minggu")
    
    # Logout button
    st.divider()
    if st.button("Keluar", use_container_width=True):
        st.session_state.login = False
        st.rerun()

# Aplikasi utama
def main():
    if not st.session_state.login:
        halaman_login()
    else:
        if st.session_state.role == "guru":
            dashboard_guru()
        else:
            dashboard_mahasiswa()

if __name__ == "__main__":
    main()
