import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import time

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Analisis AI Mahasiswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS Styling
st.markdown("""
<style>
    /* Reset sidebar */
    section[data-testid="stSidebar"] {
        display: none;
    }
    
    /* Main container */
    .main-container {
        max-width: 100%;
        padding: 20px;
    }
    
    /* Login container */
    .login-container {
        max-width: 400px;
        margin: 100px auto;
        padding: 40px;
        background: white;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Dashboard card */
    .dashboard-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        margin: 15px 0;
        border-left: 5px solid;
    }
    
    .card-aman { border-left-color: #28a745; }
    .card-teguran { border-left-color: #ffc107; }
    .card-pengawasan { border-left-color: #dc3545; }
    
    /* Level badges */
    .level-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        margin: 5px;
    }
    
    .level-aman { background: linear-gradient(135deg, #28a745, #20c997); }
    .level-teguran { background: linear-gradient(135deg, #ffc107, #fd7e14); }
    .level-pengawasan { background: linear-gradient(135deg, #dc3545, #c82333); }
    
    /* Button styling */
    .stButton button {
        width: 100%;
        border-radius: 10px;
        font-weight: bold;
        padding: 10px;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Header */
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #2c3e50, #4a6491);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        margin-bottom: 30px;
        font-weight: bold;
    }
    
    /* Menu sidebar */
    .menu-sidebar {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
        color: white;
        padding: 20px;
        height: 100vh;
        position: fixed;
    }
    
    /* Input fields */
    .stTextInput input {
        border-radius: 10px;
        padding: 10px;
        border: 2px solid #e0e0e0;
    }
    
    .stTextInput input:focus {
        border-color: #3498db;
        box-shadow: 0 0 0 0.2rem rgba(52, 152, 219, 0.25);
    }
    
    /* Table styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Alert boxes */
    .alert-aman {
        background: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .alert-teguran {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .alert-pengawasan {
        background: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Responsive design */
    @media (max-width: 768px) {
        .login-container {
            margin: 50px 20px;
            padding: 30px 20px;
        }
        .main-header {
            font-size: 2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ========== SISTEM LOGIN ==========

# Data user (username: password_hash)
users_db = {
    "guru": {
        "password": hashlib.sha256("guru123".encode()).hexdigest(),
        "role": "guru",
        "nama": "Guru Pembimbing"
    },
    "siswa": {
        "password": hashlib.sha256("siswa123".encode()).hexdigest(),
        "role": "siswa",
        "nama": "Mahasiswa"
    },
    "admin": {
        "password": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "guru",
        "nama": "Administrator"
    }
}

def initialize_session_state():
    """Initialize session state variables"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = ""
    if 'role' not in st.session_state:
        st.session_state.role = ""
    if 'nama' not in st.session_state:
        st.session_state.nama = ""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "login"

def login():
    """Tampilkan halaman login"""
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    
    st.markdown('<h1 style="text-align: center; color: #2c3e50;">🎓</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #2c3e50;">Sistem Analisis AI Mahasiswa</h2>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #7f8c8d;">Silakan login untuk melanjutkan</p>', unsafe_allow_html=True)
    
    # Form login
    username = st.text_input("👤 Username", placeholder="Masukkan username")
    password = st.text_input("🔒 Password", type="password", placeholder="Masukkan password")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        login_button = st.button("🚀 Login", use_container_width=True)
    
    if login_button:
        if username and password:
            if username in users_db:
                password_hash = hashlib.sha256(password.encode()).hexdigest()
                if users_db[username]["password"] == password_hash:
                    # Login berhasil
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.role = users_db[username]["role"]
                    st.session_state.nama = users_db[username]["nama"]
                    st.session_state.current_page = "dashboard"
                    
                    # Tampilkan pesan sukses
                    success_msg = st.success(f"✅ Login berhasil! Selamat datang, {st.session_state.nama}")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Password salah!")
            else:
                st.error("❌ Username tidak ditemukan!")
        else:
            st.warning("⚠️ Harap isi username dan password!")
    
    # Info login
    st.markdown("---")
    st.markdown("**Informasi Login:**")
    st.markdown("""
    - **Guru/Dosen:** `guru` / `guru123`
    - **Mahasiswa:** `siswa` / `siswa123`
    - **Admin:** `admin` / `admin123`
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

def logout():
    """Logout user"""
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.role = ""
    st.session_state.nama = ""
    st.session_state.current_page = "login"
    st.rerun()

# ========== FUNGSI DATA ==========

def create_dataset():
    """Buat dataset contoh"""
    np.random.seed(42)
    
    data = {
        'NIM': [f'202300{i:03d}' for i in range(1, 101)],
        'Nama': [f'Mahasiswa {i}' for i in range(1, 101)],
        'Jurusan': np.random.choice(['Informatika', 'Sistem Informasi', 'Teknik Komputer', 'Data Science'], 100),
        'Semester': np.random.choice([3, 4, 5, 6, 7, 8], 100),
        'Jam_AI_Per_Minggu': np.round(np.random.uniform(5, 40, 100), 1),
        'IPK': np.round(np.random.uniform(2.0, 4.0, 100), 2),
        'Frekuensi_Penggunaan': np.random.choice(['Sangat Jarang', 'Jarang', 'Cukup', 'Sering', 'Sangat Sering'], 100),
        'Tingkat_Kemahiran': np.random.choice(['Pemula', 'Menengah', 'Mahir'], 100),
    }
    
    df = pd.DataFrame(data)
    
    # Tambahkan kolom Level berdasarkan jam penggunaan
    def get_level(jam):
        if jam <= 10:
            return 'AMAN'
        elif jam <= 20:
            return 'PERLU TEGURAN'
        else:
            return 'BUTUH PENGAWASAN'
    
    df['Level'] = df['Jam_AI_Per_Minggu'].apply(get_level)
    
    return df

# ========== DASHBOARD GURU ==========

def dashboard_guru():
    """Dashboard untuk guru/dosen"""
    # Sidebar menu
    with st.sidebar:
        st.markdown(f"### 👋 Halo, {st.session_state.nama}")
        st.markdown("---")
        
        menu = st.radio(
            "📋 Menu",
            ["📊 Dashboard Utama", "👥 Data Mahasiswa", "🔧 Preprocessing", "📈 Analisis", "💡 Rekomendasi", "⚙️ Pengaturan"]
        )
        
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            logout()
    
    # Konten utama berdasarkan menu
    if menu == "📊 Dashboard Utama":
        show_guru_dashboard()
    elif menu == "👥 Data Mahasiswa":
        show_guru_data()
    elif menu == "🔧 Preprocessing":
        show_guru_preprocessing()
    elif menu == "📈 Analisis":
        show_guru_analysis()
    elif menu == "💡 Rekomendasi":
        show_guru_recommendations()
    elif menu == "⚙️ Pengaturan":
        show_guru_settings()

def show_guru_dashboard():
    """Dashboard utama guru"""
    st.markdown('<h1 class="main-header">📊 DASHBOARD GURU</h1>', unsafe_allow_html=True)
    
    # Load data
    df = create_dataset()
    
    # Statistik ringkas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Mahasiswa", len(df))
    
    with col2:
        avg_hours = df['Jam_AI_Per_Minggu'].mean()
        st.metric("Rata-rata Jam AI", f"{avg_hours:.1f} jam/minggu")
    
    with col3:
        avg_ipk = df['IPK'].mean()
        st.metric("Rata-rata IPK", f"{avg_ipk:.2f}")
    
    with col4:
        high_risk = len(df[df['Level'] == 'BUTUH PENGAWASAN'])
        st.metric("Butuh Pengawasan", f"{high_risk} siswa")
    
    # Ringkasan level
    st.markdown("### 📋 Ringkasan Level Pengawasan")
    
    level_counts = df['Level'].value_counts()
    
    col1, col2, col3 = st.columns(3)
    
    level_colors = {
        'AMAN': '#28a745',
        'PERLU TEGURAN': '#ffc107',
        'BUTUH PENGAWASAN': '#dc3545'
    }
    
    for level, count in level_counts.items():
        if level == 'AMAN':
            with col1:
                st.markdown(f'<div class="dashboard-card card-aman">', unsafe_allow_html=True)
                st.markdown(f'<h3>✅ {level}</h3>', unsafe_allow_html=True)
                st.markdown(f'<h2>{count} Siswa</h2>', unsafe_allow_html=True)
                st.markdown(f'<p>{count/len(df)*100:.1f}% dari total</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        elif level == 'PERLU TEGURAN':
            with col2:
                st.markdown(f'<div class="dashboard-card card-teguran">', unsafe_allow_html=True)
                st.markdown(f'<h3>⚠️ {level}</h3>', unsafe_allow_html=True)
                st.markdown(f'<h2>{count} Siswa</h2>', unsafe_allow_html=True)
                st.markdown(f'<p>{count/len(df)*100:.1f}% dari total</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            with col3:
                st.markdown(f'<div class="dashboard-card card-pengawasan">', unsafe_allow_html=True)
                st.markdown(f'<h3>🚨 {level}</h3>', unsafe_allow_html=True)
                st.markdown(f'<h2>{count} Siswa</h2>', unsafe_allow_html=True)
                st.markdown(f'<p>{count/len(df)*100:.1f}% dari total</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Grafik sederhana
    st.markdown("### 📈 Distribusi Penggunaan AI")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram jam penggunaan
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.hist(df['Jam_AI_Per_Minggu'], bins=20, color='steelblue', edgecolor='black')
        ax.axvline(x=10, color='green', linestyle='--', label='Batas Aman')
        ax.axvline(x=20, color='orange', linestyle='--', label='Batas Teguran')
        ax.set_xlabel('Jam AI per Minggu')
        ax.set_ylabel('Jumlah Mahasiswa')
        ax.set_title('Distribusi Jam Penggunaan AI')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        # Scatter plot hubungan AI-IPK
        fig, ax = plt.subplots()
        
        # Warna berdasarkan level
        colors = {'AMAN': 'green', 'PERLU TEGURAN': 'orange', 'BUTUH PENGAWASAN': 'red'}
        
        for level in df['Level'].unique():
            subset = df[df['Level'] == level]
            ax.scatter(subset['Jam_AI_Per_Minggu'], subset['IPK'], 
                      label=level, alpha=0.6, color=colors[level])
        
        ax.set_xlabel('Jam AI per Minggu')
        ax.set_ylabel('IPK')
        ax.set_title('Hubungan Penggunaan AI dengan IPK')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

def show_guru_data():
    """Tampilkan data mahasiswa"""
    st.markdown('<h1 class="main-header">👥 DATA MAHASISWA</h1>', unsafe_allow_html=True)
    
    df = create_dataset()
    
    # Filter dan pencarian
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_jurusan = st.multiselect(
            "Filter Jurusan",
            options=df['Jurusan'].unique(),
            default=df['Jurusan'].unique()
        )
    
    with col2:
        filter_level = st.multiselect(
            "Filter Level",
            options=df['Level'].unique(),
            default=df['Level'].unique()
        )
    
    with col3:
        search = st.text_input("🔍 Cari Nama/NIM")
    
    # Terapkan filter
    filtered_df = df.copy()
    if filter_jurusan:
        filtered_df = filtered_df[filtered_df['Jurusan'].isin(filter_jurusan)]
    if filter_level:
        filtered_df = filtered_df[filtered_df['Level'].isin(filter_level)]
    if search:
        filtered_df = filtered_df[
            filtered_df['Nama'].str.contains(search, case=False) | 
            filtered_df['NIM'].str.contains(search, case=False)
        ]
    
    # Tampilkan data
    st.write(f"**Menampilkan {len(filtered_df)} dari {len(df)} mahasiswa**")
    st.dataframe(filtered_df, use_container_width=True, height=400)
    
    # Opsi ekspor
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Download Data (CSV)"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Klik untuk download",
                data=csv,
                file_name="data_mahasiswa.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("🔄 Refresh Data"):
            st.rerun()

def show_guru_preprocessing():
    """Preprocessing data"""
    st.markdown('<h1 class="main-header">🔧 PREPROCESSING DATA</h1>', unsafe_allow_html=True)
    
    df = create_dataset()
    
    tab1, tab2, tab3 = st.tabs(["🧹 Data Cleaning", "🔢 Encoding", "✂️ Split Data"])
    
    with tab1:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("Data Cleaning")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Cek Missing Values"):
                missing = df.isnull().sum()
                if missing.sum() == 0:
                    st.success("✅ Tidak ada data yang hilang")
                else:
                    st.warning(f"⚠️ Ditemukan {missing.sum()} missing values")
                    st.write(missing[missing > 0])
        
        with col2:
            if st.button("Cek Duplikat"):
                duplicates = df.duplicated().sum()
                if duplicates == 0:
                    st.success("✅ Tidak ada data duplikat")
                else:
                    st.warning(f"⚠️ Ditemukan {duplicates} data duplikat")
        
        # Statistik data
        st.subheader("Statistik Data")
        st.dataframe(df.describe(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("Encoding Data Kategorikal")
        
        st.write("**Kolom kategorikal:**")
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        cat_cols = [col for col in cat_cols if col not in ['Nama', 'NIM']]
        
        for col in cat_cols:
            st.write(f"- {col}: {df[col].nunique()} nilai unik")
        
        if st.button("Lakukan Label Encoding"):
            # Simple encoding untuk demo
            df_encoded = df.copy()
            
            # Encoding untuk kolom tertentu
            for col in cat_cols:
                if col == 'Level':
                    mapping = {'AMAN': 0, 'PERLU TEGURAN': 1, 'BUTUH PENGAWASAN': 2}
                    df_encoded[col] = df_encoded[col].map(mapping)
                else:
                    # Label encoding sederhana
                    unique_vals = df_encoded[col].unique()
                    mapping = {val: i for i, val in enumerate(unique_vals)}
                    df_encoded[col] = df_encoded[col].map(mapping)
            
            st.success("✅ Encoding berhasil!")
            st.dataframe(df_encoded.head(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("Split Data")
        
        test_size = st.slider("Persentase Data Testing", 10, 40, 20)
        
        if st.button("Split Data"):
            train_size = 100 - test_size
            st.success(f"✅ Data berhasil di-split!")
            st.write(f"**Data Training:** {train_size}% ({int(len(df)*train_size/100)} sampel)")
            st.write(f"**Data Testing:** {test_size}% ({int(len(df)*test_size/100)} sampel)")
        st.markdown('</div>', unsafe_allow_html=True)

def show_guru_analysis():
    """Analisis data"""
    st.markdown('<h1 class="main-header">📈 ANALISIS DATA</h1>', unsafe_allow_html=True)
    
    df = create_dataset()
    
    tab1, tab2, tab3 = st.tabs(["Analisis Deskriptif", "Analisis Korelasi", "Analisis Trend"])
    
    with tab1:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("Analisis Deskriptif per Level")
        
        # Group by level
        level_stats = df.groupby('Level').agg({
            'Jam_AI_Per_Minggu': ['mean', 'min', 'max', 'std'],
            'IPK': ['mean', 'min', 'max', 'std'],
            'NIM': 'count'
        }).round(2)
        
        level_stats.columns = ['Rata2_Jam', 'Min_Jam', 'Max_Jam', 'Std_Jam', 
                              'Rata2_IPK', 'Min_IPK', 'Max_IPK', 'Std_IPK', 'Jumlah']
        
        st.dataframe(level_stats, use_container_width=True)
        
        # Insights
        st.subheader("📌 Insights:")
        
        # Hitung korelasi
        correlation = df['Jam_AI_Per_Minggu'].corr(df['IPK'])
        
        if correlation < -0.3:
            st.warning("⚠️ Korelasi negatif kuat terdeteksi antara penggunaan AI dan IPK")
            st.write("Semakin banyak menggunakan AI, IPK cenderung menurun")
        elif correlation < 0:
            st.info("ℹ️ Korelasi negatif lemah terdeteksi")
        elif correlation < 0.3:
            st.info("ℹ️ Korelasi positif lemah terdeteksi")
        else:
            st.success("✅ Korelasi positif kuat terdeteksi")
        
        st.write(f"**Koefisien Korelasi:** {correlation:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("Analisis Korelasi")
        
        # Matrix korelasi sederhana
        numeric_df = df[['Jam_AI_Per_Minggu', 'IPK', 'Semester']]
        corr_matrix = numeric_df.corr()
        
        st.write("**Matriks Korelasi:**")
        st.dataframe(corr_matrix, use_container_width=True)
        
        # Interpretasi
        st.subheader("Interpretasi:")
        st.write("""
        1. **Korelasi Jam_AI - IPK**: Nilai negatif menunjukkan penggunaan AI berlebihan berkaitan dengan IPK rendah
        2. **Korelasi Semester - IPK**: Biasanya positif karena pengalaman belajar
        3. **Korelasi Semester - Jam_AI**: Bisa positif karena mahasiswa senior lebih sering menggunakan AI
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("Analisis Trend per Jurusan")
        
        # Group by jurusan
        jurusan_stats = df.groupby('Jurusan').agg({
            'Jam_AI_Per_Minggu': 'mean',
            'IPK': 'mean',
            'Level': lambda x: (x == 'BUTUH PENGAWASAN').mean() * 100
        }).round(2)
        
        jurusan_stats.columns = ['Rata2_Jam_AI', 'Rata2_IPK', '%_Butuh_Pengawasan']
        
        st.dataframe(jurusan_stats, use_container_width=True)
        
        # Ranking jurusan
        st.subheader("Ranking Jurusan:")
        
        # Berdasarkan penggunaan AI
        sorted_by_ai = jurusan_stats.sort_values('Rata2_Jam_AI', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Peringkat Penggunaan AI Tertinggi:**")
            for i, (jurusan, row) in enumerate(sorted_by_ai.iterrows(), 1):
                st.write(f"{i}. {jurusan}: {row['Rata2_Jam_AI']} jam/minggu")
        
        with col2:
            st.write("**Peringkat IPK Tertinggi:**")
            sorted_by_ipk = jurusan_stats.sort_values('Rata2_IPK', ascending=False)
            for i, (jurusan, row) in enumerate(sorted_by_ipk.iterrows(), 1):
                st.write(f"{i}. {jurusan}: IPK {row['Rata2_IPK']}")
        
        st.markdown('</div>', unsafe_allow_html=True)

def show_guru_recommendations():
    """Rekomendasi untuk guru"""
    st.markdown('<h1 class="main-header">💡 REKOMENDASI PENGAWASAN</h1>', unsafe_allow_html=True)
    
    df = create_dataset()
    
    # Statistik level
    level_counts = df['Level'].value_counts()
    
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("📊 Distribusi Siswa yang Perlu Perhatian")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Butuh Pengawasan", f"{level_counts.get('BUTUH PENGAWASAN', 0)} siswa")
    with col2:
        st.metric("Perlu Teguran", f"{level_counts.get('PERLU TEGURAN', 0)} siswa")
    with col3:
        st.metric("Aman", f"{level_counts.get('AMAN', 0)} siswa")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Rekomendasi berdasarkan level
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("🎯 Rekomendasi Tindakan per Level")
    
    tab1, tab2, tab3 = st.tabs(["🚨 BUTUH PENGAWASAN", "⚠️ PERLU TEGURAN", "✅ AMAN"])
    
    with tab1:
        st.markdown('<div class="alert-pengawasan">', unsafe_allow_html=True)
        st.markdown("### 🚨 LEVEL: BUTUH PENGAWASAN")
        st.write(f"**Jumlah:** {level_counts.get('BUTUH PENGAWASAN', 0)} siswa")
        st.write("**Rekomendasi Tindakan:**")
        st.write("""
        1. **Intervensi Langsung**: Panggil mahasiswa untuk konseling wajib
        2. **Pemantauan Ketat**: Pantau penggunaan AI mingguan
        3. **Program Khusus**: Berikan tugas yang mengurangi ketergantungan AI
        4. **Kerjasama Orang Tua**: Informasikan kondisi kepada orang tua/wali
        5. **Evaluasi Berkala**: Evaluasi perkembangan setiap 2 minggu
        """)
        
        # Tampilkan siswa dengan level ini
        if level_counts.get('BUTUH PENGAWASAN', 0) > 0:
            students = df[df['Level'] == 'BUTUH PENGAWASAN'][['Nama', 'NIM', 'Jurusan', 'Jam_AI_Per_Minggu', 'IPK']]
            st.write("**Daftar Siswa:**")
            st.dataframe(students, use_container_width=True, height=200)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="alert-teguran">', unsafe_allow_html=True)
        st.markdown("### ⚠️ LEVEL: PERLU TEGURAN")
        st.write(f"**Jumlah:** {level_counts.get('PERLU TEGURAN', 0)} siswa")
        st.write("**Rekomendasi Tindakan:**")
        st.write("""
        1. **Peringatan Tertulis**: Berikan surat peringatan pertama
        2. **Konseling Ringan**: Undang untuk diskusi informal
        3. **Pembatasan**: Sarankan batas maksimal 15 jam/minggu
        4. **Mentoring**: Pasangkan dengan senior sebagai mentor
        5. **Monitoring**: Pantau perkembangan bulanan
        """)
        
        if level_counts.get('PERLU TEGURAN', 0) > 0:
            students = df[df['Level'] == 'PERLU TEGURAN'][['Nama', 'NIM', 'Jurusan', 'Jam_AI_Per_Minggu', 'IPK']]
            st.write("**Daftar Siswa:**")
            st.dataframe(students, use_container_width=True, height=200)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="alert-aman">', unsafe_allow_html=True)
        st.markdown("### ✅ LEVEL: AMAN")
        st.write(f"**Jumlah:** {level_counts.get('AMAN', 0)} siswa")
        st.write("**Rekomendasi Tindakan:**")
        st.write("""
        1. **Apresiasi**: Berikan penghargaan atas penggunaan AI yang bertanggung jawab
        2. **Role Model**: Jadikan sebagai contoh untuk siswa lain
        3. **Pengembangan**: Berikan akses ke tool AI yang lebih advance
        4. **Mentoring**: Minta untuk membimbing teman yang kesulitan
        5. **Pemantauan Ringan**: Evaluasi triwulanan cukup
        """)
        
        if level_counts.get('AMAN', 0) > 0:
            students = df[df['Level'] == 'AMAN'][['Nama', 'NIM', 'Jurusan', 'Jam_AI_Per_Minggu', 'IPK']]
            st.write("**Daftar Siswa (contoh 5 siswa):**")
            st.dataframe(students.head(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Rekomendasi umum
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("📋 Rekomendasi Kebijakan Umum")
    
    st.write("""
    **Untuk Institusi:**
    1. **Kebijakan Penggunaan AI**: Buat panduan resmi penggunaan AI dalam akademik
    2. **Workshop Edukasi**: Adakan workshop tentang penggunaan AI yang sehat
    3. **Layanan Konseling**: Sediakan layanan konseling khusus untuk masalah ketergantungan AI
    4. **Monitoring System**: Implementasi sistem monitoring penggunaan AI yang lebih baik
    5. **Kolaborasi dengan Perusahaan AI**: Buat kemitraan untuk tool yang lebih edukatif
    
    **Untuk Dosen:**
    1. **Penugasan yang Seimbang**: Rancang tugas yang membutuhkan pemikiran kritis, bukan hanya pencarian jawaban
    2. **Edukasi tentang Plagiarisme**: Jelaskan batasan penggunaan AI dalam akademik
    3. **Feedback Rutin**: Berikan feedback berkala tentang penggunaan AI
    4. **Alternatif Pembelajaran**: Sediakan metode belajar alternatif selain AI
    """)
    st.markdown('</div>', unsafe_allow_html=True)

def show_guru_settings():
    """Pengaturan untuk guru"""
    st.markdown('<h1 class="main-header">⚙️ PENGATURAN</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("Informasi Akun")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Username:**", st.session_state.username)
        st.write("**Nama:**", st.session_state.nama)
        st.write("**Role:**", st.session_state.role)
    
    with col2:
        st.write("**Status:**", "🟢 Aktif")
        st.write("**Terakhir Login:**", "Hari ini")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("Pengaturan Aplikasi")
    
    # Threshold pengawasan
    st.write("**Atur Threshold Pengawasan:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        aman_threshold = st.number_input("Batas AMAN (jam/minggu)", 1, 50, 10)
    
    with col2:
        teguran_threshold = st.number_input("Batas TEGURAN (jam/minggu)", 1, 50, 20)
    
    with col3:
        st.write(" ")
        st.write(" ")
        if st.button("Simpan Threshold"):
            st.success("✅ Threshold berhasil disimpan!")
    
    # Notifikasi
    st.write("**Pengaturan Notifikasi:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        email_notif = st.checkbox("Email Notifikasi", value=True)
        sms_notif = st.checkbox("SMS Notifikasi", value=False)
    
    with col2:
        if st.button("Simpan Pengaturan Notifikasi"):
            st.success("✅ Pengaturan notifikasi disimpan!")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========== DASHBOARD SISWA ==========

def dashboard_siswa():
    """Dashboard untuk siswa"""
    # Sidebar menu
    with st.sidebar:
        st.markdown(f"### 👋 Halo, {st.session_state.nama}")
        st.markdown("---")
        
        menu = st.radio(
            "📋 Menu",
            ["📊 Dashboard Saya", "📈 Statistik Saya", "💡 Rekomendasi", "ℹ️ Bantuan"]
        )
        
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            logout()
    
    # Konten utama
    if menu == "📊 Dashboard Saya":
        show_siswa_dashboard()
    elif menu == "📈 Statistik Saya":
        show_siswa_stats()
    elif menu == "💡 Rekomendasi":
        show_siswa_recommendations()
    elif menu == "ℹ️ Bantuan":
        show_siswa_help()

def show_siswa_dashboard():
    """Dashboard utama siswa"""
    st.markdown('<h1 class="main-header">👨‍🎓 DASHBOARD SISWA</h1>', unsafe_allow_html=True)
    
    # Data siswa (dummy data untuk demo)
    student_data = {
        'Nama': 'Mahasiswa 1',
        'NIM': '202300001',
        'Jurusan': 'Informatika',
        'Semester': 5,
        'Jam_AI_Per_Minggu': 18.5,
        'IPK': 3.25,
        'Frekuensi_Penggunaan': 'Sering',
        'Tingkat_Kemahiran': 'Menengah',
        'Level': 'PERLU TEGURAN'
    }
    
    # Tampilkan profil
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("👤 Profil Saya")
        
        st.write(f"**Nama:** {student_data['Nama']}")
        st.write(f"**NIM:** {student_data['NIM']}")
        st.write(f"**Jurusan:** {student_data['Jurusan']}")
        st.write(f"**Semester:** {student_data['Semester']}")
        st.write(f"**IPK:** {student_data['IPK']}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("📊 Status Penggunaan AI")
        
        # Tampilkan level
        level = student_data['Level']
        if level == 'AMAN':
            st.markdown('<span class="level-badge level-aman">✅ LEVEL AMAN</span>', unsafe_allow_html=True)
        elif level == 'PERLU TEGURAN':
            st.markdown('<span class="level-badge level-teguran">⚠️ PERLU TEGURAN</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="level-badge level-pengawasan">🚨 BUTUH PENGAWASAN</span>', unsafe_allow_html=True)
        
        # Progress bar
        jam = student_data['Jam_AI_Per_Minggu']
        st.write(f"**Penggunaan AI:** {jam} jam/minggu")
        
        # Batasan: 0-40 jam
        progress = min(jam / 40, 1.0)
        st.progress(progress)
        
        # Keterangan
        st.caption("**Keterangan Level:**")
        st.caption("🟢 AMAN: ≤ 10 jam/minggu")
        st.caption("🟡 TEGURAN: 11-20 jam/minggu")
        st.caption("🔴 PENGAWASAN: > 20 jam/minggu")
        
        # Detail penggunaan
        st.write("**Detail Penggunaan:**")
        st.write(f"- Frekuensi: {student_data['Frekuensi_Penggunaan']}")
        st.write(f"- Tingkat Kemahiran: {student_data['Tingkat_Kemahiran']}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Grafik penggunaan
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("📈 Riwayat Penggunaan AI (Bulan Terakhir)")
    
    # Data dummy untuk grafik
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei']
    usage = [15, 18, 22, 18, 17]
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    
    bars = ax.bar(months, usage, color=['green' if x <= 10 else 'orange' if x <= 20 else 'red' for x in usage])
    ax.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='Batas Aman')
    ax.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Batas Pengawasan')
    ax.set_xlabel('Bulan')
    ax.set_ylabel('Jam AI per Minggu')
    ax.set_title('Trend Penggunaan AI')
    ax.legend()
    
    # Tambahkan label nilai di atas bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

def show_siswa_stats():
    """Statistik siswa"""
    st.markdown('<h1 class="main-header">📈 STATISTIK SAYA</h1>', unsafe_allow_html=True)
    
    # Data siswa
    student_data = {
        'Jam_AI_Per_Minggu': 18.5,
        'IPK': 3.25,
        'Level': 'PERLU TEGURAN'
    }
    
    # Load data untuk perbandingan
    df = create_dataset()
    jurusan_siswa = 'Informatika'
    jurusan_data = df[df['Jurusan'] == jurusan_siswa]
    
    # Statistik perbandingan
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rata_jurusan = jurusan_data['Jam_AI_Per_Minggu'].mean()
        selisih_jam = student_data['Jam_AI_Per_Minggu'] - rata_jurusan
        st.metric(
            "Jam AI/Minggu", 
            f"{student_data['Jam_AI_Per_Minggu']} jam",
            f"{selisih_jam:+.1f} dari rata jurusan"
        )
    
    with col2:
        rata_ipk_jurusan = jurusan_data['IPK'].mean()
        selisih_ipk = student_data['IPK'] - rata_ipk_jurusan
        st.metric(
            "IPK",
            f"{student_data['IPK']}",
            f"{selisih_ipk:+.2f} dari rata jurusan"
        )
    
    with col3:
        # Hitung peringkat
        peringkat_ipk = (df['IPK'] >= student_data['IPK']).sum() / len(df) * 100
        st.metric(
            "Peringkat IPK",
            f"Top {peringkat_ipk:.0f}%",
            f"Dari {len(df)} mahasiswa"
        )
    
    # Perbandingan dengan jurusan
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("📊 Perbandingan dengan Jurusan")
    
    # Buat tabel perbandingan
    comparison_data = {
        'Metrik': ['Jam AI/Minggu', 'IPK', 'Level'],
        'Anda': [student_data['Jam_AI_Per_Minggu'], student_data['IPK'], student_data['Level']],
        'Rata-rata Jurusan': [round(jurusan_data['Jam_AI_Per_Minggu'].mean(), 1), 
                            round(jurusan_data['IPK'].mean(), 2),
                            'N/A'],
        'Status': [
            '✅ Di bawah rata-rata' if student_data['Jam_AI_Per_Minggu'] < rata_jurusan else '⚠️ Di atas rata-rata',
            '✅ Di atas rata-rata' if student_data['IPK'] > rata_ipk_jurusan else '⚠️ Di bawah rata-rata',
            'Lihat rekomendasi'
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Distribusi level di jurusan
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("🎯 Distribusi Level di Jurusan Anda")
    
    level_counts_jurusan = jurusan_data['Level'].value_counts()
    
    # Buat pie chart
    fig, ax = plt.subplots()
    colors = {'AMAN': '#28a745', 'PERLU TEGURAN': '#ffc107', 'BUTUH PENGAWASAN': '#dc3545'}
    level_colors = [colors.get(level, '#6c757d') for level in level_counts_jurusan.index]
    
    wedges, texts, autotexts = ax.pie(
        level_counts_jurusan.values, 
        labels=level_counts_jurusan.index,
        autopct='%1.1f%%',
        colors=level_colors,
        startangle=90
    )
    
    # Highlight bagian sesuai level siswa
    for i, level in enumerate(level_counts_jurusan.index):
        if level == student_data['Level']:
            wedges[i].set_edgecolor('black')
            wedges[i].set_linewidth(2)
    
    ax.set_title(f'Distribusi Level di Jurusan {jurusan_siswa}')
    st.pyplot(fig)
    
    # Tampilkan posisi siswa
    st.write(f"**Posisi Anda:** Termasuk dalam kategori **{student_data['Level']}**")
    st.write(f"**Jumlah mahasiswa {student_data['Level']} di jurusan Anda:** {level_counts_jurusan.get(student_data['Level'], 0)} orang")
    st.markdown('</div>', unsafe_allow_html=True)

def show_siswa_recommendations():
    """Rekomendasi untuk siswa"""
    st.markdown('<h1 class="main-header">💡 REKOMENDASI UNTUK ANDA</h1>', unsafe_allow_html=True)
    
    # Data siswa
    student_data = {
        'Jam_AI_Per_Minggu': 18.5,
        'IPK': 3.25,
        'Level': 'PERLU TEGURAN',
        'Frekuensi_Penggunaan': 'Sering',
        'Tingkat_Kemahiran': 'Menengah'
    }
    
    level = student_data['Level']
    
    # Tampilkan level
    if level == 'AMAN':
        st.markdown('<div class="alert-aman">', unsafe_allow_html=True)
        st.markdown("### ✅ STATUS: LEVEL AMAN")
        st.write("Penggunaan AI Anda dalam batas wajar. Pertahankan!")
        st.markdown('</div>', unsafe_allow_html=True)
    elif level == 'PERLU TEGURAN':
        st.markdown('<div class="alert-teguran">', unsafe_allow_html=True)
        st.markdown("### ⚠️ STATUS: PERLU TEGURAN")
        st.write("Penggunaan AI Anda mulai berlebihan. Perlu dikurangi!")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-pengawasan">', unsafe_allow_html=True)
        st.markdown("### 🚨 STATUS: BUTUH PENGAWASAN")
        st.write("Penggunaan AI Anda berlebihan! Segera konsultasi dengan dosen.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Rekomendasi spesifik
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("🎯 Rencana Aksi Rekomendasi")
    
    if level == 'AMAN':
        st.write("""
        **Target 1 Bulan Ke Depan:**
        1. ✅ Pertahankan penggunaan ≤ 10 jam/minggu
        2. ✅ Dokumentasikan penggunaan AI untuk portofolio
        3. ✅ Bagikan tips penggunaan sehat ke teman
        4. ✅ Eksplorasi tool AI baru untuk skill development
        
        **Aksi Spesifik:**
        - Catat penggunaan AI harian di aplikasi
        - Ikuti 1 workshop tentang AI dalam 1 bulan
        - Buat proyek kecil dengan bantuan AI
        """)
    
    elif level == 'PERLU TEGURAN':
        st.write("""
        **Target 1 Bulan Ke Depan:**
        1. ⚠️ Turunkan penggunaan ke ≤ 15 jam/minggu
        2. ⚠️ Konsultasi dengan dosen wali
        3. ⚠️ Ikuti workshop "Penggunaan AI Sehat"
        4. ⚠️ Buat jadwal belajar tanpa AI
        
        **Aksi Spesifik:**
        - Kurangi 30 menit penggunaan AI per hari
        - Jadwalkan konsultasi minggu depan
        - Catat situasi saat paling tergantung AI
        - Cari alternatif metode belajar
        """)
    
    else:  # BUTUH PENGAWASAN
        st.write("""
        **Target 2 Minggu Ke Depan:**
        1. 🚨 Segera konsultasi dengan dosen wali
        2. 🚨 Kurangi penggunaan menjadi ≤ 10 jam/minggu
        3. 🚨 Ikuti program bimbingan khusus
        4. 🚨 Laporkan perkembangan mingguan
        
        **Aksi Spesifik:**
        - Hubungi dosen wali HARI INI
        - Buat komitmen tertulis untuk mengurangi AI
        - Ikut sesi konseling kampus
        - Minta teman untuk mengingatkan
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tips penggunaan AI yang sehat
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("🌱 Tips Penggunaan AI yang Sehat")
    
    tips = [
        "1. **Gunakan untuk brainstorming**, bukan mengerjakan seluruh tugas",
        "2. **Selalu verifikasi** informasi dari AI dengan sumber lain",
        "3. **Batasi waktu** penggunaan maksimal 2 jam per sesi",
        "4. **Fokus pada pemahaman konsep**, bukan hanya mencari jawaban",
        "5. **Gunakan AI sebagai asisten**, bukan pengganti pemikiran",
        "6. **Dokumentasikan** apa yang dipelajari dari AI",
        "7. **Diskusikan** penggunaan AI dengan dosen/teman",
        "8. **Istirahat** 10 menit setiap 50 menit penggunaan",
        "9. **Eksplorasi tool** yang mendukung pembelajaran, bukan shortcut",
        "10. **Evaluasi berkala** pengaruh AI terhadap pemahaman Anda"
    ]
    
    for tip in tips:
        st.write(tip)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Resources tambahan
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("📚 Resources Tambahan")
    
    resources = {
        "Workshop Kampus": "Penggunaan AI dalam Akademik (setiap Jumat)",
        "Konseling": "Layanan Konseling Mahasiswa (Gedung B Lantai 3)",
        "Buku Panduan": "Pedoman Penggunaan AI untuk Mahasiswa",
        "Forum Diskusi": "Forum Mahasiswa Teknologi Informasi",
        "Kelas Online": "AI Literacy for Students (Coursera)"
    }
    
    for resource, desc in resources.items():
        with st.expander(f"📖 {resource}"):
            st.write(desc)
            if st.button(f"Akses {resource}", key=resource):
                st.info(f"Mengarahkan ke {resource}...")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_siswa_help():
    """Halaman bantuan untuk siswa"""
    st.markdown('<h1 class="main-header">ℹ️ BANTUAN</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("❓ Cara Menggunakan Dashboard")
    
    faq = {
        "Bagaimana cara melihat level saya?": "Level dapat dilihat di menu 'Dashboard Saya' atau 'Statistik Saya'",
        "Apa arti dari masing-masing level?": """
        - **AMAN**: Penggunaan AI ≤ 10 jam/minggu (sehat)
        - **PERLU TEGURAN**: 11-20 jam/minggu (perlu perhatian)
        - **BUTUH PENGAWASAN**: > 20 jam/minggu (butuh intervensi)
        """,
        "Bagaimana cara memperbaiki level saya?": "Ikuti rekomendasi di menu 'Rekomendasi' dan konsultasi dengan dosen",
        "Data saya salah, bagaimana memperbaikinya?": "Hubungi admin sistem atau dosen pembimbing",
        "Apa yang harus dilakukan jika level saya 'BUTUH PENGAWASAN'?": "Segera konsultasi dengan dosen wali dan ikuti program bimbingan",
    }
    
    for question, answer in faq.items():
        with st.expander(f"❔ {question}"):
            st.write(answer)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Kontak bantuan
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("📞 Kontak Bantuan")
    
    contacts = {
        "Dosen Pembimbing": "Dr. Ahmad, M.Kom (ahmad@kampus.ac.id)",
        "Admin Sistem": "Budi, S.Kom (budi@kampus.ac.id)",
        "Konseling Mahasiswa": "Gedung B Lantai 3 (08:00-16:00)",
        "Layanan Darurat": "Telepon: (021) 12345678",
    }
    
    for position, contact in contacts.items():
        st.write(f"**{position}:** {contact}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========== MAIN APP ==========

def main():
    """Aplikasi utama"""
    initialize_session_state()
    
    # Tampilkan halaman berdasarkan status login
    if not st.session_state.logged_in:
        login()
    else:
        if st.session_state.role == "guru":
            dashboard_guru()
        else:  # siswa
            dashboard_siswa()

if __name__ == "__main__":
    main()
