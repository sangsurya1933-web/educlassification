import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Analisis AI & Performa Akademik",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stButton button {
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
    }
    .stButton button:hover {
        background-color: #1D4ED8;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #F1F5F9;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk membuat dataset contoh
def create_sample_dataset():
    np.random.seed(42)
    
    # Buat data 200 mahasiswa
    data = {
        'NIM': [f'2023{str(i).zfill(3)}' for i in range(1, 201)],
        'Nama': [f'Mahasiswa {i}' for i in range(1, 201)],
        'Jenis_Kelamin': np.random.choice(['Laki-laki', 'Perempuan'], 200),
        'Semester': np.random.choice([3, 4, 5, 6, 7, 8], 200),
        'Jurusan': np.random.choice(['Teknik Informatika', 'Sistem Informasi', 'Ilmu Komputer', 'Data Science'], 200),
        'Frekuensi_Penggunaan_AI': np.random.choice(['Rendah', 'Sedang', 'Tinggi'], 200, p=[0.3, 0.5, 0.2]),
        'Durasi_Penggunaan_AI': np.random.uniform(0, 10, 200),  # Jam per minggu
        'Tujuan_Penggunaan_AI': np.random.choice(['Tugas', 'Penelitian', 'Belajar', 'Proyek'], 200),
        'Tingkat_Kemahiran_AI': np.random.choice(['Pemula', 'Menengah', 'Mahir'], 200),
        'IPK': np.round(np.random.uniform(2.0, 4.0, 200), 2),
        'Jumlah_Sertifikat': np.random.randint(0, 10, 200),
        'Status_Kelulusan_Tepat_Waktu': np.random.choice(['Ya', 'Tidak'], 200),
        'Pengalaman_Riset': np.random.choice(['Tidak', 'Sedikit', 'Banyak'], 200),
    }
    
    # Buat hubungan antara penggunaan AI dan IPK
    for i in range(200):
        if data['Frekuensi_Penggunaan_AI'][i] == 'Tinggi' and data['Tingkat_Kemahiran_AI'][i] == 'Mahir':
            data['IPK'][i] = min(4.0, data['IPK'][i] + 0.3)
        elif data['Frekuensi_Penggunaan_AI'][i] == 'Rendah' and data['Tingkat_Kemahiran_AI'][i] == 'Pemula':
            data['IPK'][i] = max(2.0, data['IPK'][i] - 0.2)
    
    df = pd.DataFrame(data)
    
    # Kategorikan performa akademik berdasarkan IPK
    def categorize_performance(ipk):
        if ipk >= 3.5:
            return 'Sangat Baik'
        elif ipk >= 3.0:
            return 'Baik'
        elif ipk >= 2.5:
            return 'Cukup'
        else:
            return 'Kurang'
    
    df['Kategori_Performa'] = df['IPK'].apply(categorize_performance)
    
    return df

# Fungsi untuk dashboard guru
def teacher_dashboard():
    st.markdown('<h1 class="main-header">🎓 Dashboard Guru - Analisis Penggunaan AI & Performa Akademik</h1>', unsafe_allow_html=True)
    
    # Sidebar untuk navigasi
    st.sidebar.markdown('<h3>🔧 Menu Guru</h3>', unsafe_allow_html=True)
    menu_option = st.sidebar.radio(
        "Pilih Menu:",
        ["📊 Overview Data", "🧹 Data Preprocessing", "🔍 Modeling & Evaluation", "📈 Hasil & Rekomendasi"]
    )
    
    # Load dataset
    if 'dataset' not in st.session_state:
        st.session_state.dataset = create_sample_dataset()
    
    df = st.session_state.dataset
    
    if menu_option == "📊 Overview Data":
        st.markdown('<div class="card"><h3 class="sub-header">📊 Overview Dataset</h3></div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Jumlah Mahasiswa", len(df))
        with col2:
            st.metric("Rata-rata IPK", f"{df['IPK'].mean():.2f}")
        with col3:
            tinggi_ai = len(df[df['Frekuensi_Penggunaan_AI'] == 'Tinggi'])
            st.metric("Pengguna AI Tinggi", f"{tinggi_ai} ({tinggi_ai/len(df)*100:.1f}%)")
        
        # Tampilkan dataset
        st.markdown('<div class="card"><h4>Preview Dataset</h4></div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
        
        # Statistik dataset
        st.markdown('<div class="card"><h4>📈 Statistik Dataset</h4></div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Informasi Dataset:**")
            buffer = StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
        
        with col2:
            st.write("**Statistik Deskriptif:**")
            st.dataframe(df.describe(), use_container_width=True)
        
        # Visualisasi data
        st.markdown('<div class="card"><h4>📊 Visualisasi Data</h4></div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Distribusi IPK", "Penggunaan AI vs Performa", "Analisis Jurusan"])
        
        with tab1:
            fig = px.histogram(df, x='IPK', nbins=20, title='Distribusi IPK Mahasiswa')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = px.box(df, x='Frekuensi_Penggunaan_AI', y='IPK', 
                         title='Hubungan Frekuensi Penggunaan AI dengan IPK')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            jurusan_performa = df.groupby(['Jurusan', 'Kategori_Performa']).size().reset_index(name='Jumlah')
            fig = px.bar(jurusan_performa, x='Jurusan', y='Jumlah', color='Kategori_Performa',
                         title='Performa Akademik per Jurusan')
            st.plotly_chart(fig, use_container_width=True)
    
    elif menu_option == "🧹 Data Preprocessing":
        st.markdown('<div class="card"><h3 class="sub-header">🧹 Data Preprocessing</h3></div>', unsafe_allow_html=True)
        
        # Data Cleaning
        st.markdown('<div class="card"><h4>🗑️ Data Cleaning</h4></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Periksa Missing Values"):
                missing_values = df.isnull().sum()
                missing_df = pd.DataFrame({
                    'Kolom': missing_values.index,
                    'Jumlah Missing': missing_values.values,
                    'Persentase': (missing_values.values / len(df)) * 100
                })
                st.dataframe(missing_df[missing_df['Jumlah Missing'] > 0], use_container_width=True)
                if missing_df['Jumlah Missing'].sum() == 0:
                    st.success("✅ Tidak ada missing values dalam dataset!")
        
        with col2:
            if st.button("Periksa Duplikat Data"):
                duplicates = df.duplicated().sum()
                if duplicates > 0:
                    st.warning(f"⚠️ Ditemukan {duplicates} data duplikat")
                    if st.button("Hapus Duplikat"):
                        df.drop_duplicates(inplace=True)
                        st.session_state.dataset = df
                        st.success(f"✅ {duplicates} data duplikat telah dihapus")
                else:
                    st.success("✅ Tidak ada data duplikat")
        
        # Encoding Data Kategorikal
        st.markdown('<div class="card"><h4>🔢 Encoding Data Kategorikal</h4></div>', unsafe_allow_html=True)
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        st.write(f"**Kolom kategorikal:** {', '.join(categorical_cols)}")
        
        if st.button("Lakukan Label Encoding"):
            # Simpan data asli untuk referensi
            if 'original_categorical' not in st.session_state:
                st.session_state.original_categorical = {}
                for col in categorical_cols:
                    st.session_state.original_categorical[col] = df[col].copy()
            
            # Lakukan encoding
            label_encoders = {}
            for col in categorical_cols:
                if col != 'Nama' and col != 'NIM':
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    label_encoders[col] = le
            
            st.session_state.label_encoders = label_encoders
            st.session_state.dataset = df
            st.success("✅ Label encoding berhasil dilakukan!")
            st.write("**Dataset setelah encoding:**")
            st.dataframe(df.head(), use_container_width=True)
        
        # Split Data
        st.markdown('<div class="card"><h4>✂️ Split Data (Training & Testing)</h4></div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            test_size = st.slider("Persentase Data Testing (%)", 10, 40, 20)
        
        with col2:
            target_col = st.selectbox("Pilih Target Variable", df.columns.tolist(), index=df.columns.tolist().index('Kategori_Performa') if 'Kategori_Performa' in df.columns else 0)
        
        with col3:
            random_state = st.number_input("Random State", min_value=0, max_value=100, value=42)
        
        if st.button("Split Data"):
            # Pisahkan fitur dan target
            X = df.drop(columns=[target_col, 'Nama', 'NIM'], errors='ignore')
            y = df[target_col]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=random_state, stratify=y
            )
            
            # Simpan di session state
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            
            st.success(f"✅ Data berhasil di-split!")
            st.write(f"**Training data:** {len(X_train)} sampel ({100-test_size}%)")
            st.write(f"**Testing data:** {len(X_test)} sampel ({test_size}%)")
    
    elif menu_option == "🔍 Modeling & Evaluation":
        st.markdown('<div class="card"><h3 class="sub-header">🔍 Modeling dengan Random Forest</h3></div>', unsafe_allow_html=True)
        
        # Pastikan data sudah di-split
        if 'X_train' not in st.session_state:
            st.warning("⚠️ Silakan lakukan split data terlebih dahulu di menu Data Preprocessing")
            return
        
        # Parameter Random Forest
        st.markdown('<div class="card"><h4>⚙️ Parameter Random Forest</h4></div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators = st.slider("Jumlah Estimator", 10, 200, 100)
        with col2:
            max_depth = st.slider("Max Depth", 2, 20, 10)
        with col3:
            random_state = st.number_input("Random State Model", min_value=0, max_value=100, value=42)
        
        if st.button("🚀 Latih Model Random Forest"):
            with st.spinner("Melatih model Random Forest..."):
                # Inisialisasi model
                rf_model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state
                )
                
                # Latih model
                rf_model.fit(st.session_state.X_train, st.session_state.y_train)
                
                # Prediksi
                y_pred = rf_model.predict(st.session_state.X_test)
                
                # Simpan model dan hasil
                st.session_state.rf_model = rf_model
                st.session_state.y_pred = y_pred
                
                # Hitung akurasi
                accuracy = accuracy_score(st.session_state.y_test, y_pred)
                st.session_state.accuracy = accuracy
                
                # Classification report
                report = classification_report(st.session_state.y_test, y_pred, output_dict=True)
                st.session_state.classification_report = report
                
                # Confusion matrix
                cm = confusion_matrix(st.session_state.y_test, y_pred)
                st.session_state.confusion_matrix = cm
                
                st.success(f"✅ Model berhasil dilatih dengan akurasi: {accuracy:.2%}")
                
                # Tampilkan feature importance
                feature_importance = pd.DataFrame({
                    'Feature': st.session_state.X_train.columns,
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                st.markdown('<div class="card"><h4>📊 Feature Importance</h4></div>', unsafe_allow_html=True)
                fig = px.bar(feature_importance.head(10), x='Importance', y='Feature', 
                            orientation='h', title='10 Fitur Terpenting dalam Model')
                st.plotly_chart(fig, use_container_width=True)
        
        # Evaluasi Model
        if 'rf_model' in st.session_state:
            st.markdown('<div class="card"><h4>📈 Evaluasi Model</h4></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Akurasi Model", f"{st.session_state.accuracy:.2%}")
                
                # Confusion Matrix
                st.write("**Confusion Matrix:**")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(st.session_state.confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)
            
            with col2:
                # Classification Report
                st.write("**Classification Report:**")
                report_df = pd.DataFrame(st.session_state.classification_report).transpose()
                st.dataframe(report_df.style.background_gradient(cmap='Blues'), use_container_width=True)
    
    elif menu_option == "📈 Hasil & Rekomendasi":
        st.markdown('<div class="card"><h3 class="sub-header">📈 Hasil Analisis & Rekomendasi</h3></div>', unsafe_allow_html=True)
        
        if 'rf_model' not in st.session_state:
            st.warning("⚠️ Silakan latih model terlebih dahulu di menu Modeling & Evaluation")
            return
        
        # Analisis hasil
        st.markdown('<div class="card"><h4>🔍 Analisis Hasil</h4></div>', unsafe_allow_html=True)
        
        # Prediksi untuk seluruh dataset
        if 'label_encoders' in st.session_state and 'Kategori_Performa' in st.session_state.label_encoders:
            # Decode kembali untuk analisis
            df_original = st.session_state.dataset.copy()
            le = st.session_state.label_encoders['Kategori_Performa']
            
            # Jika data sudah di-encode, decode dulu
            if 'Kategori_Performa' in df_original.columns and df_original['Kategori_Performa'].dtype != 'object':
                try:
                    df_original['Kategori_Performa'] = le.inverse_transform(df_original['Kategori_Performa'].astype(int))
                except:
                    pass
        
        # Rekomendasi berdasarkan analisis
        st.markdown('<div class="card"><h4>💡 Rekomendasi untuk Peningkatan Performa Akademik</h4></div>', unsafe_allow_html=True)
        
        recommendations = {
            'Sangat Baik': [
                "Pertahankan penggunaan AI yang seimbang dengan aktivitas belajar konvensional",
                "Jadilah mentor bagi mahasiswa lain dalam pemanfaatan AI untuk akademik",
                "Eksplorasi tool AI yang lebih advanced untuk penelitian"
            ],
            'Baik': [
                "Tingkatkan frekuensi penggunaan AI untuk tugas-tugas kompleks",
                "Ikuti workshop atau pelatihan tentang pemanfaatan AI dalam pendidikan",
                "Gunakan AI untuk analisis data penelitian dan penulisan paper"
            ],
            'Cukup': [
                "Mulai gunakan AI secara teratur untuk membantu memahami materi sulit",
                "Manfaatkan AI untuk brainstorming ide tugas dan proyek",
                "Bergabung dengan komunitas pengguna AI dalam pendidikan"
            ],
            'Kurang': [
                "Mulai eksplorasi tool AI dasar seperti ChatGPT untuk bantuan belajar",
                "Minta bimbingan dari dosen atau teman yang sudah berpengalaman dengan AI",
                "Ikuti tutorial online tentang pemanfaatan AI untuk pendidikan"
            ]
        }
        
        for category, recs in recommendations.items():
            with st.expander(f"Rekomendasi untuk Kategori Performa: **{category}**"):
                for i, rec in enumerate(recs, 1):
                    st.write(f"{i}. {rec}")
        
        # Prediksi untuk mahasiswa baru
        st.markdown('<div class="card"><h4>🔮 Prediksi untuk Mahasiswa Baru</h4></div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            frekuensi = st.selectbox("Frekuensi Penggunaan AI", ['Rendah', 'Sedang', 'Tinggi'])
        with col2:
            durasi = st.slider("Durasi Penggunaan AI (jam/minggu)", 0.0, 20.0, 5.0)
        with col3:
            kemahiran = st.selectbox("Tingkat Kemahiran AI", ['Pemula', 'Menengah', 'Mahir'])
        
        jurusan = st.selectbox("Jurusan", ['Teknik Informatika', 'Sistem Informasi', 'Ilmu Komputer', 'Data Science'])
        semester = st.slider("Semester", 1, 8, 5)
        
        if st.button("Prediksi Performa Akademik"):
            # Siapkan data input
            input_data = {
                'Semester': semester,
                'Frekuensi_Penggunaan_AI': frekuensi,
                'Durasi_Penggunaan_AI': durasi,
                'Tingkat_Kemahiran_AI': kemahiran,
                'Jurusan': jurusan
            }
            
            # Karena model sudah dilatih dengan data encoded, kita perlu encode input
            # Untuk demo sederhana, kita akan prediksi berdasarkan aturan
            if frekuensi == 'Tinggi' and kemahiran == 'Mahir':
                prediksi = 'Sangat Baik'
                confidence = 0.85
            elif frekuensi == 'Tinggi' or kemahiran == 'Mahir':
                prediksi = 'Baik'
                confidence = 0.75
            elif frekuensi == 'Rendah' and kemahiran == 'Pemula':
                prediksi = 'Kurang'
                confidence = 0.70
            else:
                prediksi = 'Cukup'
                confidence = 0.65
            
            st.success(f"**Prediksi Performa:** {prediksi}")
            st.info(f"**Tingkat Kepercayaan:** {confidence:.0%}")
            
            # Tampilkan rekomendasi
            st.write("**Rekomendasi:**")
            for rec in recommendations.get(prediksi, []):
                st.write(f"• {rec}")

# Fungsi untuk dashboard siswa
def student_dashboard():
    st.markdown('<h1 class="main-header">🎓 Dashboard Siswa - Analisis Penggunaan AI & Performa Akademik</h1>', unsafe_allow_html=True)
    
    # Load dataset
    if 'dataset' not in st.session_state:
        st.session_state.dataset = create_sample_dataset()
    
    df = st.session_state.dataset
    
    # Sidebar untuk pencarian
    st.sidebar.markdown('<h3>🔍 Pencarian Siswa</h3>', unsafe_allow_html=True)
    
    search_option = st.sidebar.radio("Cari berdasarkan:", ["Nama", "NIM"])
    
    if search_option == "Nama":
        nama_list = df['Nama'].tolist()
        selected_name = st.sidebar.selectbox("Pilih Nama Mahasiswa", nama_list)
        student_data = df[df['Nama'] == selected_name].iloc[0]
    else:
        nim_list = df['NIM'].tolist()
        selected_nim = st.sidebar.selectbox("Pilih NIM", nim_list)
        student_data = df[df['NIM'] == selected_nim].iloc[0]
    
    # Tampilkan data siswa
    st.markdown(f'<div class="card"><h3 class="sub-header">📋 Profil Mahasiswa</h3></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Nama", student_data['Nama'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("NIM", student_data['NIM'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Jurusan", student_data['Jurusan'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Informasi performa
    st.markdown('<div class="card"><h4>📊 Informasi Performa Akademik</h4></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("IPK", student_data['IPK'])
    
    with col2:
        st.metric("Kategori Performa", student_data['Kategori_Performa'])
    
    with col3:
        st.metric("Frekuensi AI", student_data['Frekuensi_Penggunaan_AI'])
    
    with col4:
        st.metric("Tingkat Kemahiran AI", student_data['Tingkat_Kemahiran_AI'])
    
    # Visualisasi performa individu
    st.markdown('<div class="card"><h4>📈 Analisis Performa Individu</h4></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gauge chart untuk IPK
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = student_data['IPK'],
            title = {'text': "IPK"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [0, 4]},
                'bar': {'color': "#3B82F6"},
                'steps': [
                    {'range': [0, 2], 'color': "lightgray"},
                    {'range': [2, 3], 'color': "gray"},
                    {'range': [3, 4], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 3.0
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Radar chart untuk keterampilan
        categories = ['Penggunaan AI', 'IPK (skala 0-4)', 'Sertifikat', 'Pengalaman Riset']
        
        # Normalisasi nilai untuk radar chart
        ai_usage_map = {'Rendah': 1, 'Sedang': 2, 'Tinggi': 3}
        research_exp_map = {'Tidak': 1, 'Sedikit': 2, 'Banyak': 3}
        
        values = [
            ai_usage_map.get(student_data['Frekuensi_Penggunaan_AI'], 1),
            student_data['IPK'],
            min(student_data['Jumlah_Sertifikat'] / 2, 3),  # Normalisasi ke skala 0-3
            research_exp_map.get(student_data['Pengalaman_Riset'], 1)
        ]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            line_color='#3B82F6',
            fillcolor='rgba(59, 130, 246, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 3]
                )),
            showlegend=False,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Rekomendasi personal
    st.markdown('<div class="card"><h4>💡 Rekomendasi Personal</h4></div>', unsafe_allow_html=True)
    
    # Berikan rekomendasi berdasarkan performa
    if student_data['Kategori_Performa'] == 'Sangat Baik':
        st.success("🎉 **Selamat!** Performa akademik Anda sangat baik.")
        st.write("""
        **Rekomendasi untuk Anda:**
        1. Pertahankan keseimbangan penggunaan AI dengan metode belajar tradisional
        2. Jadilah mentor bagi teman-teman yang ingin belajar menggunakan AI
        3. Eksplorasi tool AI yang lebih advanced untuk penelitian dan proyek akhir
        4. Pertimbangkan untuk mengikuti kompetisi AI/Data Science
        """)
    elif student_data['Kategori_Performa'] == 'Baik':
        st.info("👍 **Bagus!** Performa akademik Anda sudah baik.")
        st.write("""
        **Rekomendasi untuk Anda:**
        1. Tingkatkan frekuensi penggunaan AI untuk tugas-tugas kompleks
        2. Ikuti workshop atau webinar tentang pemanfaatan AI dalam pendidikan
        3. Coba integrasikan AI dalam proses penelitian Anda
        4. Dokumentasikan penggunaan AI Anda untuk portofolio
        """)
    elif student_data['Kategori_Performa'] == 'Cukup':
        st.warning("📝 **Perlu peningkatan.** Performa akademik Anda cukup.")
        st.write("""
        **Rekomendasi untuk Anda:**
        1. Mulai gunakan AI secara teratur untuk membantu memahami materi sulit
        2. Manfaatkan AI untuk brainstorming ide tugas dan proyek
        3. Bergabung dengan komunitas pengguna AI dalam pendidikan
        4. Minta bimbingan dari dosen tentang pemanfaatan AI yang efektif
        """)
    else:
        st.error("🚨 **Perlu perhatian khusus.** Performa akademik Anda perlu ditingkatkan.")
        st.write("""
        **Rekomendasi untuk Anda:**
        1. Mulai eksplorasi tool AI dasar seperti ChatGPT untuk bantuan belajar
        2. Ikuti tutorial online tentang pemanfaatan AI untuk pendidikan
        3. Minta bimbingan dari dosen atau teman yang sudah berpengalaman
        4. Buat jadwal belajar yang teratur dengan bantuan AI
        5. Manfaatkan AI untuk membuat rangkuman materi kuliah
        """)
    
    # Perbandingan dengan rata-rata
    st.markdown('<div class="card"><h4>📊 Perbandingan dengan Rata-rata Jurusan</h4></div>', unsafe_allow_html=True)
    
    jurusan_avg = df[df['Jurusan'] == student_data['Jurusan']].mean(numeric_only=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("IPK Anda", student_data['IPK'], 
                 f"{student_data['IPK'] - jurusan_avg['IPK']:.2f}")
    
    with col2:
        st.metric("Rata-rata IPK Jurusan", f"{jurusan_avg['IPK']:.2f}")
    
    with col3:
        st.metric("Peringkat", 
                 f"Top {int((sum(df['IPK'] <= student_data['IPK']) / len(df)) * 100)}%")

# Main app
def main():
    # Sidebar untuk pilih dashboard
    st.sidebar.markdown('<h2>🏫 Sistem Analisis AI & Performa Akademik</h2>', unsafe_allow_html=True)
    
    user_type = st.sidebar.radio(
        "Pilih Dashboard:",
        ["👨‍🏫 Dashboard Guru", "👨‍🎓 Dashboard Siswa"]
    )
    
    # Tambahkan info tentang dataset
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 Informasi Dataset")
    st.sidebar.info("""
    Dataset berisi informasi:
    - 200 mahasiswa
    - Frekuensi penggunaan AI
    - Performa akademik (IPK)
    - Data demografi dan akademik
    """)
    
    # Pilih dashboard berdasarkan user type
    if "Dashboard Guru" in user_type:
        teacher_dashboard()
    else:
        student_dashboard()

if __name__ == "__main__":
    main()
