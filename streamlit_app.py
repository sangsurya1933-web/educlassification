import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Sistem Pemantauan Penggunaan AI - Mahasiswa",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set custom CSS dengan tema gradien untuk level pengawasan
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        padding: 10px;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #3B82F6;
        padding-left: 15px;
    }
    .card {
        background-color: #FFFFFF;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        border: 1px solid #E5E7EB;
    }
    .level-card {
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        color: white;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s;
    }
    .level-card:hover {
        transform: translateY(-5px);
    }
    .level-aman {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
    }
    .level-teguran {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
    }
    .level-pengawasan {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
    }
    .metric-card {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 5px 15px rgba(59, 130, 246, 0.3);
    }
    .stButton button {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 0.75rem 1.5rem;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 20px rgba(59, 130, 246, 0.4);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%);
    }
    .warning-box {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        border-left: 5px solid #F59E0B;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .danger-box {
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
        border-left: 5px solid #EF4444;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%);
        border-left: 5px solid #10B981;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #10B981 0%, #3B82F6 50%, #EF4444 100%);
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk membuat dataset contoh dengan fokus pada penggunaan AI
def create_sample_dataset():
    np.random.seed(42)
    
    # Buat data 250 mahasiswa
    data = {
        'NIM': [f'AI2023{str(i).zfill(3)}' for i in range(1, 251)],
        'Nama': [f'Mahasiswa {i}' for i in range(1, 251)],
        'Jenis_Kelamin': np.random.choice(['Laki-laki', 'Perempuan'], 250),
        'Semester': np.random.choice([3, 4, 5, 6, 7, 8], 250),
        'Jurusan': np.random.choice(['AI & Machine Learning', 'Data Science', 'Computer Science', 'IT'], 250),
        'Frekuensi_Penggunaan_AI_per_minggu': np.random.choice([1, 2, 3, 4, 5, 6, 7], 250),  # Hari per minggu
        'Durasi_Penggunaan_AI_jam_per_hari': np.round(np.random.uniform(0.5, 8, 250), 1),  # Jam per hari
        'Tujuan_Penggunaan_AI': np.random.choice(['Mengerjakan Tugas', 'Penelitian', 'Belajar Mandiri', 'Proyek Akhir', 'Kompetisi'], 250),
        'Tingkat_Kemahiran_AI': np.random.choice(['Pemula', 'Menengah', 'Mahir', 'Expert'], 250),
        'Jenis_AI_yang_Digunakan': np.random.choice(['ChatGPT', 'Copilot', 'Bard', 'Claude', 'Multiple Tools'], 250),
        'IPK': np.round(np.random.uniform(2.0, 4.0, 250), 2),
        'Motivasi_Penggunaan_AI': np.random.choice(['Efisiensi', 'Kesulitan Materi', 'Trend', 'Rekomendasi Dosen'], 250),
        'Dampak_Pada_Pemahaman': np.random.choice(['Meningkat', 'Menurun', 'Tidak Berubah'], 250),
        'Keterlibatan_Dosen': np.random.choice(['Tinggi', 'Sedang', 'Rendah'], 250),
    }
    
    df = pd.DataFrame(data)
    
    # Hitung total penggunaan AI per minggu (jam)
    df['Total_Penggunaan_AI_jam_per_minggu'] = df['Frekuensi_Penggunaan_AI_per_minggu'] * df['Durasi_Penggunaan_AI_jam_per_hari']
    
    # Kategorikan level pengawasan berdasarkan penggunaan AI
    def categorize_supervision_level(row):
        total_hours = row['Total_Penggunaan_AI_jam_per_minggu']
        frequency = row['Frekuensi_Penggunaan_AI_per_minggu']
        
        # Kriteria level pengawasan
        if total_hours <= 10 and frequency <= 3:
            return 'AMAN'
        elif total_hours <= 20 and frequency <= 5:
            return 'PERLU TEGURAN'
        else:
            return 'BUTUH PENGAWASAN'
    
    df['Level_Pengawasan'] = df.apply(categorize_supervision_level, axis=1)
    
    # Pengaruh penggunaan AI terhadap IPK (untuk analisis)
    for i in range(len(df)):
        if df.loc[i, 'Level_Pengawasan'] == 'BUTUH PENGAWASAN':
            # Penggunaan AI berlebihan cenderung menurunkan IPK
            if np.random.random() > 0.3:
                df.loc[i, 'IPK'] = max(2.0, df.loc[i, 'IPK'] - np.random.uniform(0.1, 0.5))
        elif df.loc[i, 'Level_Pengawasan'] == 'AMAN':
            # Penggunaan AI wajar cenderung meningkatkan IPK
            if np.random.random() > 0.6:
                df.loc[i, 'IPK'] = min(4.0, df.loc[i, 'IPK'] + np.random.uniform(0.1, 0.3))
    
    return df

# Fungsi untuk menampilkan level pengawasan dengan visual yang menarik
def display_supervision_level(level):
    if level == 'AMAN':
        return st.markdown(f"""
        <div class="level-card level-aman">
            <h3 style="color: white; margin: 0;">✅ LEVEL AMAN</h3>
            <p style="color: white; margin: 5px 0 0 0;">Penggunaan AI dalam batas wajar</p>
        </div>
        """, unsafe_allow_html=True)
    elif level == 'PERLU TEGURAN':
        return st.markdown(f"""
        <div class="level-card level-teguran">
            <h3 style="color: white; margin: 0;">⚠️ LEVEL PERLU TEGURAN</h3>
            <p style="color: white; margin: 5px 0 0 0;">Penggunaan AI mulai berlebihan, perlu peringatan</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        return st.markdown(f"""
        <div class="level-card level-pengawasan">
            <h3 style="color: white; margin: 0;">🚨 LEVEL BUTUH PENGAWASAN</h3>
            <p style="color: white; margin: 5px 0 0 0;">Penggunaan AI berlebihan, butuh intervensi</p>
        </div>
        """, unsafe_allow_html=True)

# Inisialisasi session state
if 'dataset' not in st.session_state:
    st.session_state.dataset = create_sample_dataset()
    st.session_state.model_trained = False
    st.session_state.data_processed = False

# Fungsi untuk dashboard guru
def teacher_dashboard():
    st.markdown('<h1 class="main-header">🤖 DASHBOARD GURU - SISTEM PEMANTAUAN PENGGUNAAN AI MAHASISWA</h1>', unsafe_allow_html=True)
    
    # Sidebar untuk navigasi
    st.sidebar.markdown('<h3>🔧 MENU GURU</h3>', unsafe_allow_html=True)
    menu_option = st.sidebar.radio(
        "Pilih Menu:",
        ["📊 OVERVIEW DATA", "🧹 PREPROCESSING DATA", "🤖 MODELING & EVALUASI", "📈 REKOMENDASI PENGAWASAN"]
    )
    
    df = st.session_state.dataset
    
    if menu_option == "📊 OVERVIEW DATA":
        st.markdown('<div class="card"><h3 class="sub-header">📊 ANALISIS DISTRIBUSI PENGGUNAAN AI</h3></div>', unsafe_allow_html=True)
        
        # Statistik utama
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_hours = df['Total_Penggunaan_AI_jam_per_minggu'].sum()
            st.metric("Total Jam AI/Minggu", f"{total_hours:.0f} jam")
        
        with col2:
            avg_hours = df['Total_Penggunaan_AI_jam_per_minggu'].mean()
            st.metric("Rata-rata Jam/Minggu", f"{avg_hours:.1f} jam")
        
        with col3:
            high_usage = len(df[df['Level_Pengawasan'] == 'BUTUH PENGAWASAN'])
            st.metric("Butuh Pengawasan", f"{high_usage} ({high_usage/len(df)*100:.1f}%)")
        
        with col4:
            safe_usage = len(df[df['Level_Pengawasan'] == 'AMAN'])
            st.metric("Level Aman", f"{safe_usage} ({safe_usage/len(df)*100:.1f}%)")
        
        # Tampilkan distribusi level pengawasan
        st.markdown('<div class="card"><h4>📈 DISTRIBUSI LEVEL PENGAWASAN</h4></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            level_counts = df['Level_Pengawasan'].value_counts()
            fig = px.pie(values=level_counts.values, names=level_counts.index,
                        color=level_counts.index,
                        color_discrete_map={'AMAN':'#10B981', 'PERLU TEGURAN':'#F59E0B', 'BUTUH PENGAWASAN':'#EF4444'},
                        hole=0.4)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 KRITERIA LEVEL")
            st.markdown("""
            <div class="success-box">
                <strong>✅ AMAN:</strong><br>
                ≤ 10 jam/minggu<br>
                ≤ 3 hari/minggu
            </div>
            <div class="warning-box">
                <strong>⚠️ PERLU TEGURAN:</strong><br>
                11-20 jam/minggu<br>
                4-5 hari/minggu
            </div>
            <div class="danger-box">
                <strong>🚨 BUTUH PENGAWASAN:</strong><br>
                > 20 jam/minggu<br>
                > 5 hari/minggu
            </div>
            """, unsafe_allow_html=True)
        
        # Tampilkan dataset
        st.markdown('<div class="card"><h4>📋 PREVIEW DATASET</h4></div>', unsafe_allow_html=True)
        
        # Filter kolom yang relevan untuk ditampilkan
        display_cols = ['Nama', 'Jurusan', 'Total_Penggunaan_AI_jam_per_minggu', 
                       'Frekuensi_Penggunaan_AI_per_minggu', 'Level_Pengawasan', 'IPK']
        st.dataframe(df[display_cols].head(15), use_container_width=True)
        
        # Visualisasi hubungan penggunaan AI dengan IPK
        st.markdown('<div class="card"><h4>📊 HUBUNGAN PENGGUNAAN AI DENGAN IPK</h4></div>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Scatter Plot", "Box Plot"])
        
        with tab1:
            fig = px.scatter(df, x='Total_Penggunaan_AI_jam_per_minggu', y='IPK',
                            color='Level_Pengawasan',
                            color_discrete_map={'AMAN':'#10B981', 'PERLU TEGURAN':'#F59E0B', 'BUTUH PENGAWASAN':'#EF4444'},
                            hover_data=['Nama', 'Jurusan'],
                            title='Hubungan Total Penggunaan AI dengan IPK',
                            labels={'Total_Penggunaan_AI_jam_per_minggu': 'Total Jam Penggunaan AI/Minggu', 'IPK': 'IPK'})
            fig.add_hline(y=3.0, line_dash="dash", line_color="red", annotation_text="Batas IPK Baik")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = px.box(df, x='Level_Pengawasan', y='IPK',
                        color='Level_Pengawasan',
                        color_discrete_map={'AMAN':'#10B981', 'PERLU TEGURAN':'#F59E0B', 'BUTUH PENGAWASAN':'#EF4444'},
                        title='Distribusi IPK per Level Pengawasan')
            st.plotly_chart(fig, use_container_width=True)
    
    elif menu_option == "🧹 PREPROCESSING DATA":
        st.markdown('<div class="card"><h3 class="sub-header">🧹 PREPROCESSING DATA</h3></div>', unsafe_allow_html=True)
        
        # Data Cleaning
        st.markdown('<div class="card"><h4>🗑️ DATA CLEANING</h4></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Periksa Missing Values", use_container_width=True):
                missing_values = df.isnull().sum()
                missing_df = pd.DataFrame({
                    'Kolom': missing_values.index,
                    'Jumlah Missing': missing_values.values,
                    'Persentase': (missing_values.values / len(df)) * 100
                })
                missing_df = missing_df[missing_df['Jumlah Missing'] > 0]
                
                if len(missing_df) > 0:
                    st.dataframe(missing_df, use_container_width=True)
                    if st.button("🔄 Isi Missing Values dengan Median/Modus"):
                        for col in missing_df['Kolom']:
                            if df[col].dtype in ['int64', 'float64']:
                                df[col].fillna(df[col].median(), inplace=True)
                            else:
                                df[col].fillna(df[col].mode()[0], inplace=True)
                        st.session_state.dataset = df
                        st.success("✅ Missing values telah diisi!")
                else:
                    st.success("✅ Tidak ada missing values dalam dataset!")
        
        with col2:
            if st.button("🔎 Periksa Duplikat", use_container_width=True):
                duplicates = df.duplicated().sum()
                if duplicates > 0:
                    st.warning(f"⚠️ Ditemukan {duplicates} data duplikat")
                    if st.button("🗑️ Hapus Duplikat", use_container_width=True):
                        df.drop_duplicates(inplace=True)
                        st.session_state.dataset = df
                        st.success(f"✅ {duplicates} data duplikat telah dihapus")
                else:
                    st.success("✅ Tidak ada data duplikat")
        
        # Encoding Data Kategorikal
        st.markdown('<div class="card"><h4>🔢 ENCODING DATA KATEGORIKAL</h4></div>', unsafe_allow_html=True)
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col not in ['Nama', 'NIM', 'Level_Pengawasan']]
        
        st.write(f"**Kolom kategorikal yang akan di-encode:**")
        for col in categorical_cols:
            st.write(f"- {col}: {df[col].nunique()} unique values")
        
        if st.button("🚀 Lakukan Label Encoding", use_container_width=True):
            # Simpan data asli untuk referensi
            if 'original_categorical' not in st.session_state:
                st.session_state.original_categorical = {}
            
            # Lakukan Label Encoding
            label_encoders = {}
            df_encoded = df.copy()
            
            for col in categorical_cols:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                label_encoders[col] = le
                st.session_state.original_categorical[col] = df[col].copy()
            
            st.session_state.label_encoders = label_encoders
            st.session_state.dataset_encoded = df_encoded
            st.session_state.data_processed = True
            st.success("✅ Label encoding berhasil dilakukan!")
            
            # Tampilkan contoh hasil encoding
            st.write("**Contoh data setelah encoding:**")
            st.dataframe(df_encoded.iloc[:, :10].head(), use_container_width=True)
        
        # Split Data
        if st.session_state.data_processed:
            st.markdown('<div class="card"><h4>✂️ SPLIT DATA (TRAINING & TESTING)</h4></div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                test_size = st.slider("Persentase Data Testing (%)", 10, 40, 20, help="Persentase data yang akan digunakan untuk testing")
            
            with col2:
                target_col = 'Level_Pengawasan'
                st.info(f"Target Variable: {target_col}")
            
            with col3:
                random_state = st.number_input("Random State", min_value=0, max_value=100, value=42)
            
            if st.button("🎯 Split Data", use_container_width=True):
                # Pisahkan fitur dan target
                X = st.session_state.dataset_encoded.drop(columns=['Nama', 'NIM', 'Level_Pengawasan'], errors='ignore')
                y = st.session_state.dataset_encoded[target_col]
                
                # Encoding target variable
                le_target = LabelEncoder()
                y_encoded = le_target.fit_transform(y)
                st.session_state.le_target = le_target
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=test_size/100, random_state=random_state, stratify=y_encoded
                )
                
                # Simpan di session state
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.target_col = target_col
                
                st.success(f"✅ Data berhasil di-split!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Training Data", f"{len(X_train)} sampel ({100-test_size}%)")
                with col2:
                    st.metric("Testing Data", f"{len(X_test)} sampel ({test_size}%)")
                
                # Tampilkan distribusi kelas
                from collections import Counter
                train_dist = Counter(le_target.inverse_transform(y_train))
                test_dist = Counter(le_target.inverse_transform(y_test))
                
                train_df = pd.DataFrame({'Level': list(train_dist.keys()), 'Jumlah': list(train_dist.values()), 'Set': 'Training'})
                test_df = pd.DataFrame({'Level': list(test_dist.keys()), 'Jumlah': list(test_dist.values()), 'Set': 'Testing'})
                dist_df = pd.concat([train_df, test_df])
                
                fig = px.bar(dist_df, x='Level', y='Jumlah', color='Set',
                            barmode='group', color_discrete_map={'Training': '#3B82F6', 'Testing': '#10B981'},
                            title='Distribusi Kelas di Training dan Testing Data')
                st.plotly_chart(fig, use_container_width=True)
    
    elif menu_option == "🤖 MODELING & EVALUASI":
        st.markdown('<div class="card"><h3 class="sub-header">🤖 MODELING DENGAN RANDOM FOREST</h3></div>', unsafe_allow_html=True)
        
        # Pastikan data sudah di-split
        if 'X_train' not in st.session_state:
            st.warning("⚠️ Silakan lakukan split data terlebih dahulu di menu Preprocessing Data")
            return
        
        # Parameter Random Forest
        st.markdown('<div class="card"><h4>⚙️ PARAMETER RANDOM FOREST</h4></div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators = st.slider("Jumlah Estimator", 50, 300, 100, 50,
                                    help="Jumlah pohon dalam random forest")
        with col2:
            max_depth = st.slider("Max Depth", 5, 50, 20, 5,
                                 help="Kedalaman maksimal setiap pohon")
        with col3:
            min_samples_split = st.slider("Min Samples Split", 2, 20, 5, 1,
                                         help="Jumlah sampel minimal untuk split")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 2, 1,
                                        help="Jumlah sampel minimal di leaf")
        with col2:
            max_features = st.selectbox("Max Features", ['sqrt', 'log2', None])
        with col3:
            random_state = st.number_input("Random State Model", 0, 100, 42)
        
        if st.button("🚀 Latih Model Random Forest", use_container_width=True):
            with st.spinner("Melatih model Random Forest..."):
                # Inisialisasi model
                rf_model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=random_state,
                    n_jobs=-1
                )
                
                # Latih model
                rf_model.fit(st.session_state.X_train, st.session_state.y_train)
                
                # Prediksi
                y_pred = rf_model.predict(st.session_state.X_test)
                y_pred_proba = rf_model.predict_proba(st.session_state.X_test)
                
                # Simpan model dan hasil
                st.session_state.rf_model = rf_model
                st.session_state.y_pred = y_pred
                st.session_state.y_pred_proba = y_pred_proba
                
                # Hitung metrik evaluasi
                accuracy = accuracy_score(st.session_state.y_test, y_pred)
                st.session_state.accuracy = accuracy
                
                # Classification report
                report = classification_report(st.session_state.y_test, y_pred, 
                                              target_names=st.session_state.le_target.classes_, 
                                              output_dict=True)
                st.session_state.classification_report = report
                
                # Confusion matrix
                cm = confusion_matrix(st.session_state.y_test, y_pred)
                st.session_state.confusion_matrix = cm
                
                st.success(f"✅ Model berhasil dilatih dengan akurasi: {accuracy:.2%}")
                st.session_state.model_trained = True
                
                # Feature Importance
                feature_importance = pd.DataFrame({
                    'Feature': st.session_state.X_train.columns,
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                st.session_state.feature_importance = feature_importance
        
        # Tampilkan hasil jika model sudah dilatih
        if 'rf_model' in st.session_state:
            # Tampilkan Feature Importance
            st.markdown('<div class="card"><h4>📊 FEATURE IMPORTANCE</h4></div>', unsafe_allow_html=True)
            
            fig = px.bar(st.session_state.feature_importance.head(15), 
                        x='Importance', y='Feature', 
                        orientation='h',
                        color='Importance',
                        color_continuous_scale='Viridis',
                        title='15 Fitur Terpenting dalam Prediksi Level Pengawasan')
            st.plotly_chart(fig, use_container_width=True)
            
            # Evaluasi Model
            st.markdown('<div class="card"><h4>📈 EVALUASI MODEL</h4></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Confusion Matrix
                st.write("**Confusion Matrix:**")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(st.session_state.confusion_matrix, 
                           annot=True, fmt='d', 
                           cmap='Reds',
                           xticklabels=st.session_state.le_target.classes_,
                           yticklabels=st.session_state.le_target.classes_,
                           ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)
            
            with col2:
                # Metrics
                st.metric("Accuracy", f"{st.session_state.accuracy:.2%}")
                
                # Classification Report
                st.write("**Classification Report:**")
                report_df = pd.DataFrame(st.session_state.classification_report).transpose()
                st.dataframe(report_df[['precision', 'recall', 'f1-score', 'support']].style.background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']), 
                           use_container_width=True)
    
    elif menu_option == "📈 REKOMENDASI PENGAWASAN":
        st.markdown('<div class="card"><h3 class="sub-header">📈 REKOMENDASI STRATEGI PENGAWASAN</h3></div>', unsafe_allow_html=True)
        
        if 'rf_model' not in st.session_state:
            st.warning("⚠️ Silakan latih model terlebih dahulu di menu Modeling & Evaluation")
            return
        
        # Analisis tingkat pengawasan
        st.markdown('<div class="card"><h4>🎯 DISTRIBUSI LEVEL PENGAWASAN BERDASARKAN PREDIKSI</h4></div>', unsafe_allow_html=True)
        
        # Decode prediksi
        y_pred_decoded = st.session_state.le_target.inverse_transform(st.session_state.y_pred)
        y_test_decoded = st.session_state.le_target.inverse_transform(st.session_state.y_test)
        
        # Hitung distribusi
        pred_counts = pd.Series(y_pred_decoded).value_counts()
        actual_counts = pd.Series(y_test_decoded).value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(x=pred_counts.index, y=pred_counts.values,
                        color=pred_counts.index,
                        color_discrete_map={'AMAN':'#10B981', 'PERLU TEGURAN':'#F59E0B', 'BUTUH PENGAWASAN':'#EF4444'},
                        title='Distribusi Prediksi Level Pengawasan',
                        labels={'x': 'Level Pengawasan', 'y': 'Jumlah'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(x=actual_counts.index, y=actual_counts.values,
                        color=actual_counts.index,
                        color_discrete_map={'AMAN':'#10B981', 'PERLU TEGURAN':'#F59E0B', 'BUTUH PENGAWASAN':'#EF4444'},
                        title='Distribusi Aktual Level Pengawasan',
                        labels={'x': 'Level Pengawasan', 'y': 'Jumlah'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Rekomendasi strategi pengawasan
        st.markdown('<div class="card"><h4>💡 REKOMENDASI STRATEGI PENGAWASAN PER LEVEL</h4></div>', unsafe_allow_html=True)
        
        # Tampilkan rekomendasi dalam bentuk card
        recommendations = {
            'AMAN': {
                'title': '✅ LEVEL AMAN',
                'description': 'Penggunaan AI dalam batas wajar',
                'color': '#10B981',
                'actions': [
                    'Berikan apresiasi atas penggunaan AI yang bertanggung jawab',
                    'Dorong untuk berbagi pengalaman positif dengan teman sejawat',
                    'Rekomendasikan tool AI yang lebih advanced untuk pengembangan skill',
                    'Pertahankan frekuensi dan durasi penggunaan AI saat ini'
                ],
                'monitoring': 'Pemantauan bulanan cukup'
            },
            'PERLU TEGURAN': {
                'title': '⚠️ LEVEL PERLU TEGURAN',
                'description': 'Penggunaan AI mulai berlebihan',
                'color': '#F59E0B',
                'actions': [
                    'Berikan teguran lisan secara personal',
                    'Atur batasan waktu penggunaan AI',
                    'Berikan tugas yang mengurangi ketergantungan pada AI',
                    'Lakukan konseling tentang penggunaan AI yang sehat',
                    'Pantau perkembangan penggunaan AI mingguan'
                ],
                'monitoring': 'Pemantauan mingguan diperlukan'
            },
            'BUTUH PENGAWASAN': {
                'title': '🚨 LEVEL BUTUH PENGAWASAN',
                'description': 'Penggunaan AI berlebihan, butuh intervensi',
                'color': '#EF4444',
                'actions': [
                    'Lakukan intervensi langsung oleh dosen wali',
                    'Buat kontrak belajar dengan batasan ketat penggunaan AI',
                    'Wajibkan konseling dengan psikolog kampus',
                    'Berikan sanksi akademik jika tidak ada perbaikan',
                    'Libatkan orang tua dalam pemantauan'
                ],
                'monitoring': 'Pemantauan harian dan intervensi intensif'
            }
        }
        
        # Tampilkan rekomendasi dalam tabs
        tab1, tab2, tab3 = st.tabs(["🚨 BUTUH PENGAWASAN", "⚠️ PERLU TEGURAN", "✅ AMAN"])
        
        with tab1:
            rec = recommendations['BUTUH PENGAWASAN']
            st.markdown(f"<h3 style='color:{rec['color']}'>{rec['title']}</h3>", unsafe_allow_html=True)
            st.markdown(f"<p>{rec['description']}</p>", unsafe_allow_html=True)
            st.markdown("**Tindakan yang Direkomendasikan:**")
            for i, action in enumerate(rec['actions'], 1):
                st.markdown(f"{i}. {action}")
            st.markdown(f"**Frekuensi Pemantauan:** {rec['monitoring']}")
            st.markdown(f"**Jumlah Mahasiswa:** {pred_counts.get('BUTUH PENGAWASAN', 0)} orang")
        
        with tab2:
            rec = recommendations['PERLU TEGURAN']
            st.markdown(f"<h3 style='color:{rec['color']}'>{rec['title']}</h3>", unsafe_allow_html=True)
            st.markdown(f"<p>{rec['description']}</p>", unsafe_allow_html=True)
            st.markdown("**Tindakan yang Direkomendasikan:**")
            for i, action in enumerate(rec['actions'], 1):
                st.markdown(f"{i}. {action}")
            st.markdown(f"**Frekuensi Pemantauan:** {rec['monitoring']}")
            st.markdown(f"**Jumlah Mahasiswa:** {pred_counts.get('PERLU TEGURAN', 0)} orang")
        
        with tab3:
            rec = recommendations['AMAN']
            st.markdown(f"<h3 style='color:{rec['color']}'>{rec['title']}</h3>", unsafe_allow_html=True)
            st.markdown(f"<p>{rec['description']}</p>", unsafe_allow_html=True)
            st.markdown("**Tindakan yang Direkomendasikan:**")
            for i, action in enumerate(rec['actions'], 1):
                st.markdown(f"{i}. {action}")
            st.markdown(f"**Frekuensi Pemantauan:** {rec['monitoring']}")
            st.markdown(f"**Jumlah Mahasiswa:** {pred_counts.get('AMAN', 0)} orang")
        
        # Simulasi prediksi untuk mahasiswa baru
        st.markdown('<div class="card"><h4>🔮 SIMULASI PREDIKSI UNTUK MAHASISWA BARU</h4></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            frekuensi = st.slider("Frekuensi Penggunaan AI (hari/minggu)", 1, 7, 3)
            durasi = st.slider("Durasi Penggunaan AI (jam/hari)", 0.5, 10.0, 2.0, 0.5)
        
        with col2:
            kemahiran = st.selectbox("Tingkat Kemahiran AI", ['Pemula', 'Menengah', 'Mahir', 'Expert'])
            jurusan = st.selectbox("Jurusan", ['AI & Machine Learning', 'Data Science', 'Computer Science', 'IT'])
        
        if st.button("🎯 Prediksi Level Pengawasan"):
            # Hitung total jam per minggu
            total_jam = frekuensi * durasi
            
            # Tentukan level berdasarkan aturan bisnis
            if total_jam <= 10 and frekuensi <= 3:
                prediksi = 'AMAN'
                confidence = 0.85
            elif total_jam <= 20 and frekuensi <= 5:
                prediksi = 'PERLU TEGURAN'
                confidence = 0.75
            else:
                prediksi = 'BUTUH PENGAWASAN'
                confidence = 0.90
            
            # Tampilkan hasil
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Jam/Minggu", f"{total_jam} jam")
                st.metric("Frekuensi/Minggu", f"{frekuensi} hari")
            
            with col2:
                display_supervision_level(prediksi)
                st.metric("Tingkat Kepercayaan", f"{confidence:.0%}")
            
            # Tampilkan rekomendasi
            st.markdown("**📋 Rekomendasi:**")
            for i, action in enumerate(recommendations[prediksi]['actions'], 1):
                st.markdown(f"{i}. {action}")

# Fungsi untuk dashboard siswa
def student_dashboard():
    st.markdown('<h1 class="main-header">👨‍🎓 DASHBOARD SISWA - MONITORING PENGGUNAAN AI</h1>', unsafe_allow_html=True)
    
    df = st.session_state.dataset
    
    # Sidebar untuk pencarian
    st.sidebar.markdown('<h3>🔍 PENCARIAN MAHASISWA</h3>', unsafe_allow_html=True)
    
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
    st.markdown(f'<div class="card"><h3 class="sub-header">📋 PROFIL MAHASISWA</h3></div>', unsafe_allow_html=True)
    
    # Info dasar
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**👤 Nama**")
        st.markdown(f"### {student_data['Nama']}")
    
    with col2:
        st.markdown("**🎓 NIM**")
        st.markdown(f"### {student_data['NIM']}")
    
    with col3:
        st.markdown("**🏫 Jurusan**")
        st.markdown(f"### {student_data['Jurusan']}")
    
    with col4:
        st.markdown("**📚 Semester**")
        st.markdown(f"### {student_data['Semester']}")
    
    # Tampilkan level pengawasan
    st.markdown("---")
    st.markdown(f'<div class="card"><h3 class="sub-header">📊 STATUS PENGGUNAAN AI</h3></div>', unsafe_allow_html=True)
    
    # Tampilkan level dengan card khusus
    display_supervision_level(student_data['Level_Pengawasan'])
    
    # Metrik penggunaan AI
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Jam/Minggu", f"{student_data['Total_Penggunaan_AI_jam_per_minggu']:.1f} jam")
    
    with col2:
        st.metric("Frekuensi/Minggu", f"{student_data['Frekuensi_Penggunaan_AI_per_minggu']} hari")
    
    with col3:
        st.metric("IPK", student_data['IPK'])
    
    with col4:
        st.metric("Tingkat Kemahiran", student_data['Tingkat_Kemahiran_AI'])
    
    # Visualisasi penggunaan AI
    st.markdown('<div class="card"><h4>📈 VISUALISASI PENGGUNAAN AI</h4></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gauge chart untuk total jam penggunaan
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = student_data['Total_Penggunaan_AI_jam_per_minggu'],
            title = {'text': "Total Jam AI/Minggu"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [0, 40]},
                'bar': {'color': "#3B82F6"},
                'steps': [
                    {'range': [0, 10], 'color': "#10B981"},
                    {'range': [10, 20], 'color': "#F59E0B"},
                    {'range': [20, 40], 'color': "#EF4444"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': student_data['Total_Penggunaan_AI_jam_per_minggu']
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Radar chart untuk profil penggunaan
        categories = ['Frekuensi', 'Durasi/Hari', 'Kemahiran', 'Tujuan', 'Dampak']
        
        # Normalisasi nilai (skala 1-5)
        frekuensi_norm = min(student_data['Frekuensi_Penggunaan_AI_per_minggu'] / 7 * 5, 5)
        
        # Mapping durasi ke skala 1-5
        durasi = student_data['Durasi_Penggunaan_AI_jam_per_hari']
        durasi_norm = min(durasi / 2, 5)  # Asumsi 10 jam maksimal = skala 5
        
        kemahiran_map = {'Pemula': 1, 'Menengah': 2, 'Mahir': 4, 'Expert': 5}
        kemahiran_norm = kemahiran_map.get(student_data['Tingkat_Kemahiran_AI'], 2.5)
        
        tujuan_norm = 3  # Default
        
        dampak_map = {'Menurun': 1, 'Tidak Berubah': 3, 'Meningkat': 5}
        dampak_norm = dampak_map.get(student_data['Dampak_Pada_Pemahaman'], 3)
        
        values = [frekuensi_norm, durasi_norm, kemahiran_norm, tujuan_norm, dampak_norm]
        
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
                    range=[0, 5]
                )),
            showlegend=False,
            height=300,
            title="Profil Penggunaan AI"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Rekomendasi personal berdasarkan level
    st.markdown('<div class="card"><h4>💡 REKOMENDASI PERSONAL</h4></div>', unsafe_allow_html=True)
    
    if student_data['Level_Pengawasan'] == 'AMAN':
        st.markdown("""
        <div class="success-box">
            <h4>🎉 SELAMAT! Penggunaan AI Anda Masuk Kategori AMAN</h4>
            <p><strong>Rekomendasi untuk Anda:</strong></p>
            <ol>
                <li>Pertahankan pola penggunaan AI yang sehat ini</li>
                <li>Gunakan AI untuk mengembangkan skill, bukan menggantikan proses belajar</li>
                <li>Bagikan pengalaman positif Anda dengan teman-teman</li>
                <li>Eksplorasi tool AI yang lebih advance untuk meningkatkan produktivitas</li>
                <li>Tetap utamakan pemahaman konsep secara mandiri</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    elif student_data['Level_Pengawasan'] == 'PERLU TEGURAN':
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ PERHATIAN! Penggunaan AI Anda Mulai Berlebihan</h4>
            <p><strong>Rekomendasi untuk Anda:</strong></p>
            <ol>
                <li>Kurangi durasi penggunaan AI menjadi maksimal 2 jam/hari</li>
                <li>Batasi penggunaan AI hanya untuk tugas-tugas tertentu</li>
                <li>Latih kemampuan pemecahan masalah secara mandiri</li>
                <li>Konsultasikan dengan dosen tentang penggunaan AI yang tepat</li>
                <li>Ikuti workshop "Penggunaan AI yang Bertanggung Jawab"</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    else:  # BUTUH PENGAWASAN
        st.markdown("""
        <div class="danger-box">
            <h4>🚨 PERINGATAN! Penggunaan AI Anda BERLEBIHAN</h4>
            <p><strong>Rekomendasi untuk Anda:</strong></p>
            <ol>
                <li>Segera kurangi penggunaan AI minimal 50%</li>
                <li>Wajib konsultasi dengan dosen wali dalam 3 hari</li>
                <li>Ikuti program "Detoksifikasi AI" dari kampus</li>
                <li>Buat jadwal belajar tanpa AI</li>
                <li>Laporkan perkembangan penggunaan AI mingguan ke dosen</li>
                <li>Pertimbangkan konseling dengan psikolog kampus</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # Perbandingan dengan rata-rata
    st.markdown('<div class="card"><h4>📊 PERBANDINGAN DENGAN RATA-RATA</h4></div>', unsafe_allow_html=True)
    
    jurusan_avg = df[df['Jurusan'] == student_data['Jurusan']].mean(numeric_only=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        diff_ipk = student_data['IPK'] - jurusan_avg['IPK']
        st.metric("IPK Anda", f"{student_data['IPK']:.2f}", 
                 f"{diff_ipk:+.2f}" if diff_ipk != 0 else "0.00")
    
    with col2:
        diff_hours = student_data['Total_Penggunaan_AI_jam_per_minggu'] - jurusan_avg['Total_Penggunaan_AI_jam_per_minggu']
        st.metric("Jam AI/Minggu", f"{student_data['Total_Penggunaan_AI_jam_per_minggu']:.1f}", 
                 f"{diff_hours:+.1f}" if diff_hours != 0 else "0.0")
    
    with col3:
        ranking = sum(df['IPK'] <= student_data['IPK']) / len(df) * 100
        st.metric("Peringkat", f"Top {ranking:.0f}%")
    
    # Tampilkan progress bar untuk batasan penggunaan AI
    st.markdown("---")
    st.markdown("**🎯 BATASAN PENGGUNAAN AI YANG DIREKOMENDASIKAN:**")
    
    # Progress bar untuk jam per minggu
    max_recommended_hours = 20  # Batas maksimal sehat
    current_hours = student_data['Total_Penggunaan_AI_jam_per_minggu']
    progress_percent = min(current_hours / max_recommended_hours * 100, 100)
    
    st.progress(progress_percent / 100)
    st.caption(f"{current_hours:.1f} jam dari {max_recommended_hours} jam maksimal yang direkomendasikan per minggu")
    
    if progress_percent > 100:
        st.error(f"⚠️ Anda melebihi batas rekomendasi sebesar {current_hours - max_recommended_hours:.1f} jam!")
    elif progress_percent > 70:
        st.warning("⚠️ Anda mendekati batas rekomendasi penggunaan AI")
    else:
        st.success("✅ Penggunaan AI Anda masih dalam batas wajar")

# Main app
def main():
    # Sidebar untuk pilih dashboard
    st.sidebar.markdown("""
    <h2 style="background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               margin-bottom: 20px;">
    🤖 SISTEM PEMANTAUAN AI
    </h2>
    """, unsafe_allow_html=True)
    
    user_type = st.sidebar.radio(
        "PILIH DASHBOARD:",
        ["👨‍🏫 DASHBOARD GURU", "👨‍🎓 DASHBOARD SISWA"]
    )
    
    # Tambahkan info tentang sistem
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 SISTEM LEVEL PENGAWASAN")
    st.sidebar.info("""
    **✅ AMAN:** ≤10 jam/minggu, ≤3 hari/minggu  
    **⚠️ TEGURAN:** 11-20 jam/minggu, 4-5 hari/minggu  
    **🚨 PENGAWASAN:** >20 jam/minggu, >5 hari/minggu
    """)
    
    st.sidebar.markdown("### 📊 INFORMASI DATASET")
    st.sidebar.info("""
    Dataset berisi 250 mahasiswa dengan:
    - Data penggunaan AI (frekuensi & durasi)
    - Level pengawasan otomatis
    - Performa akademik (IPK)
    - Faktor pendukung lainnya
    """)
    
    # Pilih dashboard berdasarkan user type
    if "DASHBOARD GURU" in user_type:
        teacher_dashboard()
    else:
        student_dashboard()

if __name__ == "__main__":
    main()
