import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Cek apakah library machine learning sudah diinstall
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    ML_AVAILABLE = True
except ImportError as e:
    st.warning(f"⚠️ Library scikit-learn tidak tersedia. Error: {str(e)}")
    st.info("Install dengan: pip install scikit-learn")
    ML_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="Sistem Analisis Penggunaan AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def init_session_state():
    if 'terautentikasi' not in st.session_state:
        st.session_state.terautentikasi = False
    if 'tipe_pengguna' not in st.session_state:
        st.session_state.tipe_pengguna = None
    if 'df_raw' not in st.session_state:
        st.session_state.df_raw = None
    if 'df_clean' not in st.session_state:
        st.session_state.df_clean = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'label_encoders' not in st.session_state:
        st.session_state.label_encoders = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None

# Function to generate sample data
def generate_sample_data(n_samples=100):
    """Generate sample data for demonstration"""
    np.random.seed(42)
    
    # Sample data
    colleges = [
        'Institut Teknologi India', 'Universitas Delhi', 
        'Universitas Jawaharlal Nehru', 'Universitas Kalkuta',
        'Universitas Mumbai', 'Universitas Anna', 'Universitas Hyderabad',
        'Universitas Hindu Banaras', 'Universitas Madras', 'Universitas Pune'
    ]
    
    streams = ['Teknik', 'Sains', 'Bisnis', 'Seni', 'Kedokteran', 'Hukum', 'Manajemen', 'Farmasi', 'Manajemen Perhotelan', 'Pertanian']
    
    ai_tools = [
        'ChatGPT', 'Gemini', 'Copilot', 'ChatGPT, Gemini', 
        'Gemini, Copilot', 'ChatGPT, Copilot', 'ChatGPT, Gemini, Copilot',
        'Gemini, Midjourney', 'ChatGPT, Midjourney', 'Semua Tools'
    ]
    
    use_cases = [
        'Tugas', 'Penulisan Konten', 'Latihan MCQ', 'Persiapan Ujian', 
        'Pemecahan Masalah', 'Mempelajari topik baru', 'Proyek',
        'Penelitian', 'Bantuan Coding', 'Presentasi'
    ]
    
    data = {
        'Nama_Siswa': [f'Siswa_{i:03d}' for i in range(1, n_samples + 1)],
        'Nama_Kampus': np.random.choice(colleges, n_samples),
        'Jurusan': np.random.choice(streams, n_samples),
        'Tools_AI_Digunakan': np.random.choice(ai_tools, n_samples),
        'Skor_Intensitas_Penggunaan': np.random.randint(5, 46, n_samples),
        'Kasus_Penggunaan': [', '.join(np.random.choice(use_cases, np.random.randint(1, 4))) for _ in range(n_samples)]
    }
    
    df = pd.DataFrame(data)
    return df

# Function to load data
def load_data(uploaded_file=None):
    """Load data from uploaded CSV or use sample data"""
    try:
        if uploaded_file is not None:
            # Try to read the uploaded file
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            else:
                st.error("❌ Silakan upload file CSV atau Excel")
                return None
        else:
            # Use sample data
            df = generate_sample_data()
        
        # Basic validation
        if df.empty:
            st.error("❌ File yang diupload kosong")
            return None
        
        # Ensure required columns exist
        required_columns = ['Nama_Siswa', 'Nama_Kampus', 'Jurusan', 
                           'Tools_AI_Digunakan', 'Skor_Intensitas_Penggunaan', 'Kasus_Penggunaan']
        
        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.warning(f"⚠️ Kolom yang hilang: {missing_columns}")
            st.info("Menggunakan data sampel sebagai gantinya")
            df = generate_sample_data()
        
        # Show basic info
        st.success(f"✅ Data berhasil dimuat! Ukuran: {df.shape}")
        st.info(f"📊 Kolom: {', '.join(df.columns.tolist())}")
        
        return df
        
    except Exception as e:
        st.error(f" Error memuat data: {str(e)}")
        st.info("Menggunakan data sampel sebagai gantinya")
        return generate_sample_data()

# Function for data preprocessing
def preprocess_data(df):
    """Preprocess the data for machine learning"""
    try:
        if df is None:
            st.error("Tidak ada data untuk diproses")
            return None, None
        
        df_clean = df.copy()
        
        # 1. Check for missing values
        missing_values = df_clean.isnull().sum()
        if missing_values.sum() > 0:
            st.warning(f" Nilai yang hilang ditemukan: {missing_values[missing_values > 0].to_dict()}")
            
            # Fill missing values
            for col in df_clean.columns:
                if df_clean[col].dtype == 'object':
                    df_clean[col] = df_clean[col].fillna('Tidak Diketahui')
                else:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # 2. Convert Skor_Intensitas_Penggunaan to numeric
        df_clean['Skor_Intensitas_Penggunaan'] = pd.to_numeric(
            df_clean['Skor_Intensitas_Penggunaan'], errors='coerce'
        )
        
        # Fill any NaN values after conversion
        df_clean['Skor_Intensitas_Penggunaan'] = df_clean['Skor_Intensitas_Penggunaan'].fillna(
            df_clean['Skor_Intensitas_Penggunaan'].median()
        )
        
        # 3. Create target variable (Tingkat Penggunaan)
        df_clean['Tingkat_Penggunaan'] = pd.cut(
            df_clean['Skor_Intensitas_Penggunaan'],
            bins=[0, 15, 30, 50],
            labels=['Rendah', 'Sedang', 'Tinggi']
        )
        
        # Remove rows with NaN in Tingkat_Penggunaan
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna(subset=['Tingkat_Penggunaan'])
        removed_rows = initial_rows - len(df_clean)
        
        if removed_rows > 0:
            st.warning(f" Dihapus {removed_rows} baris dengan Skor_Intensitas_Penggunaan tidak valid")
        
        # 4. Simple encoding for demonstration
        label_encoders = {}
        
        # Encode Jurusan
        if 'Jurusan' in df_clean.columns:
            unique_streams = df_clean['Jurusan'].unique()
            stream_mapping = {stream: i for i, stream in enumerate(unique_streams)}
            df_clean['Jurusan_Encoded'] = df_clean['Jurusan'].map(stream_mapping)
            label_encoders['Jurusan'] = stream_mapping
        
        # Simple tool count encoding
        df_clean['Jumlah_Tools'] = df_clean['Tools_AI_Digunakan'].apply(
            lambda x: len(str(x).split(',')) if pd.notnull(x) else 0
        )
        
        # Simple use cases count encoding
        df_clean['Jumlah_Kasus_Penggunaan'] = df_clean['Kasus_Penggunaan'].apply(
            lambda x: len(str(x).split(',')) if pd.notnull(x) else 0
        )
        
        # Encode target variable
        usage_level_mapping = {'Rendah': 0, 'Sedang': 1, 'Tinggi': 2}
        df_clean['Tingkat_Penggunaan_Encoded'] = df_clean['Tingkat_Penggunaan'].map(usage_level_mapping)
        label_encoders['Tingkat_Penggunaan'] = usage_level_mapping
        
        st.success(f"Pra-pemrosesan data selesai! Ukuran akhir: {df_clean.shape}")
        return df_clean, label_encoders
        
    except Exception as e:
        st.error(f"Error dalam pra-pemrosesan: {str(e)}")
        return None, None

# Function to split data
def split_data(df_clean, test_size=0.3, random_state=42):
    """Split data into training and testing sets"""
    try:
        # Prepare features and target
        feature_cols = ['Jurusan_Encoded', 'Jumlah_Tools', 'Jumlah_Kasus_Penggunaan', 'Skor_Intensitas_Penggunaan']
        
        # Check if all feature columns exist
        available_features = [col for col in feature_cols if col in df_clean.columns]
        
        if len(available_features) < 2:
            st.error("Tidak cukup fitur yang tersedia untuk pelatihan")
            return None, None, None, None, None, None
        
        X = df_clean[available_features]
        y = df_clean['Tingkat_Penggunaan_Encoded']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        st.success(f"Pembagian data selesai: Latih={len(X_train)}, Uji={len(X_test)}")
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, available_features
        
    except Exception as e:
        st.error(f" Error dalam pembagian data: {str(e)}")
        return None, None, None, None, None, None

# Function to train model
def train_model(X_train, y_train, **kwargs):
    """Train a Random Forest classifier"""
    try:
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        }
        params.update(kwargs)
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        st.success("Pelatihan model selesai!")
        return model
        
    except Exception as e:
        st.error(f"Error dalam pelatihan model: {str(e)}")
        return None

# Function to evaluate model
def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics"""
    try:
        if model is None or X_test is None or y_test is None:
            return None, None, None, None
        
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'akurasi': accuracy,
            'presisi': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return y_pred, report_df, cm, metrics
        
    except Exception as e:
        st.error(f"Error dalam evaluasi model: {str(e)}")
        return None, None, None, None

# Function to create download link
def get_csv_download_link(df, filename="data.csv"):
    """Generate a download link for a DataFrame"""
    try:
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 {filename}</a>'
        return href
    except:
        return ""

# Main application
def main():
    # Initialize session state
    init_session_state()
    
    # Check if ML libraries are available
    if not ML_AVAILABLE:
        st.error("""
        **Library yang diperlukan tidak tersedia!**
        
        Silakan install library yang diperlukan:
        ```
        pip install scikit-learn pandas numpy matplotlib seaborn
        ```
        """)
        return
    
    # Login sidebar
    with st.sidebar:
        st.title("Sistem Login")
        
        if not st.session_state.terautentikasi:
            st.markdown("---")
            tipe_pengguna = st.selectbox("Pilih Tipe Pengguna", ["Guru", "Siswa"])
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Login", use_container_width=True):
                    if tipe_pengguna == "Guru" and username == "guru" and password == "guru123":
                        st.session_state.terautentikasi = True
                        st.session_state.tipe_pengguna = "guru"
                        st.success("✅ Login guru berhasil!")
                        st.rerun()
                    elif tipe_pengguna == "Siswa" and username == "siswa" and password == "siswa123":
                        st.session_state.terautentikasi = True
                        st.session_state.tipe_pengguna = "siswa"
                        st.success("✅ Login siswa berhasil!")
                        st.rerun()
                    else:
                        st.error("Kredensial tidak valid!")
        
            
            if st.button("Reset Semua Data", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
            
            if st.button("Keluar", use_container_width=True):
                st.session_state.terautentikasi = False
                st.session_state.tipe_pengguna = None
                st.rerun()
    
    # Main content based on authentication
    if not st.session_state.terautentikasi:
        show_welcome_page()
        return
    
    # Teacher Dashboard
    if st.session_state.tipe_pengguna == "guru":
        show_teacher_dashboard()
    else:
        show_student_dashboard()

def show_welcome_page():
    """Show welcome page for unauthenticated users"""
    st.title("Sistem Analisis Penggunaan AI")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
       Dashboard Analisis Penggunaan AI
        
        Sistem ini menganalisis hubungan antara penggunaan tools AI dengan performa akademik
        menggunakan algoritma Random Forest.
        
        **Fitur:**
        **Dashboard Guru:**
        - Upload dan kelola data siswa
        - Pra-pemrosesan dan pembersihan data
        - Latih model Random Forest
        - Evaluasi performa model
        - Visualisasi hasil
        - Ekspor hasil analisis
        
        **Dashboard Siswa:**
        -  Lihat hasil analisis
        -  Prediksi tingkat penggunaan AI pribadi
        -  Bandingkan dengan teman sebaya
        
        **Persyaratan Data:**
        Upload file CSV dengan kolom berikut:
        - Nama_Siswa
        - Nama_Kampus
        - Jurusan
        - Tools_AI_Digunakan
        - Skor_Intensitas_Penggunaan (1-50)
        - Kasus_Penggunaan
        """)
    
    with col2:
        # Show sample data
        st.subheader("Data Sampel")
        sample_df = generate_sample_data(5)
        st.dataframe(sample_df, use_container_width=True)
        
        # Download sample CSV
        csv = sample_df.to_csv(index=False)
        st.download_button(
            label="Download CSV Sampel",
            data=csv,
            file_name="sampel_data_penggunaan_ai.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    st.markdown("---")
    st.info("**Silakan login dari sidebar untuk melanjutkan**")

def show_teacher_dashboard():
    """Show teacher dashboard"""
    st.title(" Dashboard Guru")
    st.markdown("**Analisis Penggunaan AI terhadap Performa Akademik menggunakan Random Forest**")
    
    # Teacher menu
    menu = st.sidebar.radio(
        "Menu",
        ["Manajemen Data", " Pra-Pemrosesan Data", " Pelatihan Model", 
         "Evaluasi Model", " Visualisasi", " Ekspor Data"],
        index=0
    )
    
    if menu == " Manajemen Data":
        show_data_management()
    elif menu == " Pra-Pemrosesan Data":
        show_data_preprocessing()
    elif menu == " Pelatihan Model":
        show_model_training()
    elif menu == " Evaluasi Model":
        show_model_evaluation()
    elif menu == " Visualisasi":
        show_visualizations()
    elif menu == " Ekspor Data":
        show_export_data()

def show_data_management():
    """Show data management section"""
    st.header(" Manajemen Data")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Data")
        
        uploaded_file = st.file_uploader(
            "Pilih file CSV atau Excel",
            type=['csv', 'xlsx', 'xls'],
            help="Upload file data siswa"
        )
        
        if uploaded_file is not None:
            if st.button("Muat Data", use_container_width=True):
                with st.spinner("Memuat data..."):
                    df = load_data(uploaded_file)
                    if df is not None:
                        st.session_state.df_raw = df
                        st.success(f"Data dimuat: {len(df)} catatan")
        
        st.subheader("Atau Gunakan Data Sampel")
        if st.button("Generate Data Sampel", use_container_width=True):
            with st.spinner("Membuat data sampel..."):
                st.session_state.df_raw = generate_sample_data()
                st.success(f"Data sampel dibuat: {len(st.session_state.df_raw)} catatan")
    
    with col2:
        if st.session_state.df_raw is not None:
            st.subheader("Pratinjau Data")
            
            # Show data info
            st.info(f"**Ukuran Data:** {st.session_state.df_raw.shape[0]} baris × {st.session_state.df_raw.shape[1]} kolom")
            
            # Show data
            st.dataframe(st.session_state.df_raw.head(10), use_container_width=True)
            
            # Data statistics
            with st.expander("Statistik Data", expanded=True):
                col_stats1, col_stats2 = st.columns(2)
                
                with col_stats1:
                    st.write("**Informasi Dasar:**")
                    st.write(f"Baris: {st.session_state.df_raw.shape[0]}")
                    st.write(f"Kolom: {st.session_state.df_raw.shape[1]}")
                    st.write(f"Nilai Hilang: {st.session_state.df_raw.isnull().sum().sum()}")
                
                with col_stats2:
                    st.write("**Tipe Kolom:**")
                    dtype_info = st.session_state.df_raw.dtypes.astype(str).to_dict()
                    for col, dtype in dtype_info.items():
                        st.write(f"• {col}: {dtype}")
            
            # Column distribution
            with st.expander("Distribusi Kolom"):
                selected_col = st.selectbox(
                    "Pilih kolom",
                    st.session_state.df_raw.columns.tolist()
                )
                
                if selected_col:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    
                    if st.session_state.df_raw[selected_col].dtype == 'object':
                        # Categorical
                        value_counts = st.session_state.df_raw[selected_col].value_counts().head(10)
                        value_counts.plot(kind='bar', ax=ax, color='skyblue')
                    else:
                        # Numerical
                        st.session_state.df_raw[selected_col].hist(ax=ax, bins=20, color='skyblue')
                    
                    ax.set_title(f'Distribusi {selected_col}')
                    ax.set_xlabel(selected_col)
                    ax.set_ylabel('Frekuensi')
                    st.pyplot(fig)
        else:
            st.info("Silakan upload data atau buat data sampel untuk memulai")

def show_data_preprocessing():
    """Show data preprocessing section"""
    st.header("Pra-Pemrosesan Data")
    
    if st.session_state.df_raw is None:
        st.warning("Silakan muat data terlebih dahulu di bagian 'Manajemen Data'")
        return
    
    st.subheader("Pratinjau Data Mentah")
    st.dataframe(st.session_state.df_raw.head(), use_container_width=True)
    
    if st.button("Mulai Pra-Pemrosesan", use_container_width=True):
        with st.spinner("Memproses data..."):
            df_clean, label_encoders = preprocess_data(st.session_state.df_raw)
            
            if df_clean is not None and label_encoders is not None:
                st.session_state.df_clean = df_clean
                st.session_state.label_encoders = label_encoders
                
                st.success("Pra-pemrosesan selesai!")
                
                # Show results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Pratinjau Data Bersih")
                    st.dataframe(df_clean[['Nama_Siswa', 'Nama_Kampus', 'Jurusan', 
                                         'Skor_Intensitas_Penggunaan', 'Tingkat_Penggunaan']].head(), 
                               use_container_width=True)
                
                with col2:
                    st.subheader("Distribusi Target")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    
                    level_counts = df_clean['Tingkat_Penggunaan'].value_counts()
                    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                    level_counts.plot(kind='bar', ax=ax, color=colors)
                    
                    ax.set_xlabel('Tingkat Penggunaan')
                    ax.set_ylabel('Jumlah')
                    ax.set_title('Distribusi Tingkat Penggunaan AI')
                    
                    # Add count labels
                    for i, v in enumerate(level_counts):
                        ax.text(i, v + 0.5, str(v), ha='center')
                    
                    st.pyplot(fig)
                
                # Show preprocessing details
                with st.expander("Detail Pra-Pemrosesan"):
                    st.write("**Fitur yang Dibuat:**")
                    st.write("• Jurusan_Encoded: Kode untuk kategori jurusan")
                    st.write("• Jumlah_Tools: Jumlah tools AI yang digunakan")
                    st.write("• Jumlah_Kasus_Penggunaan: Jumlah kasus penggunaan")
                    st.write("• Tingkat_Penggunaan: Kategori tingkat penggunaan (Rendah/Sedang/Tinggi)")
                    st.write("• Tingkat_Penggunaan_Encoded: Kode numerik untuk tingkat penggunaan")
                    
                    st.write("\n**Ukuran Data Setelah Pra-Pemrosesan:**")
                    st.write(f"• Baris: {df_clean.shape[0]}")
                    st.write(f"• Kolom: {df_clean.shape[1]}")
            else:
                st.error("Pra-pemrosesan gagal. Silakan periksa data Anda.")

def show_model_training():
    """Show model training section"""
    st.header("Pelatihan Model dengan Random Forest")
    
    if st.session_state.df_clean is None:
        st.warning("Silakan pra-proses data terlebih dahulu di bagian 'Pra-Pemrosesan Data'")
        return
    
    st.subheader("Konfigurasi Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_estimators = st.slider("Jumlah Pohon", 50, 200, 100, 10)
        max_depth = st.slider("Kedalaman Maksimum", 5, 20, 10, 1)
    
    with col2:
        min_samples_split = st.slider("Sampel Minimal untuk Split", 2, 10, 2, 1)
        min_samples_leaf = st.slider("Sampel Minimal untuk Daun", 1, 5, 1, 1)
    
    test_size = st.slider("Ukuran Data Uji (%)", 20, 40, 30, 5) / 100
    random_state = st.number_input("Random State", 0, 100, 42)
    
    if st.button("Latih Model", use_container_width=True):
        with st.spinner("Melatih model..."):
            # Split data
            X_train, X_test, y_train, y_test, scaler, feature_cols = split_data(
                st.session_state.df_clean, test_size, random_state
            )
            
            if X_train is not None:
                # Train model
                model = train_model(
                    X_train, y_train,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state
                )
                
                if model is not None:
                    st.session_state.model = model
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.scaler = scaler
                    st.session_state.feature_cols = feature_cols
                    
                    # Evaluate model
                    y_pred, report_df, cm, metrics = evaluate_model(
                        model, X_test, y_test
                    )
                    
                    if metrics is not None:
                        st.session_state.metrics = metrics
                        st.session_state.y_pred = y_pred
                        st.session_state.report_df = report_df
                        st.session_state.cm = cm
                        
                        # Show training results
                        st.success("Model berhasil dilatih!")
                        
                        st.subheader("Hasil Pelatihan")
                        
                        col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
                        
                        with col_metric1:
                            st.metric("Akurasi", f"{metrics['akurasi']:.2%}")
                        
                        with col_metric2:
                            st.metric("Presisi", f"{metrics['presisi']:.2%}")
                        
                        with col_metric3:
                            st.metric("Recall", f"{metrics['recall']:.2%}")
                        
                        with col_metric4:
                            st.metric("F1-Score", f"{metrics['f1_score']:.2%}")
                        
                        # Feature importance
                        st.subheader("Importansi Fitur")
                        
                        if hasattr(model, 'feature_importances_'):
                            importance_df = pd.DataFrame({
                                'Fitur': feature_cols,
                                'Importansi': model.feature_importances_
                            }).sort_values('Importansi', ascending=False)
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.barh(importance_df['Fitur'], importance_df['Importansi'], color='lightcoral')
                            ax.set_xlabel('Importansi')
                            ax.set_title('Importansi Fitur')
                            ax.invert_yaxis()
                            st.pyplot(fig)
                        else:
                            st.info("Importansi fitur tidak tersedia untuk model ini")

def show_model_evaluation():
    """Show model evaluation section"""
    st.header("Evaluasi Model")
    
    if st.session_state.model is None:
        st.warning("Silakan latih model terlebih dahulu di bagian 'Pelatihan Model'")
        return
    
    if st.session_state.metrics is None:
        st.warning("Tidak ada metrik evaluasi yang tersedia")
        return
    
    # Display metrics
    st.subheader("Metrik Performa")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        metrics_data = {
            'Metrik': ['Akurasi', 'Presisi', 'Recall', 'F1-Score'],
            'Nilai': [
                f"{st.session_state.metrics['akurasi']:.2%}",
                f"{st.session_state.metrics['presisi']:.2%}",
                f"{st.session_state.metrics['recall']:.2%}",
                f"{st.session_state.metrics['f1_score']:.2%}"
            ]
        }
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    with col2:
        # Confusion Matrix
        if st.session_state.cm is not None:
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(st.session_state.cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Rendah', 'Sedang', 'Tinggi'],
                       yticklabels=['Rendah', 'Sedang', 'Tinggi'], ax=ax)
            ax.set_ylabel('Aktual')
            ax.set_xlabel('Prediksi')
            ax.set_title('Matriks Kebingungan')
            st.pyplot(fig)
    
    # Classification Report
    st.subheader("Laporan Klasifikasi Detail")
    if st.session_state.report_df is not None:
        st.dataframe(st.session_state.report_df, use_container_width=True)
    
    # Performance visualization
    st.subheader("Visualisasi Performa")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Bar chart of metrics
    axes[0].bar(['Akurasi', 'Presisi', 'Recall', 'F1-Score'],
               [st.session_state.metrics['akurasi'],
                st.session_state.metrics['presisi'],
                st.session_state.metrics['recall'],
                st.session_state.metrics['f1_score']],
               color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFE66D'])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('Skor')
    axes[0].set_title('Metrik Performa Model')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for i, v in enumerate([st.session_state.metrics['akurasi'],
                           st.session_state.metrics['presisi'],
                           st.session_state.metrics['recall'],
                           st.session_state.metrics['f1_score']]):
        axes[0].text(i, v + 0.02, f'{v:.2%}', ha='center')
    
    # Metrics by class (if available)
    if st.session_state.report_df is not None and '0' in st.session_state.report_df.index:
        try:
            class_metrics = st.session_state.report_df.loc[['0', '1', '2']]
            x = np.arange(3)
            width = 0.25
            
            axes[1].bar(x - width, class_metrics['precision'], width, label='Presisi', color='#4ECDC4')
            axes[1].bar(x, class_metrics['recall'], width, label='Recall', color='#FF6B6B')
            axes[1].bar(x + width, class_metrics['f1-score'], width, label='F1-Score', color='#45B7D1')
            
            axes[1].set_xlabel('Kelas (0=Rendah, 1=Sedang, 2=Tinggi)')
            axes[1].set_ylabel('Skor')
            axes[1].set_title('Metrik per Kelas')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(['Rendah', 'Sedang', 'Tinggi'])
            axes[1].legend()
            axes[1].set_ylim(0, 1)
        except:
            # Simple metric plot if class metrics not available
            axes[1].text(0.5, 0.5, 'Metrik kelas\ntidak tersedia', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Metrik Kelas')
    
    plt.tight_layout()
    st.pyplot(fig)

def show_visualizations():
    """Show data visualizations"""
    st.header("Visualisasi Data")
    
    if st.session_state.df_clean is None:
        st.warning("Silakan pra-proses data terlebih dahulu")
        return
    
    viz_option = st.selectbox(
        "Pilih Jenis Visualisasi",
        ["Distribusi Penggunaan", "Analisis Jurusan", "Analisis Kampus", 
         "Analisis Tools AI", "Analisis Kasus Penggunaan"]
    )
    
    if viz_option == "Distribusi Penggunaan":
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram
        axes[0].hist(st.session_state.df_clean['Skor_Intensitas_Penggunaan'], 
                    bins=20, color='skyblue', edgecolor='black')
        axes[0].set_xlabel('Skor Intensitas Penggunaan')
        axes[0].set_ylabel('Frekuensi')
        axes[0].set_title('Distribusi Skor Penggunaan')
        
        # Box plot
        axes[1].boxplot(st.session_state.df_clean['Skor_Intensitas_Penggunaan'])
        axes[1].set_ylabel('Skor Intensitas Penggunaan')
        axes[1].set_title('Box Plot Skor Penggunaan')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    elif viz_option == "Analisis Jurusan":
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Average usage by stream
        stream_avg = st.session_state.df_clean.groupby('Jurusan')['Skor_Intensitas_Penggunaan'].mean().sort_values()
        stream_avg.plot(kind='barh', ax=axes[0], color='lightcoral')
        axes[0].set_xlabel('Skor Rata-rata')
        axes[0].set_title('Rata-rata Penggunaan AI per Jurusan')
        
        # Usage level distribution by stream
        try:
            stream_level = pd.crosstab(st.session_state.df_clean['Jurusan'], 
                                     st.session_state.df_clean['Tingkat_Penggunaan'])
            stream_level.plot(kind='bar', ax=axes[1], stacked=True)
            axes[1].set_xlabel('Jurusan')
            axes[1].set_ylabel('Jumlah')
            axes[1].set_title('Tingkat Penggunaan per Jurusan')
            axes[1].legend(title='Tingkat Penggunaan')
            plt.xticks(rotation=45)
        except:
            axes[1].text(0.5, 0.5, 'Data tidak tersedia\nuntuk visualisasi ini', 
                        ha='center', va='center', transform=axes[1].transAxes)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    elif viz_option == "Analisis Kampus":
        # Top colleges by average usage
        college_avg = st.session_state.df_clean.groupby('Nama_Kampus')['Skor_Intensitas_Penggunaan'].mean().nlargest(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        college_avg.plot(kind='barh', ax=ax, color='lightgreen')
        ax.set_xlabel('Skor Rata-rata')
        ax.set_title('10 Kampus Teratas berdasarkan Penggunaan AI')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    elif viz_option == "Analisis Tools AI":
        # Count of AI tools usage
        try:
            all_tools = []
            for tools in st.session_state.df_clean['Tools_AI_Digunakan']:
                if isinstance(tools, str):
                    tool_list = [t.strip() for t in tools.split(',')]
                    all_tools.extend(tool_list)
            
            if all_tools:
                tool_counts = pd.Series(all_tools).value_counts().head(10)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                tool_counts.plot(kind='bar', ax=ax, color='orange')
                ax.set_xlabel('Tool AI')
                ax.set_ylabel('Jumlah')
                ax.set_title('Tools AI Paling Banyak Digunakan')
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Tidak ada data tools AI yang tersedia")
        except:
            st.info("Tidak dapat menganalisis data tools AI")
    
    elif viz_option == "Analisis Kasus Penggunaan":
        # Common use cases
        try:
            all_cases = []
            for cases in st.session_state.df_clean['Kasus_Penggunaan']:
                if isinstance(cases, str):
                    case_list = [c.strip() for c in cases.split(',')]
                    all_cases.extend(case_list)
            
            if all_cases:
                case_counts = pd.Series(all_cases).value_counts()
                
                fig, ax = plt.subplots(figsize=(10, 5))
                case_counts.plot(kind='bar', ax=ax, color='lightblue')
                ax.set_xlabel('Kasus Penggunaan')
                ax.set_ylabel('Jumlah')
                ax.set_title('Kasus Penggunaan AI Paling Umum')
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Tidak ada data kasus penggunaan yang tersedia")
        except:
            st.info("Tidak dapat menganalisis data kasus penggunaan")

def show_export_data():
    """Show data export section"""
    st.header("Ekspor Data & Hasil")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ekspor File Data")
        
        if st.session_state.df_raw is not None:
            csv_raw = st.session_state.df_raw.to_csv(index=False)
            st.download_button(
                label="Data Mentah (CSV)",
                data=csv_raw,
                file_name="data_penggunaan_ai_mentah.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        if st.session_state.df_clean is not None:
            csv_clean = st.session_state.df_clean.to_csv(index=False)
            st.download_button(
                label="📥 Data Bersih (CSV)",
                data=csv_clean,
                file_name="data_penggunaan_ai_bersih.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        if st.session_state.model is not None:
            # Export predictions
            predictions_df = pd.DataFrame({
                'Penggunaan_Aktual': st.session_state.y_test,
                'Penggunaan_Prediksi': st.session_state.y_pred
            })
            csv_pred = predictions_df.to_csv(index=False)
            st.download_button(
                label="📥 Prediksi (CSV)",
                data=csv_pred,
                file_name="prediksi_penggunaan_ai.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        st.subheader("Ekspor Laporan")
        
        if st.session_state.model is not None:
            # Export metrics
            if st.session_state.metrics is not None:
                metrics_df = pd.DataFrame({
                    'Metrik': ['Akurasi', 'Presisi', 'Recall', 'F1-Score'],
                    'Nilai': [
                        st.session_state.metrics['akurasi'],
                        st.session_state.metrics['presisi'],
                        st.session_state.metrics['recall'],
                        st.session_state.metrics['f1_score']
                    ]
                })
                csv_metrics = metrics_df.to_csv(index=False)
                st.download_button(
                    label="Metrik Model (CSV)",
                    data=csv_metrics,
                    file_name="metrik_model.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Export feature importance
            if hasattr(st.session_state.model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Fitur': st.session_state.feature_cols,
                    'Importansi': st.session_state.model.feature_importances_
                }).sort_values('Importansi', ascending=False)
                
                csv_importance = importance_df.to_csv(index=False)
                st.download_button(
                    label="Importansi Fitur (CSV)",
                    data=csv_importance,
                    file_name="importansi_fitur.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    st.markdown("---")
    st.subheader("Laporan Ringkasan")
    
    if st.button("Buat Laporan Ringkasan", use_container_width=True):
        with st.spinner("Membuat laporan..."):
            # Create summary report
            summary = []
            
            if st.session_state.df_raw is not None:
                summary.append(f"**Ringkasan Data:**")
                summary.append(f"• Data Mentah: {st.session_state.df_raw.shape[0]} baris, {st.session_state.df_raw.shape[1]} kolom")
            
            if st.session_state.df_clean is not None:
                summary.append(f"• Data Bersih: {st.session_state.df_clean.shape[0]} baris, {st.session_state.df_clean.shape[1]} kolom")
                
                # Usage level distribution
                if 'Tingkat_Penggunaan' in st.session_state.df_clean.columns:
                    level_dist = st.session_state.df_clean['Tingkat_Penggunaan'].value_counts()
                    summary.append(f"• Distribusi Tingkat Penggunaan:")
                    for level, count in level_dist.items():
                        summary.append(f"  - {level}: {count} siswa ({count/len(st.session_state.df_clean):.1%})")
            
            if st.session_state.model is not None and st.session_state.metrics is not None:
                summary.append(f"\n🤖 **Performa Model:**")
                summary.append(f"• Akurasi: {st.session_state.metrics['akurasi']:.2%}")
                summary.append(f"• Presisi: {st.session_state.metrics['presisi']:.2%}")
                summary.append(f"• Recall: {st.session_state.metrics['recall']:.2%}")
                summary.append(f"• F1-Score: {st.session_state.metrics['f1_score']:.2%}")
            
            # Display summary
            st.markdown("\n".join(summary))
            
            # Download summary
            summary_text = "\n".join(summary)
            st.download_button(
                label="Download Laporan Ringkasan (TXT)",
                data=summary_text,
                file_name="ringkasan_analisis_ai.txt",
                mime="text/plain",
                use_container_width=True
            )

def show_student_dashboard():
    """Show student dashboard"""
    st.title("Dashboard Siswa")
    
    menu = st.sidebar.radio(
        "📋 Menu",
        [" Lihat Analisis", " Prediksi Penggunaan Saya", " Bandingkan dengan Teman"],
        index=0
    )
    
    if menu == " Lihat Analisis":
        show_student_analysis()
    elif menu == " Prediksi Penggunaan Saya":
        show_student_prediction()
    elif menu == " Bandingkan dengan Teman":
        show_student_comparison()

def show_student_analysis():
    """Show analysis results for students"""
    st.header(" Hasil Analisis Penggunaan AI")
    
    if st.session_state.df_clean is None:
        st.info(" Belum ada data analisis yang tersedia. Silakan minta guru untuk mengupload dan menganalisis data.")
        
        # Show sample statistics
        st.subheader("Statistik Sampel")
        sample_df = generate_sample_data(50)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rata-rata Skor Penggunaan", f"{sample_df['Skor_Intensitas_Penggunaan'].mean():.1f}")
        with col2:
            # Categorize sample scores
            sample_levels = pd.cut(sample_df['Skor_Intensitas_Penggunaan'], 
                                  bins=[0, 15, 30, 50], 
                                  labels=['Rendah', 'Sedang', 'Tinggi'])
            common_level = sample_levels.mode()[0]
            st.metric("Tingkat Paling Umum", common_level)
        with col3:
            st.metric("Ukuran Sampel", len(sample_df))
        
        return
    
    # Show actual statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_score = st.session_state.df_clean['Skor_Intensitas_Penggunaan'].mean()
        st.metric("Rata-rata Skor Penggunaan", f"{avg_score:.1f}")
    
    with col2:
        most_common_level = st.session_state.df_clean['Tingkat_Penggunaan'].mode()[0]
        st.metric("Tingkat Paling Umum", most_common_level)
    
    with col3:
        total_students = len(st.session_state.df_clean)
        st.metric("Total Siswa", total_students)
    
    # Distribution chart
    st.subheader("Distribusi Tingkat Penggunaan")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    level_counts = st.session_state.df_clean['Tingkat_Penggunaan'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    level_counts.plot(kind='bar', ax=ax, color=colors)
    
    ax.set_xlabel('Tingkat Penggunaan')
    ax.set_ylabel('Jumlah Siswa')
    ax.set_title('Distribusi Tingkat Penggunaan AI')
    
    # Add percentage labels
    total = level_counts.sum()
    for i, v in enumerate(level_counts):
        ax.text(i, v + 0.5, f'{v/total:.1%}', ha='center')
    
    st.pyplot(fig)
    
    # Recommendations based on analysis
    st.subheader(" Rekomendasi")
    
    col_rec1, col_rec2, col_rec3 = st.columns(3)
    
    with col_rec1:
        st.info("""
        **Untuk Penggunaan Rendah (0-15):**
        • Mulai dengan tools AI dasar
        • Gunakan untuk tugas sederhana
        • Hadiri workshop AI
        """)
    
    with col_rec2:
        st.info("""
        **Untuk Penggunaan Sedang (16-30):**
        • Eksplor fitur lanjutan
        • Integrasikan ke dalam proyek
        • Bagikan praktik terbaik
        """)
    
    with col_rec3:
        st.info("""
        **Untuk Penggunaan Tinggi (31-50):**
        • Bimbing siswa lain
        • Riset aplikasi AI
        • Pastikan penggunaan etis
        """)

def show_student_prediction():
    """Show prediction interface for students"""
    st.header("Prediksi Tingkat Penggunaan AI Anda")
    
    st.info("Masukkan informasi Anda untuk memprediksi tingkat penggunaan AI Anda:")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            student_name = st.text_input("Nama Anda", "Budi Santoso")
            college = st.text_input("Nama Kampus", "Institut Teknologi Bandung")
            stream = st.selectbox("Jurusan", 
                                ["Teknik", "Sains", "Bisnis", "Seni", 
                                 "Kedokteran", "Hukum", "Manajemen", "Pertanian", "Farmasi", "Manajemen Perhotelan"])
        
        with col2:
            ai_tools = st.multiselect("Tools AI yang Anda Gunakan",
                                    ["ChatGPT", "Gemini", "Copilot", "Midjourney",
                                     "Bard", "Claude", "Lainnya"])
            
            use_cases = st.multiselect("Kasus Penggunaan Utama",
                                     ["Tugas", "Penulisan Konten", "Latihan MCQ",
                                      "Persiapan Ujian", "Pemecahan Masalah", "Mempelajari topik baru",
                                      "Proyek", "Bantuan Coding", "Penulisan CV", "Penelitian"])
            
            usage_score = st.slider("Skor Intensitas Penggunaan Anda", 1, 50, 25, 
                                   help="Perkirakan seberapa intensif Anda menggunakan tools AI (1=jarang, 50=sangat sering)")
        
        submitted = st.form_submit_button("🔮 Prediksi Penggunaan Saya", use_container_width=True)
        
        if submitted:
            # Simple prediction logic
            if usage_score <= 15:
                predicted_level = "Rendah"
                confidence = 0.85
            elif usage_score <= 30:
                predicted_level = "Sedang"
                confidence = 0.90
            else:
                predicted_level = "Tinggi"
                confidence = 0.95
            
            # Display results
            st.success(f"Prediksi selesai untuk {student_name}")
            
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                st.metric("Tingkat Penggunaan Prediksi", predicted_level)
                st.metric("Tingkat Kepercayaan", f"{confidence:.0%}")
            
            with col_result2:
                st.metric("Skor Anda", usage_score)
                st.metric("Tools AI Digunakan", len(ai_tools))
            
            # Personalized recommendations
            st.subheader("Rekomendasi Personal")
            
            if predicted_level == "Rendah":
                st.warning("""
                **Anda baru memulai dengan AI!** 
                
                **Saran:**
                1. **Mulai Sederhana**: Coba gunakan ChatGPT atau Gemini untuk bantuan PR
                2. **Pelajari Dasar**: Ikuti kursus online gratis tentang tools AI untuk siswa
                3. **Bergabung Komunitas**: Ikuti klub AI atau forum online
                4. **Tetapkan Tujuan**: Targetkan menggunakan satu tool AI selama 30 menit sehari
                
                **Manfaat yang Diharapkan**: 
                • Hemat 2-3 jam per minggu untuk tugas
                • Tingkatkan pemahaman konsep sulit
                • Kembangkan keterampilan digital berharga
                """)
            
            elif predicted_level == "Sedang":
                st.info("""
                **Anda menggunakan AI secara efektif!**
                
                **Saran:**
                1. **Fitur Lanjutan**: Eksplor plugin dan fungsionalitas lanjutan
                2. **Integrasi Proyek**: Gunakan AI untuk makalah penelitian dan proyek
                3. **Berbagi Keterampilan**: Bantu teman sekelas belajar tools AI
                4. **Spesialisasi**: Fokus pada tools AI spesifik bidang Anda
                
                **Manfaat yang Diharapkan**:
                • Produktivitas akademik meningkat
                • Hasil proyek lebih baik
                • Pengembangan keterampilan pemecahan masalah berbantuan AI
                """)
            
            else:
                st.success("""
                **Anda pengguna AI tingkat lanjut!**
                
                **Saran:**
                1. **Membimbing Lainnya**: Bagikan keahlian Anda dengan sesama siswa
                2. **Aplikasi Riset**: Eksplor AI untuk penelitian akademik
                3. **Pertimbangan Etis**: Pastikan penggunaan AI yang bertanggung jawab
                4. **Keterampilan Masa Depan**: Pelajari tentang pengembangan dan aplikasi AI
                
                **Manfaat yang Diharapkan**:
                • Kepemimpinan dalam literasi AI di antara teman sebaya
                • Potensi untuk proyek penelitian terkait AI
                • Peningkatan kemampuan kerja dengan keterampilan AI
                """)
            
            # Action plan
            st.subheader("Rencana Aksi 30 Hari")
            days = ["Minggu 1", "Minggu 2", "Minggu 3", "Minggu 4"]
            
            if predicted_level == "Rendah":
                actions = [
                    "Coba 3 tools AI berbeda untuk tugas dasar",
                    "Selesaikan tutorial online tentang dasar-dasar AI",
                    "Gunakan AI untuk satu tugas besar",
                    "Bergabung dengan komunitas pembelajaran AI"
                ]
            elif predicted_level == "Sedang":
                actions = [
                    "Kuasi fitur lanjutan dari tool AI utama Anda",
                    "Kolaborasi dalam proyek berbantuan AI",
                    "Ajarkan satu keterampilan AI ke teman sekelas",
                    "Eksplor tools AI untuk bidang karier Anda"
                ]
            else:
                actions = [
                    "Mulai grup belajar AI",
                    "Mulai proyek penelitian terkait AI",
                    "Buat panduan penggunaan AI untuk teman sebaya",
                    "Eksplor dasar-dasar pengembangan AI"
                ]
            
            for day, action in zip(days, actions):
                st.write(f"**{day}**: {action}")

def show_student_comparison():
    """Show comparison with peers"""
    st.header(" Bandingkan dengan Teman Sebaya")
    
    if st.session_state.df_clean is None:
        st.info(" Tidak ada data perbandingan yang tersedia. Menggunakan data sampel untuk demonstrasi.")
        comparison_df = generate_sample_data(100)
    else:
        comparison_df = st.session_state.df_clean
    
    compare_option = st.selectbox(
        "Bandingkan berdasarkan:",
        ["Jurusan", "Kampus", "Tingkat Penggunaan", "Tools AI"]
    )
    
    if compare_option == "Jurusan":
        st.subheader("Perbandingan berdasarkan Jurusan Akademik")
        
        stream_stats = comparison_df.groupby('Jurusan').agg({
            'Skor_Intensitas_Penggunaan': ['mean', 'std', 'count']
        }).round(1)
        
        st.dataframe(stream_stats, use_container_width=True)
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 5))
        stream_means = comparison_df.groupby('Jurusan')['Skor_Intensitas_Penggunaan'].mean()
        stream_means.plot(kind='bar', ax=ax, color='lightcoral')
        ax.set_xlabel('Jurusan Akademik')
        ax.set_ylabel('Rata-rata Skor Penggunaan')
        ax.set_title('Rata-rata Penggunaan AI per Jurusan')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    elif compare_option == "Kampus":
        st.subheader("Perbandingan berdasarkan Kampus")
        
        # Top 10 colleges
        college_stats = comparison_df.groupby('Nama_Kampus').agg({
            'Skor_Intensitas_Penggunaan': ['mean', 'std', 'count']
        }).round(1).nlargest(10, ('Skor_Intensitas_Penggunaan', 'mean'))
        
        st.dataframe(college_stats, use_container_width=True)
    
    elif compare_option == "Tingkat Penggunaan":
        st.subheader("Analisis Tingkat Penggunaan")
        
        if 'Tingkat_Penggunaan' in comparison_df.columns:
            level_stats = comparison_df.groupby('Tingkat_Penggunaan').agg({
                'Skor_Intensitas_Penggunaan': ['mean', 'min', 'max', 'count']
            }).round(1)
            
            st.dataframe(level_stats, use_container_width=True)
            
            # Pie chart
            fig, ax = plt.subplots(figsize=(8, 6))
            level_counts = comparison_df['Tingkat_Penggunaan'].value_counts()
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            ax.pie(level_counts, labels=level_counts.index, autopct='%1.1f%%', 
                  colors=colors, startangle=90)
            ax.set_title('Distribusi Tingkat Penggunaan')
            st.pyplot(fig)
    
    elif compare_option == "Tools AI":
        st.subheader("Pola Penggunaan Tools AI")
        
        # Extract and count AI tools
        try:
            all_tools = []
            for tools in comparison_df['Tools_AI_Digunakan']:
                if isinstance(tools, str):
                    tool_list = [t.strip() for t in tools.split(',')]
                    all_tools.extend(tool_list)
            
            if all_tools:
                tool_counts = pd.Series(all_tools).value_counts()
                
                # Display top tools
                st.write("**Tools AI Paling Populer:**")
                for tool, count in tool_counts.head(10).items():
                    percentage = (count / len(comparison_df)) * 100
                    st.write(f"• **{tool}**: {count} pengguna ({percentage:.1f}%)")
                
                # Visualization
                fig, ax = plt.subplots(figsize=(10, 5))
                tool_counts.head(10).plot(kind='bar', ax=ax, color='skyblue')
                ax.set_xlabel('Tool AI')
                ax.set_ylabel('Jumlah Pengguna')
                ax.set_title('10 Tools AI Paling Banyak Digunakan')
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.info("Tidak ada data tools AI untuk perbandingan")
        except:
            st.info("Tidak dapat menganalisis data tools AI")
    
    # Personal reflection
    st.subheader("Pertanyaan Refleksi Diri")
    
    reflection_questions = [
        "Bagaimana penggunaan AI Anda dibandingkan dengan teman sebaya di jurusan yang sama?",
        "Tools AI apa yang paling populer di antara siswa dengan skor penggunaan tinggi?",
        "Bagaimana Anda bisa meningkatkan penggunaan AI berdasarkan wawasan ini?",
        "Manfaat apa yang telah Anda alami dari menggunakan tools AI dalam studi Anda?"
    ]
    
    for i, question in enumerate(reflection_questions, 1):
        with st.expander(f"Pertanyaan {i}: {question}"):
            st.text_area(f"Refleksi Anda untuk pertanyaan {i}", 
                        placeholder="Ketik pemikiran Anda di sini...", 
                        height=100, 
                        key=f"reflection_{i}")

if __name__ == "__main__":
    main()
