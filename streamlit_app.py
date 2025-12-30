import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import hashlib
import pickle

# Set page configuration
st.set_page_config(
    page_title="Sistem Analisis Penggunaan AI Mahasiswa",
    page_icon="🎓",
    layout="wide"
)

# CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86C1;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #E8F4F8, #D6EAF8);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 5px solid #2E86C1;
    }
    .success-card {
        background-color: #D5F4E6;
        border-left: 5px solid #28B463;
    }
    .warning-card {
        background-color: #FCF3CF;
        border-left: 5px solid #F39C12;
    }
    .danger-card {
        background-color: #FADBD8;
        border-left: 5px solid #E74C3C;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #2E86C1, #3498DB);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #2874A6, #2E86C1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'encoders' not in st.session_state:
    st.session_state.encoders = {}
if 'results' not in st.session_state:
    st.session_state.results = None
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = None

class KnowledgeBaseSystem:
    """Sistem Knowledge Base untuk rekomendasi penggunaan AI"""
    
    def __init__(self):
        self.rules = {
            'RENDAH': {
                'score_range': (0, 4),
                'label': 'AMAN',
                'color': '#2ECC71',
                'icon': '✅',
                'recommendations': [
                    'Penggunaan AI dalam batas wajar',
                    'Teruskan pola penggunaan yang sehat',
                    'AI digunakan sebagai alat bantu yang tepat',
                    'Pertahankan keseimbangan antara AI dan belajar mandiri'
                ],
                'actions': [
                    'Monitor berkala (bulanan)',
                    'Berikan apresiasi untuk penggunaan bijak',
                    'Dukung eksplorasi fitur AI yang bermanfaat'
                ]
            },
            'SEDANG': {
                'score_range': (5, 7),
                'label': 'PERLU PERHATIAN',
                'color': '#F39C12',
                'icon': '⚠️',
                'recommendations': [
                    'Penggunaan AI mulai intensif',
                    'Perlu pengawasan lebih lanjut',
                    'Evaluasi ketergantungan pada AI',
                    'Batasi penggunaan untuk tugas tertentu saja'
                ],
                'actions': [
                    'Konsultasi dengan dosen pembimbing',
                    'Buat jadwal penggunaan AI',
                    'Ikuti workshop penggunaan AI yang bertanggung jawab',
                    'Laporan penggunaan mingguan'
                ]
            },
            'TINGGI': {
                'score_range': (8, 10),
                'label': 'BUTUH PENGAWASAN',
                'color': '#E74C3C',
                'icon': '🚨',
                'recommendations': [
                    'Penggunaan AI sangat intensif',
                    'Potensi ketergantungan tinggi',
                    'Perlu intervensi segera',
                    'Risiko plagiarisme meningkat'
                ],
                'actions': [
                    'Pembatasan akses tools AI',
                    'Konsultasi wajib dengan pembimbing',
                    'Monitoring ketat setiap minggu',
                    'Program pengurangan ketergantungan AI',
                    'Evaluasi ulang semua tugas'
                ]
            }
        }
    
    def classify_usage(self, score):
        """Klasifikasi berdasarkan skor penggunaan"""
        if score <= 4:
            return 'RENDAH'
        elif 5 <= score <= 7:
            return 'SEDANG'
        else:
            return 'TINGGI'
    
    def get_recommendation(self, classification, student_data):
        """Dapatkan rekomendasi berdasarkan klasifikasi"""
        rule = self.rules[classification]
        
        recommendation = {
            'classification': classification,
            'label': rule['label'],
            'color': rule['color'],
            'icon': rule['icon'],
            'student_name': student_data.get('Nama', ''),
            'jurusan': student_data.get('Studi_Jurusan', ''),
            'semester': student_data.get('Semester', ''),
            'ai_tool': student_data.get('AI_Tools', ''),
            'trust_level': student_data.get('Trust_Level', ''),
            'usage_score': student_data.get('Usage_Intensity_Score', ''),
            'main_recommendation': f"{rule['icon']} {rule['label']}",
            'details': rule['recommendations'],
            'actions': rule['actions'],
            'monitoring_level': {
                'RENDAH': 'RINGAN',
                'SEDANG': 'SEDANG',
                'TINGGI': 'TINGGI'
            }[classification]
        }
        
        return recommendation
    
    def generate_summary_report(self, classifications):
        """Hasilkan laporan ringkasan"""
        total = len(classifications)
        counts = {
            'RENDAH': classifications.count('RENDAH'),
            'SEDANG': classifications.count('SEDANG'),
            'TINGGI': classifications.count('TINGGI')
        }
        
        percentages = {k: (v/total)*100 for k, v in counts.items()}
        
        return {
            'total_students': total,
            'counts': counts,
            'percentages': percentages,
            'dominant_class': max(counts, key=counts.get),
            'risk_level': 'TINGGI' if percentages['TINGGI'] > 30 else 'SEDANG' if percentages['TINGGI'] > 10 else 'RENDAH'
        }

def load_dataset():
    """Load dataset dari data yang diberikan"""
    data = {
        'Nama': [
            'Althaf Rayyan Putra', 'Ayesha Kinanti', 'Salsabila Nurfadila', 'Anindya Safira',
            'Iqbal Ramadhan', 'Muhammad Rizky Pratama', 'Fikri Alfarizi', 'Iqbal Ramadhan',
            'Citra Maharani', 'Iqbal Ramadhan', 'Zidan Harits', 'Rizky Kurniawan Putra',
            'Raka Bimantara', 'Zahra Alya Safitri', 'Muhammad Naufal Haidar', 'Citra Maharani',
            'Ammar Zaky Firmansyah', 'Ilham Nurhadi', 'Muhammad Rizky Pratama', 'Nayla Syakira',
            'Zidan Harits', 'Citra Maharani', 'Arfan Maulana', 'Nabila Khairunnisa',
            'Safira Azzahra Putri', 'Farah Amalia', 'Muhammad Reza Ananda', 'Citra Maharani',
            'Aulia Rahma', 'Yusuf Al Hakim', 'Salsabila Nurfadila', 'Aulia Rahma',
            'Ayesha Kinanti', 'Damar Alif Prakoso', 'Zidan Harits', 'Ammar Zaky Firmansyah',
            'Citra Maharani', 'Nabila Khairunnisa', 'Farah Amalia', 'Ammar Zaky Firmansyah',
            'Zidan Harits', 'Farah Amalia', 'Ahmad Fauzan Maulana', 'Khansa Humaira Zahira'
        ],
        'Studi_Jurusan': [
            'Teknologi Informasi', 'Teknologi Informasi', 'Teknik Informatika', 'Teknik Informatika',
            'Farmasi', 'Teknologi Informasi', 'Teknologi Informasi', 'Farmasi',
            'Keperawatan', 'Farmasi', 'Farmasi', 'Teknik Informatika',
            'Farmasi', 'Teknik Informatika', 'Farmasi', 'Keperawatan',
            'Farmasi', 'Teknologi Informasi', 'Teknologi Informasi', 'Teknologi Informasi',
            'Farmasi', 'Keperawatan', 'Teknik Informatika', 'Teknologi Informasi',
            'Teknologi Informasi', 'Teknologi Informasi', 'Teknologi Informasi', 'Keperawatan',
            'Teknik Informatika', 'Teknik Informatika', 'Teknik Informatika', 'Teknik Informatika',
            'Teknologi Informasi', 'Teknologi Informasi', 'Farmasi', 'Farmasi',
            'Keperawatan', 'Teknologi Informasi', 'Teknologi Informasi', 'Farmasi',
            'Farmasi', 'Teknologi Informasi', 'Farmasi', 'Teknik Informatika'
        ],
        'Semester': [7, 3, 1, 5, 1, 5, 1, 3, 5, 3, 7, 5, 3, 3, 1, 1, 3, 1, 1, 7,
                     3, 5, 7, 1, 1, 1, 1, 5, 3, 5, 1, 5, 1, 7, 3, 1, 3, 1, 7, 1,
                     7, 5, 1, 1],
        'AI_Tools': [
            'Gemini', 'Gemini', 'Gemini', 'Gemini', 'Gemini', 'Gemini', 'ChatGPT', 'Gemini',
            'Multiple', 'Gemini', 'Gemini', 'ChatGPT', 'ChatGPT', 'Gemini', 'Gemini', 'Copilot',
            'ChatGPT', 'Gemini', 'Multiple', 'ChatGPT', 'Gemini', 'ChatGPT', 'ChatGPT', 'Gemini',
            'Gemini', 'ChatGPT', 'ChatGPT', 'ChatGPT', 'Gemini', 'ChatGPT', 'Multiple', 'ChatGPT',
            'Copilot', 'ChatGPT', 'ChatGPT', 'ChatGPT', 'Gemini', 'Copilot', 'ChatGPT', 'Gemini',
            'ChatGPT', 'ChatGPT', 'Copilot', 'ChatGPT'
        ],
        'Trust_Level': [4, 4, 5, 4, 5, 4, 4, 5, 4, 5, 4, 4, 4, 4, 4, 5, 5, 5, 5, 4,
                        5, 2, 1, 5, 5, 5, 4, 1, 1, 3, 4, 4, 3, 4, 2, 4, 4, 5, 4, 4,
                        2, 3, 4, 4],
        'Usage_Intensity_Score': [
            8, 9, 3, 6, 10, 4, 7, 9, 2, 9, 4, 3, 4, 10, 3, 8, 7, 9, 3, 8,
            10, 8, 2, 9, 7, 6, 3, 3, 2, 8, 3, 10, 4, 3, 7, 10, 8, 6, 6, 2,
            1, 5, 7, 7
        ]
    }
    
    df = pd.DataFrame(data)
    return df

def clean_data(df):
    """Pembersihan data"""
    df_clean = df.copy()
    
    # Handle duplicate entries (keep first)
    df_clean = df_clean.drop_duplicates(subset=['Nama'], keep='first')
    
    # Convert Usage_Intensity_Score to numeric
    df_clean['Usage_Intensity_Score'] = df_clean['Usage_Intensity_Score'].apply(
        lambda x: 10 if str(x) == '10+' else float(x)
    )
    
    # Handle missing values
    for col in ['Semester', 'Trust_Level', 'Usage_Intensity_Score']:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    return df_clean

def encode_categorical_data(df):
    """Encode data kategorikal"""
    df_encoded = df.copy()
    encoders = {}
    
    # Encode Studi_Jurusan
    le_jurusan = LabelEncoder()
    df_encoded['Studi_Jurusan_Encoded'] = le_jurusan.fit_transform(df_encoded['Studi_Jurusan'])
    encoders['Studi_Jurusan'] = le_jurusan
    
    # Encode AI_Tools
    le_tools = LabelEncoder()
    df_encoded['AI_Tools_Encoded'] = le_tools.fit_transform(df_encoded['AI_Tools'])
    encoders['AI_Tools'] = le_tools
    
    # Create target variable (klasifikasi penggunaan)
    kb = KnowledgeBaseSystem()
    df_encoded['Usage_Level'] = df_encoded['Usage_Intensity_Score'].apply(kb.classify_usage)
    
    # Encode target variable
    le_target = LabelEncoder()
    df_encoded['Usage_Level_Encoded'] = le_target.fit_transform(df_encoded['Usage_Level'])
    encoders['Usage_Level'] = le_target
    
    return df_encoded, encoders

def prepare_features(df_encoded):
    """Persiapkan fitur untuk model"""
    features = ['Semester', 'Trust_Level', 'Studi_Jurusan_Encoded', 'AI_Tools_Encoded']
    X = df_encoded[features]
    y = df_encoded['Usage_Level_Encoded']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def train_random_forest(X, y):
    """Train model Random Forest"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def create_visualizations(df, predictions):
    """Buat visualisasi data"""
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Distribusi Tingkat Penggunaan AI', 
                       'Penggunaan AI per Jurusan',
                       'Hubungan Trust Level vs Penggunaan',
                       'Tools AI yang Digunakan'),
        specs=[[{'type': 'pie'}, {'type': 'bar'}],
               [{'type': 'scatter'}, {'type': 'bar'}]]
    )
    
    # 1. Pie chart for usage distribution
    usage_counts = predictions['Usage_Level'].value_counts()
    fig.add_trace(
        go.Pie(labels=usage_counts.index, values=usage_counts.values, hole=0.3),
        row=1, col=1
    )
    
    # 2. Bar chart for usage per jurusan
    jurusan_usage = df.groupby('Studi_Jurusan')['Usage_Intensity_Score'].mean().reset_index()
    fig.add_trace(
        go.Bar(x=jurusan_usage['Studi_Jurusan'], y=jurusan_usage['Usage_Intensity_Score']),
        row=1, col=2
    )
    
    # 3. Scatter plot for trust vs usage
    fig.add_trace(
        go.Scatter(x=df['Trust_Level'], y=df['Usage_Intensity_Score'],
                  mode='markers', marker=dict(size=10, color=df['Usage_Intensity_Score'])),
        row=2, col=1
    )
    
    # 4. Bar chart for AI tools usage
    tools_count = df['AI_Tools'].value_counts()
    fig.add_trace(
        go.Bar(x=tools_count.index, y=tools_count.values),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False, title_text="Analisis Penggunaan AI Mahasiswa")
    return fig

def login_page():
    """Halaman login"""
    st.markdown("<h1 class='main-header'>🔐 Sistem Analisis Penggunaan AI Mahasiswa</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        role = st.selectbox("Pilih Peran", ["Guru/Dosen", "Mahasiswa"])
        
        if role == "Guru/Dosen":
            password = st.text_input("Password", type="password")
            login_button = st.button("Login sebagai Guru")
            
            if login_button:
                # Simple password for demo
                if password == "guru123":
                    st.session_state.authenticated = True
                    st.session_state.user_role = "guru"
                    st.success("Login berhasil! Mengarahkan ke dashboard...")
                    st.rerun()
                else:
                    st.error("Password salah!")
        
        else:  # Mahasiswa
            student_name = st.text_input("Nama Mahasiswa")
            login_button = st.button("Login sebagai Mahasiswa")
            
            if login_button:
                if student_name.strip():
                    st.session_state.authenticated = True
                    st.session_state.user_role = "mahasiswa"
                    st.session_state.student_name = student_name.strip()
                    st.success(f"Selamat datang, {student_name}!")
                    st.rerun()
                else:
                    st.error("Harap masukkan nama!")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Info tambahan
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### ℹ️ Informasi Sistem")
        st.markdown("""
        **Sistem ini bertujuan untuk:**
        - Menganalisis tingkat penggunaan AI di kalangan mahasiswa
        - Mengklasifikasikan tingkat penggunaan (Rendah, Sedang, Tinggi)
        - Memberikan rekomendasi berdasarkan hasil analisis
        - Membantu monitoring penggunaan AI yang sehat
        """)
        st.markdown("</div>", unsafe_allow_html=True)

def guru_dashboard():
    """Dashboard untuk Guru/Dosen"""
    st.markdown("<h1 class='main-header'>🧑‍🏫 Dashboard Guru/Dosen</h1>", unsafe_allow_html=True)
    
    # Initialize knowledge base
    if st.session_state.knowledge_base is None:
        st.session_state.knowledge_base = KnowledgeBaseSystem()
    
    # Sidebar menu
    with st.sidebar:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Menu Analisis")
        menu = st.radio(
            "Pilih Menu:",
            ["📁 Data Management", "🧹 Data Cleaning", "🔧 Data Processing", 
             "🤖 Model Training", "📈 Evaluasi Model", "🎯 Rekomendasi", "📊 Dashboard Analitik"]
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Logout button
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.session_state.user_role = None
            st.rerun()
    
    # Main content based on menu
    if menu == "📁 Data Management":
        data_management()
    elif menu == "🧹 Data Cleaning":
        data_cleaning_section()
    elif menu == "🔧 Data Processing":
        data_processing()
    elif menu == "🤖 Model Training":
        model_training()
    elif menu == "📈 Evaluasi Model":
        model_evaluation()
    elif menu == "🎯 Rekomendasi":
        recommendations_section()
    elif menu == "📊 Dashboard Analitik":
        analytics_dashboard()

def data_management():
    """Manajemen data"""
    st.header("📁 Management Data")
    
    # Load dataset
    if st.session_state.df is None:
        if st.button("📥 Load Dataset"):
            st.session_state.df = load_dataset()
            st.success("Dataset berhasil dimuat!")
    
    if st.session_state.df is not None:
        # Display data
        st.subheader("📋 Dataset Mahasiswa")
        st.dataframe(st.session_state.df, use_container_width=True)
        
        # Data statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Jumlah Data", len(st.session_state.df))
        with col2:
            st.metric("Jumlah Jurusan", st.session_state.df['Studi_Jurusan'].nunique())
        with col3:
            avg_usage = st.session_state.df['Usage_Intensity_Score'].apply(
                lambda x: 10 if str(x) == '10+' else float(x)
            ).mean()
            st.metric("Rata-rata Penggunaan", f"{avg_usage:.2f}")
        with col4:
            st.metric("Tools AI Berbeda", st.session_state.df['AI_Tools'].nunique())
        
        # Data preview
        st.subheader("👀 Preview Data")
        tab1, tab2, tab3 = st.tabs(["Statistik", "Distribusi", "Info"])
        
        with tab1:
            st.write(st.session_state.df.describe())
        
        with tab2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Distribution of Usage Intensity
            usage_scores = st.session_state.df['Usage_Intensity_Score'].apply(
                lambda x: 10 if str(x) == '10+' else float(x)
            )
            ax1.hist(usage_scores, bins=10, edgecolor='black', alpha=0.7)
            ax1.set_title('Distribusi Skor Penggunaan AI')
            ax1.set_xlabel('Skor Penggunaan')
            ax1.set_ylabel('Frekuensi')
            
            # Distribution of Trust Level
            trust_counts = st.session_state.df['Trust_Level'].value_counts().sort_index()
            ax2.bar(trust_counts.index, trust_counts.values, alpha=0.7)
            ax2.set_title('Distribusi Tingkat Kepercayaan AI')
            ax2.set_xlabel('Tingkat Kepercayaan')
            ax2.set_ylabel('Frekuensi')
            
            plt.tight_layout()
            st.pyplot(fig)

def data_cleaning_section():
    """Data cleaning section"""
    st.header("🧹 Data Cleaning")
    
    if st.session_state.df is None:
        st.warning("Harap load dataset terlebih dahulu di menu Data Management!")
        return
    
    st.info("""
    **Proses Data Cleaning:**
    1. Menghapus data duplikat
    2. Mengonversi nilai '10+' menjadi 10
    3. Menangani missing values
    4. Validasi tipe data
    """)
    
    if st.button("🚀 Jalankan Data Cleaning", type="primary"):
        with st.spinner("Melakukan data cleaning..."):
            st.session_state.df_clean = clean_data(st.session_state.df)
            st.success("✅ Data cleaning selesai!")
            
            # Show cleaning results
            st.subheader("📊 Hasil Data Cleaning")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("**Sebelum Cleaning:**")
                st.write(f"- Jumlah data: {len(st.session_state.df)}")
                st.write(f"- Data duplikat: {st.session_state.df.duplicated().sum()}")
                st.write(f"- Missing values: {st.session_state.df.isnull().sum().sum()}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='card success-card'>", unsafe_allow_html=True)
                st.markdown("**Setelah Cleaning:**")
                st.write(f"- Jumlah data: {len(st.session_state.df_clean)}")
                st.write(f"- Data duplikat: {st.session_state.df_clean.duplicated().sum()}")
                st.write(f"- Missing values: {st.session_state.df_clean.isnull().sum().sum()}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Show cleaned data
            st.subheader("📋 Data Setelah Cleaning")
            st.dataframe(st.session_state.df_clean, use_container_width=True)

def data_processing():
    """Data processing section"""
    st.header("🔧 Data Processing & Encoding")
    
    if st.session_state.df_clean is None:
        st.warning("Harap jalankan data cleaning terlebih dahulu!")
        return
    
    st.info("""
    **Proses Data Processing:**
    1. Encoding data kategorikal (jurusan, tools AI)
    2. Menyiapkan fitur untuk model
    3. Membuat target variable (klasifikasi)
    4. Normalisasi data
    """)
    
    if st.button("⚙️ Proses Data", type="primary"):
        with st.spinner("Memproses data..."):
            # Encode categorical data
            df_encoded, encoders = encode_categorical_data(st.session_state.df_clean)
            st.session_state.encoders = encoders
            
            # Prepare features
            X_scaled, y, scaler = prepare_features(df_encoded)
            
            # Save processed data
            st.session_state.processed_data = {
                'X': X_scaled,
                'y': y,
                'df_encoded': df_encoded,
                'scaler': scaler
            }
            
            st.success("✅ Data processing selesai!")
            
            # Show encoding results
            st.subheader("📊 Hasil Encoding")
            
            # Show encoded values
            st.write("**Mapping Jurusan:**")
            jurusan_mapping = dict(zip(
                st.session_state.encoders['Studi_Jurusan'].classes_,
                range(len(st.session_state.encoders['Studi_Jurusan'].classes_))
            ))
            st.write(jurusan_mapping)
            
            st.write("**Mapping AI Tools:**")
            tools_mapping = dict(zip(
                st.session_state.encoders['AI_Tools'].classes_,
                range(len(st.session_state.encoders['AI_Tools'].classes_))
            ))
            st.write(tools_mapping)
            
            # Show processed data
            st.subheader("📋 Data Setelah Processing")
            st.dataframe(df_encoded.head(), use_container_width=True)
            
            # Show target distribution
            st.subheader("🎯 Distribusi Target (Usage Level)")
            level_counts = df_encoded['Usage_Level'].value_counts()
            
            fig = px.pie(values=level_counts.values, names=level_counts.index,
                        title="Distribusi Tingkat Penggunaan AI")
            st.plotly_chart(fig, use_container_width=True)

def model_training():
    """Model training section"""
    st.header("🤖 Model Training - Random Forest")
    
    if 'processed_data' not in st.session_state:
        st.warning("Harap proses data terlebih dahulu di menu Data Processing!")
        return
    
    st.info("""
    **Algoritma Random Forest:**
    - Ensemble learning method
    - Multiple decision trees
    - Bagging technique
    - Robust terhadap overfitting
    """)
    
    # Model parameters
    st.subheader("⚙️ Parameter Model")
    
    col1, col2 = st.columns(2)
    with col1:
        n_estimators = st.slider("Jumlah Trees", 50, 200, 100)
        max_depth = st.slider("Max Depth", 3, 10, 5)
    
    with col2:
        min_samples_split = st.slider("Min Samples Split", 2, 10, 5)
        min_samples_leaf = st.slider("Min Samples Leaf", 1, 5, 2)
    
    if st.button("🎯 Train Model", type="primary"):
        with st.spinner("Melatih model Random Forest..."):
            # Train model
            X = st.session_state.processed_data['X']
            y = st.session_state.processed_data['y']
            
            model_params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'random_state': 42
            }
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            model = RandomForestClassifier(**model_params)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Save model and results
            st.session_state.model = {
                'model': model,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'y_pred': y_pred,
                'params': model_params
            }
            
            st.success("✅ Model berhasil dilatih!")
            
            # Show training results
            accuracy = accuracy_score(y_test, y_pred)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{accuracy:.2%}")
            with col2:
                st.metric("Jumlah Trees", n_estimators)
            with col3:
                st.metric("Max Depth", max_depth)
            
            # Feature importance
            st.subheader("📊 Feature Importance")
            feature_names = ['Semester', 'Trust Level', 'Jurusan', 'AI Tools']
            importances = model.feature_importances_
            
            fig = px.bar(x=feature_names, y=importances,
                        title="Importance Fitur dalam Model")
            st.plotly_chart(fig, use_container_width=True)

def model_evaluation():
    """Model evaluation section"""
    st.header("📈 Evaluasi Model")
    
    if 'model' not in st.session_state:
        st.warning("Harap train model terlebih dahulu!")
        return
    
    model_data = st.session_state.model
    
    st.info("""
    **Metrik Evaluasi:**
    - Accuracy: Proporsi prediksi benar dari total prediksi
    - Precision: Proporsi prediksi positif yang benar
    - Recall: Proporsi data positif yang terdeteksi
    - F1-Score: Rata-rata harmonik precision dan recall
    """)
    
    # Calculate metrics
    accuracy = accuracy_score(model_data['y_test'], model_data['y_pred'])
    report = classification_report(model_data['y_test'], model_data['y_pred'], 
                                  target_names=['RENDAH', 'SEDANG', 'TINGGI'],
                                  output_dict=True)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{accuracy:.2%}")
    with col2:
        st.metric("Precision", f"{report['weighted avg']['precision']:.2%}")
    with col3:
        st.metric("Recall", f"{report['weighted avg']['recall']:.2%}")
    with col4:
        st.metric("F1-Score", f"{report['weighted avg']['f1-score']:.2%}")
    
    # Classification report
    st.subheader("📋 Classification Report")
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
    
    # Confusion matrix
    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(model_data['y_test'], model_data['y_pred'])
    
    fig = px.imshow(cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['RENDAH', 'SEDANG', 'TINGGI'],
                    y=['RENDAH', 'SEDANG', 'TINGGI'],
                    title="Confusion Matrix",
                    text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Actual vs Predicted comparison
    st.subheader("📈 Perbandingan Aktual vs Prediksi")
    
    # Decode predictions
    le_target = st.session_state.encoders['Usage_Level']
    y_test_decoded = le_target.inverse_transform(model_data['y_test'])
    y_pred_decoded = le_target.inverse_transform(model_data['y_pred'])
    
    comparison_df = pd.DataFrame({
        'Actual': y_test_decoded,
        'Predicted': y_pred_decoded
    })
    
    st.dataframe(comparison_df, use_container_width=True)

def recommendations_section():
    """Recommendations section"""
    st.header("🎯 Rekomendasi Berdasarkan Analisis")
    
    if st.session_state.df_clean is None or st.session_state.knowledge_base is None:
        st.warning("Data belum siap untuk analisis rekomendasi!")
        return
    
    # Generate recommendations for all students
    if st.button("🔄 Generate Rekomendasi", type="primary"):
        with st.spinner("Membuat rekomendasi..."):
            kb = st.session_state.knowledge_base
            df = st.session_state.df_clean
            
            # Classify each student
            recommendations = []
            classifications = []
            
            for _, student in df.iterrows():
                score = student['Usage_Intensity_Score']
                if isinstance(score, str) and score == '10+':
                    score = 10
                
                classification = kb.classify_usage(score)
                classifications.append(classification)
                
                recommendation = kb.get_recommendation(classification, student)
                recommendations.append(recommendation)
            
            # Generate summary report
            summary = kb.generate_summary_report(classifications)
            
            # Save results
            st.session_state.results = {
                'recommendations': recommendations,
                'summary': summary,
                'classifications': classifications
            }
            
            st.success("✅ Rekomendasi berhasil dibuat!")
    
    if st.session_state.results:
        results = st.session_state.results
        summary = results['summary']
        
        # Display summary
        st.subheader("📊 Ringkasan Analisis")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Mahasiswa", summary['total_students'])
        with col2:
            st.metric("Tingkat Risiko", summary['risk_level'])
        with col3:
            st.metric("Level Dominan", summary['dominant_class'])
        with col4:
            st.metric("Butuh Pengawasan", f"{summary['counts']['TINGGI']} siswa")
        
        # Display recommendations by category
        st.subheader("📋 Detail Rekomendasi per Kategori")
        
        tabs = st.tabs(["🔴 TINGGI", "🟡 SEDANG", "🟢 RENDAH"])
        
        categories = ['TINGGI', 'SEDANG', 'RENDAH']
        for tab, category in zip(tabs, categories):
            with tab:
                category_students = [r for r in results['recommendations'] if r['classification'] == category]
                
                if category_students:
                    st.markdown(f"### {len(category_students)} Mahasiswa dengan Level {category}")
                    
                    for student_rec in category_students:
                        color_class = {
                            'TINGGI': 'danger-card',
                            'SEDANG': 'warning-card',
                            'RENDAH': 'success-card'
                        }[category]
                        
                        st.markdown(f"<div class='card {color_class}'>", unsafe_allow_html=True)
                        st.markdown(f"**{student_rec['icon']} {student_rec['student_name']}**")
                        st.markdown(f"*{student_rec['jurusan']} - Semester {student_rec['semester']}*")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**AI Tool:** {student_rec['ai_tool']}")
                            st.markdown(f"**Trust Level:** {student_rec['trust_level']}/5")
                        with col2:
                            st.markdown(f"**Usage Score:** {student_rec['usage_score']}/10")
                            st.markdown(f"**Monitoring:** {student_rec['monitoring_level']}")
                        
                        st.markdown("---")
                        st.markdown("**Rekomendasi:**")
                        for rec in student_rec['details']:
                            st.markdown(f"• {rec}")
                        
                        st.markdown("**Tindakan:**")
                        for action in student_rec['actions']:
                            st.markdown(f"▶️ {action}")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.info(f"Tidak ada mahasiswa dengan level {category}")
        
        # Export options
        st.subheader("📥 Export Hasil")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Export ke CSV"):
                # Convert to DataFrame for export
                export_data = []
                for rec in results['recommendations']:
                    export_data.append({
                        'Nama': rec['student_name'],
                        'Jurusan': rec['jurusan'],
                        'Semester': rec['semester'],
                        'AI_Tool': rec['ai_tool'],
                        'Trust_Level': rec['trust_level'],
                        'Usage_Score': rec['usage_score'],
                        'Klasifikasi': rec['classification'],
                        'Label': rec['label'],
                        'Monitoring_Level': rec['monitoring_level']
                    })
                
                export_df = pd.DataFrame(export_data)
                csv = export_df.to_csv(index=False)
                
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="rekomendasi_penggunaan_ai.csv",
                    mime="text/csv"
                )

def analytics_dashboard():
    """Analytics dashboard"""
    st.header("📊 Dashboard Analitik")
    
    if st.session_state.df_clean is None:
        st.warning("Data belum tersedia untuk analisis!")
        return
    
    df = st.session_state.df_clean
    
    # Key metrics
    st.subheader("📈 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_usage = df['Usage_Intensity_Score'].apply(
            lambda x: 10 if str(x) == '10+' else float(x)
        ).mean()
        st.metric("Rata-rata Penggunaan", f"{avg_usage:.2f}")
    
    with col2:
        avg_trust = df['Trust_Level'].mean()
        st.metric("Rata-rata Trust Level", f"{avg_trust:.2f}")
    
    with col3:
        unique_tools = df['AI_Tools'].nunique()
        st.metric("Jenis Tools AI", unique_tools)
    
    with col4:
        high_users = df[df['Usage_Intensity_Score'].apply(
            lambda x: 10 if str(x) == '10+' else float(x)
        ) > 7].shape[0]
        st.metric("Pengguna Intensif", high_users)
    
    # Interactive visualizations
    st.subheader("📊 Visualisasi Interaktif")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Distribusi", "Per Jurusan", "Tools AI", "Korelasi"])
    
    with tab1:
        # Distribution visualization
        fig = px.histogram(df, x='Usage_Intensity_Score',
                          title='Distribusi Skor Penggunaan AI',
                          nbins=10,
                          color_discrete_sequence=['#2E86C1'])
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Usage by study program
        jurusan_usage = df.groupby('Studi_Jurusan').agg({
            'Usage_Intensity_Score': 'mean',
            'Trust_Level': 'mean',
            'Nama': 'count'
        }).reset_index()
        jurusan_usage.columns = ['Jurusan', 'Avg_Usage', 'Avg_Trust', 'Jumlah_Mahasiswa']
        
        fig = px.bar(jurusan_usage, x='Jurusan', y='Avg_Usage',
                    title='Rata-rata Penggunaan AI per Jurusan',
                    color='Avg_Usage',
                    color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # AI Tools usage
        tools_usage = df.groupby('AI_Tools').agg({
            'Usage_Intensity_Score': 'mean',
            'Trust_Level': 'mean',
            'Nama': 'count'
        }).reset_index()
        
        fig = px.scatter(tools_usage, x='Usage_Intensity_Score', y='Trust_Level',
                        size='Nama', color='AI_Tools',
                        title='Tools AI vs Penggunaan & Kepercayaan',
                        hover_name='AI_Tools')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Correlation matrix
        numeric_df = df.copy()
        numeric_df['Usage_Intensity_Score'] = numeric_df['Usage_Intensity_Score'].apply(
            lambda x: 10 if str(x) == '10+' else float(x)
        )
        
        # Create correlation matrix
        correlation = numeric_df[['Semester', 'Trust_Level', 'Usage_Intensity_Score']].corr()
        
        fig = px.imshow(correlation,
                       title='Korelasi antar Variabel',
                       color_continuous_scale='RdBu',
                       zmin=-1, zmax=1)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed analysis
    if st.session_state.results:
        st.subheader("🎯 Analisis Rekomendasi")
        
        results = st.session_state.results
        summary = results['summary']
        
        # Pie chart of classifications
        fig = px.pie(values=list(summary['counts'].values()),
                    names=list(summary['counts'].keys()),
                    title='Distribusi Klasifikasi Penggunaan AI',
                    color=list(summary['counts'].keys()),
                    color_discrete_map={
                        'RENDAH': '#2ECC71',
                        'SEDANG': '#F39C12',
                        'TINGGI': '#E74C3C'
                    })
        st.plotly_chart(fig, use_container_width=True)

def mahasiswa_dashboard():
    """Dashboard untuk Mahasiswa"""
    st.markdown(f"<h1 class='main-header'>👨‍🎓 Dashboard Mahasiswa</h1>", unsafe_allow_html=True)
    
    student_name = st.session_state.get('student_name', '')
    
    if student_name:
        st.markdown(f"<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"### 👋 Selamat datang, **{student_name}**!")
        st.markdown("Di sini Anda dapat melihat hasil analisis penggunaan AI Anda.")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Check if analysis is available
        if st.session_state.df_clean is not None and st.session_state.knowledge_base is not None:
            # Find student data
            student_data = st.session_state.df_clean[
                st.session_state.df_clean['Nama'].str.contains(student_name, case=False, na=False)
            ]
            
            if not student_data.empty:
                student = student_data.iloc[0]
                
                # Classify student
                kb = st.session_state.knowledge_base
                score = student['Usage_Intensity_Score']
                if isinstance(score, str) and score == '10+':
                    score = 10
                
                classification = kb.classify_usage(score)
                recommendation = kb.get_recommendation(classification, student)
                
                # Display student info
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("### 📋 Profil Akademik")
                    st.markdown(f"**Nama:** {student['Nama']}")
                    st.markdown(f"**Jurusan:** {student['Studi_Jurusan']}")
                    st.markdown(f"**Semester:** {student['Semester']}")
                    st.markdown(f"**Tools AI:** {student['AI_Tools']}")
                    st.markdown(f"**Trust Level:** {student['Trust_Level']}/5")
                    st.markdown(f"**Usage Score:** {student['Usage_Intensity_Score']}/10")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    # Display classification with appropriate color
                    color_class = {
                        'RENDAH': 'success-card',
                        'SEDANG': 'warning-card',
                        'TINGGI': 'danger-card'
                    }[classification]
                    
                    st.markdown(f"<div class='card {color_class}'>", unsafe_allow_html=True)
                    st.markdown(f"### 🎯 Hasil Analisis")
                    st.markdown(f"**{recommendation['icon']} Klasifikasi:** {classification}")
                    st.markdown(f"**🏷️ Label:** {recommendation['label']}")
                    st.markdown(f"**📊 Level Monitoring:** {recommendation['monitoring_level']}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Display recommendations
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("### 📝 Rekomendasi untuk Anda")
                
                st.markdown("**Detail Rekomendasi:**")
                for rec in recommendation['details']:
                    st.markdown(f"• {rec}")
                
                st.markdown("**Tindakan yang Disarankan:**")
                for action in recommendation['actions']:
                    st.markdown(f"▶️ {action}")
                
                # Additional tips
                st.markdown("**💡 Tips Penggunaan AI yang Sehat:**")
                tips = [
                    "Gunakan AI sebagai alat bantu, bukan pengganti pemikiran",
                    "Selalu verifikasi informasi dari AI dengan sumber terpercaya",
                    "Jaga keseimbangan antara penggunaan AI dan belajar mandiri",
                    "Diskusikan dengan dosen tentang penggunaan AI yang tepat",
                    "Ikuti perkembangan etika penggunaan AI dalam akademik"
                ]
                
                for tip in tips:
                    st.markdown(f"✨ {tip}")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Visual representation
                st.markdown("### 📊 Visualisasi Penggunaan")
                
                # Create gauge chart for usage score
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=float(score),
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Skor Penggunaan AI"},
                    gauge={
                        'axis': {'range': [0, 10]},
                        'bar': {'color': recommendation['color']},
                        'steps': [
                            {'range': [0, 4], 'color': '#2ECC71'},
                            {'range': [4, 7], 'color': '#F39C12'},
                            {'range': [7, 10], 'color': '#E74C3C'}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': float(score)
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.warning(f"Data tidak ditemukan untuk nama '{student_name}'.")
                st.info("Pastikan nama Anda terdaftar dalam sistem atau hubungi administrator.")
        else:
            st.info("🔄 Analisis sedang dipersiapkan. Silakan coba lagi nanti.")
    
    # Logout button
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.session_state.user_role = None
        st.rerun()

def main():
    """Main application"""
    if not st.session_state.authenticated:
        login_page()
    else:
        if st.session_state.user_role == "guru":
            guru_dashboard()
        elif st.session_state.user_role == "mahasiswa":
            mahasiswa_dashboard()

if __name__ == "__main__":
    main()
