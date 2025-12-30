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
import io
import base64
from datetime import datetime

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
    .upload-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .file-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border: 2px dashed #dee2e6;
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
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = "default"

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

def load_default_dataset():
    """Load dataset default dari data yang diberikan"""
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

def validate_dataset(df):
    """Validasi dataset yang diupload"""
    required_columns = ['Nama', 'Studi_Jurusan', 'Semester', 'AI_Tools', 'Trust_Level', 'Usage_Intensity_Score']
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Kolom yang hilang: {', '.join(missing_columns)}"
    
    # Check data types
    try:
        df['Semester'] = pd.to_numeric(df['Semester'])
        df['Trust_Level'] = pd.to_numeric(df['Trust_Level'])
        df['Usage_Intensity_Score'] = pd.to_numeric(df['Usage_Intensity_Score'].astype(str).str.replace('10\+', '10', regex=True))
    except:
        return False, "Error konversi tipe data. Pastikan Semester, Trust_Level, dan Usage_Intensity_Score adalah angka"
    
    # Check for empty values
    if df[required_columns].isnull().any().any():
        return False, "Dataset mengandung nilai kosong (null)"
    
    return True, "Dataset valid"

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

def get_download_link(df, filename="dataset.csv", text="Download CSV"):
    """Generate download link for CSV file"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; border-radius: 5px;">{text}</a>'
    return href

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
        
        **Format Dataset:**
        - File CSV dengan kolom: Nama, Studi_Jurusan, Semester, AI_Tools, Trust_Level, Usage_Intensity_Score
        - Semester, Trust_Level, Usage_Intensity_Score harus angka
        - Usage_Intensity_Score: 0-10 (atau 10+ akan dianggap sebagai 10)
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
            ["📁 Upload Dataset", "🧹 Data Cleaning", "🔧 Data Processing", 
             "🤖 Model Training", "📈 Evaluasi Model", "🎯 Rekomendasi", "📊 Dashboard Analitik"]
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Data source info
        if st.session_state.data_source:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 📂 Sumber Data")
            if st.session_state.data_source == "uploaded":
                st.success("✅ Dataset dari file upload")
            else:
                st.info("📊 Dataset default")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Logout button
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.session_state.user_role = None
            st.session_state.uploaded_file = None
            st.session_state.data_source = "default"
            st.rerun()
    
    # Main content based on menu
    if menu == "📁 Upload Dataset":
        upload_dataset_section()
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

def upload_dataset_section():
    """Section untuk upload dataset"""
    st.header("📁 Upload Dataset CSV")
    
    st.markdown("""
    <div class="upload-section">
        <h3>📤 Upload Dataset Baru</h3>
        <p>Upload file CSV dengan data mahasiswa untuk dianalisis.</p>
        <p><strong>Format yang diperlukan:</strong></p>
        <ul>
            <li>Kolom: <code>Nama</code>, <code>Studi_Jurusan</code>, <code>Semester</code>, <code>AI_Tools</code>, <code>Trust_Level</code>, <code>Usage_Intensity_Score</code></li>
            <li>File harus dalam format CSV</li>
            <li>Encoding: UTF-8</li>
            <li>Usage_Intensity_Score: skala 0-10 (10+ akan dikonversi ke 10)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload file section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Pilih file CSV",
            type=["csv"],
            help="Upload dataset dalam format CSV"
        )
    
    with col2:
        st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
        use_default = st.button("📊 Gunakan Dataset Default")
    
    # Handle file upload
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Validate the dataset
            is_valid, message = validate_dataset(df)
            
            if is_valid:
                st.session_state.df = df
                st.session_state.uploaded_file = uploaded_file.name
                st.session_state.data_source = "uploaded"
                
                # Clear previous processed data
                if 'df_clean' in st.session_state:
                    del st.session_state.df_clean
                if 'model' in st.session_state:
                    del st.session_state.model
                if 'results' in st.session_state:
                    del st.session_state.results
                
                st.success(f"✅ Dataset berhasil diupload: {uploaded_file.name}")
                
                # Show dataset info
                st.markdown("<div class='file-info'>", unsafe_allow_html=True)
                st.markdown(f"**📊 Info Dataset:**")
                st.markdown(f"- **Nama file:** {uploaded_file.name}")
                st.markdown(f"- **Jumlah data:** {len(df)} baris")
                st.markdown(f"- **Jumlah kolom:** {len(df.columns)}")
                st.markdown(f"- **Kolom:** {', '.join(df.columns.tolist())}")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Show data preview
                st.subheader("👀 Preview Data")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Download template button
                st.markdown("---")
                st.subheader("📥 Download Template")
                st.markdown("Jika perlu template dataset, download template berikut:")
                
                # Create template DataFrame
                template_data = {
                    'Nama': ['Mahasiswa 1', 'Mahasiswa 2', 'Mahasiswa 3'],
                    'Studi_Jurusan': ['Teknologi Informasi', 'Teknik Informatika', 'Farmasi'],
                    'Semester': [3, 5, 7],
                    'AI_Tools': ['ChatGPT', 'Gemini', 'Multiple'],
                    'Trust_Level': [4, 3, 5],
                    'Usage_Intensity_Score': [6, 8, 3]
                }
                template_df = pd.DataFrame(template_data)
                
                # Generate download link
                st.markdown(get_download_link(template_df, "template_dataset.csv", "📥 Download Template CSV"), unsafe_allow_html=True)
                
            else:
                st.error(f"❌ Dataset tidak valid: {message}")
                st.info("Pastikan dataset memiliki kolom yang diperlukan dan format yang benar.")
                
        except Exception as e:
            st.error(f"Error membaca file: {str(e)}")
    
    # Handle default dataset
    elif use_default:
        st.session_state.df = load_default_dataset()
        st.session_state.data_source = "default"
        st.session_state.uploaded_file = None
        
        # Clear previous processed data
        if 'df_clean' in st.session_state:
            del st.session_state.df_clean
        if 'model' in st.session_state:
            del st.session_state.model
        if 'results' in st.session_state:
            del st.session_state.results
        
        st.success("✅ Dataset default berhasil dimuat!")
        
        # Show dataset info
        df = st.session_state.df
        st.markdown("<div class='file-info'>", unsafe_allow_html=True)
        st.markdown(f"**📊 Info Dataset:**")
        st.markdown(f"- **Sumber:** Dataset Default")
        st.markdown(f"- **Jumlah data:** {len(df)} baris")
        st.markdown(f"- **Jumlah kolom:** {len(df.columns)}")
        st.markdown(f"- **Kolom:** {', '.join(df.columns.tolist())}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Show data preview
        st.subheader("👀 Preview Data")
        st.dataframe(df.head(10), use_container_width=True)
    
    # Show current dataset if exists
    elif st.session_state.df is not None:
        st.subheader("📋 Dataset Saat Ini")
        
        # Show dataset info
        df = st.session_state.df
        source = "Uploaded File" if st.session_state.data_source == "uploaded" else "Default"
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sumber Data", source)
        with col2:
            st.metric("Jumlah Data", len(df))
        with col3:
            st.metric("Jumlah Kolom", len(df.columns))
        
        # Data preview
        st.dataframe(df.head(10), use_container_width=True)
        
        # Option to clear dataset
        if st.button("🗑️ Hapus Dataset Saat Ini"):
            st.session_state.df = None
            st.session_state.df_clean = None
            st.session_state.model = None
            st.session_state.results = None
            st.session_state.uploaded_file = None
            st.session_state.data_source = "default"
            st.rerun()
    
    else:
        st.info("👆 Silakan upload file CSV atau gunakan dataset default untuk memulai analisis.")

def data_cleaning_section():
    """Data cleaning section"""
    st.header("🧹 Data Cleaning")
    
    if st.session_state.df is None:
        st.warning("Harap upload dataset terlebih dahulu di menu 'Upload Dataset'!")
        return
    
    df = st.session_state.df
    
    st.info("""
    **Proses Data Cleaning:**
    1. Menghapus data duplikat
    2. Mengonversi nilai '10+' menjadi 10
    3. Menangani missing values
    4. Validasi tipe data
    """)
    
    # Show data before cleaning
    st.subheader("📊 Data Sebelum Cleaning")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        duplicates = df.duplicated(subset=['Nama']).sum()
        st.metric("Duplikat Nama", duplicates)
    with col2:
        missing = df.isnull().sum().sum()
        st.metric("Missing Values", missing)
    with col3:
        invalid_scores = df['Usage_Intensity_Score'].apply(
            lambda x: isinstance(x, str) and not x.replace('10+', '10').isdigit()
        ).sum()
        st.metric("Skor Tidak Valid", invalid_scores)
    with col4:
        data_types_ok = all([
            pd.api.types.is_numeric_dtype(df['Semester']),
            pd.api.types.is_numeric_dtype(df['Trust_Level'])
        ])
        st.metric("Tipe Data OK", "✅" if data_types_ok else "❌")
    
    if st.button("🚀 Jalankan Data Cleaning", type="primary"):
        with st.spinner("Melakukan data cleaning..."):
            df_clean = clean_data(df)
            st.session_state.df_clean = df_clean
            st.success("✅ Data cleaning selesai!")
            
            # Show cleaning results
            st.subheader("📊 Hasil Data Cleaning")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("**Sebelum Cleaning:**")
                st.write(f"- Jumlah data: {len(df)}")
                st.write(f"- Data duplikat: {duplicates}")
                st.write(f"- Missing values: {missing}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='card success-card'>", unsafe_allow_html=True)
                st.markdown("**Setelah Cleaning:**")
                st.write(f"- Jumlah data: {len(df_clean)}")
                st.write(f"- Data duplikat: {df_clean.duplicated().sum()}")
                st.write(f"- Missing values: {df_clean.isnull().sum().sum()}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Show cleaned data
            st.subheader("📋 Data Setelah Cleaning")
            st.dataframe(df_clean, use_container_width=True)
            
            # Data quality metrics
            st.subheader("📈 Metrik Kualitas Data")
            
            # Calculate data quality score
            total_rows = len(df_clean)
            quality_metrics = {
                'Kompleteness': 1 - (df_clean.isnull().sum().sum() / (total_rows * len(df_clean.columns))),
                'Uniqueness': 1 - (df_clean.duplicated().sum() / total_rows),
                'Validity': 1 - (df_clean['Usage_Intensity_Score'].apply(
                    lambda x: not (0 <= float(x) <= 10)
                ).sum() / total_rows),
                'Consistency': 1 - (df_clean['Semester'].apply(
                    lambda x: not (1 <= x <= 8)
                ).sum() / total_rows)
            }
            
            # Display quality metrics
            quality_df = pd.DataFrame.from_dict(quality_metrics, orient='index', columns=['Score'])
            quality_df['Score'] = quality_df['Score'].apply(lambda x: f"{x:.1%}")
            
            st.dataframe(quality_df, use_container_width=True)
    
    elif st.session_state.df_clean is not None:
        st.success("✅ Data sudah dibersihkan sebelumnya!")
        
        df_clean = st.session_state.df_clean
        
        # Show cleaned data
        st.subheader("📋 Data Setelah Cleaning")
        st.dataframe(df_clean.head(10), use_container_width=True)
        
        # Option to re-run cleaning
        if st.button("🔄 Jalankan Ulang Data Cleaning"):
            st.session_state.df_clean = None
            st.rerun()

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
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("**Mapping Jurusan:**")
                jurusan_mapping = dict(zip(
                    st.session_state.encoders['Studi_Jurusan'].classes_,
                    range(len(st.session_state.encoders['Studi_Jurusan'].classes_))
                ))
                for key, value in jurusan_mapping.items():
                    st.write(f"{key} → {value}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("**Mapping AI Tools:**")
                tools_mapping = dict(zip(
                    st.session_state.encoders['AI_Tools'].classes_,
                    range(len(st.session_state.encoders['AI_Tools'].classes_))
                ))
                for key, value in tools_mapping.items():
                    st.write(f"{key} → {value}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Show processed data
            st.subheader("📋 Data Setelah Processing")
            st.dataframe(df_encoded.head(), use_container_width=True)
            
            # Show target distribution
            st.subheader("🎯 Distribusi Target (Usage Level)")
            level_counts = df_encoded['Usage_Level'].value_counts()
            
            # Create visualization
            fig = px.pie(values=level_counts.values, names=level_counts.index,
                        title="Distribusi Tingkat Penggunaan AI",
                        color=level_counts.index,
                        color_discrete_map={
                            'RENDAH': '#2ECC71',
                            'SEDANG': '#F39C12',
                            'TINGGI': '#E74C3C'
                        })
            st.plotly_chart(fig, use_container_width=True)
            
            # Show statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Level Rendah", level_counts.get('RENDAH', 0))
            with col2:
                st.metric("Level Sedang", level_counts.get('SEDANG', 0))
            with col3:
                st.metric("Level Tinggi", level_counts.get('TINGGI', 0))

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
                        title="Importance Fitur dalam Model",
                        labels={'x': 'Fitur', 'y': 'Importance'},
                        color=importances,
                        color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed feature importance
            importance_df = pd.DataFrame({
                'Fitur': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            st.dataframe(importance_df, use_container_width=True)

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
                    text_auto=True,
                    color_continuous_scale='Blues')
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
    
    # Calculate and display mismatch
    mismatch = (comparison_df['Actual'] != comparison_df['Predicted']).sum()
    match = len(comparison_df) - mismatch
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prediksi Sesuai", match)
    with col2:
        st.metric("Prediksi Tidak Sesuai", mismatch)

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
        
        with col2:
            # Export summary report
            if st.button("📊 Export Ringkasan"):
                summary_data = {
                    'Total_Mahasiswa': [summary['total_students']],
                    'Level_Rendah': [summary['counts']['RENDAH']],
                    'Level_Sedang': [summary['counts']['SEDANG']],
                    'Level_Tinggi': [summary['counts']['TINGGI']],
                    'Level_Dominan': [summary['dominant_class']],
                    'Tingkat_Risiko': [summary['risk_level']],
                    'Tanggal_Analisis': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                }
                
                summary_df = pd.DataFrame(summary_data)
                csv = summary_df.to_csv(index=False)
                
                st.download_button(
                    label="Download Ringkasan",
                    data=csv,
                    file_name="ringkasan_analisis_ai.csv",
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
                
                # Download personal report
                st.markdown("### 📥 Download Laporan Pribadi")
                
                # Create personal report
                personal_report = f"""
                LAPORAN ANALISIS PENGGUNAAN AI - {student['Nama']}
                Tanggal: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                
                PROFIL MAHASISWA:
                - Nama: {student['Nama']}
                - Jurusan: {student['Studi_Jurusan']}
                - Semester: {student['Semester']}
                - Tools AI yang Digunakan: {student['AI_Tools']}
                - Tingkat Kepercayaan AI: {student['Trust_Level']}/5
                - Skor Penggunaan AI: {student['Usage_Intensity_Score']}/10
                
                HASIL ANALISIS:
                - Klasifikasi: {classification}
                - Label: {recommendation['label']}
                - Level Monitoring: {recommendation['monitoring_level']}
                
                REKOMENDASI:
                {chr(10).join(['- ' + rec for rec in recommendation['details']])}
                
                TINDAKAN YANG DISARANKAN:
                {chr(10).join(['- ' + action for action in recommendation['actions']])}
                
                TIPS PENGGUNAAN AI YANG SEHAT:
                {chr(10).join(['- ' + tip for tip in tips])}
                """
                
                # Convert to bytes for download
                b64 = base64.b64encode(personal_report.encode()).decode()
                href = f'<a href="data:file/txt;base64,{b64}" download="laporan_ai_{student["Nama"].replace(" ", "_")}.txt" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; border-radius: 5px;">📄 Download Laporan Pribadi</a>'
                st.markdown(href, unsafe_allow_html=True)
                
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
