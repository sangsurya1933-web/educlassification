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
    .score-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .interpretation-table {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
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
        # Definisi skor dan interpretasi frekuensi
        self.score_interpretation = {
            1: {"frequency": "1-2 kali/minggu", "description": "Penggunaan sangat jarang"},
            2: {"frequency": "1-2 kali/minggu", "description": "Penggunaan jarang"},
            3: {"frequency": "3-5 kali/minggu", "description": "Penggunaan cukup sering"},
            4: {"frequency": "3-5 kali/minggu", "description": "Penggunaan cukup sering"},
            5: {"frequency": "6-10 kali/minggu", "description": "Penggunaan sering"},
            6: {"frequency": "6-10 kali/minggu", "description": "Penggunaan sering"},
            7: {"frequency": "11-20 kali/minggu", "description": "Penggunaan sangat sering"},
            8: {"frequency": "11-20 kali/minggu", "description": "Penggunaan sangat sering"},
            9: {"frequency": ">20 kali/minggu", "description": "Penggunaan ekstrem"},
            10: {"frequency": ">20 kali/minggu", "description": "Penggunaan ekstrem"}
        }
        
        self.rules = {
            'RENDAH': {
                'score_range': (1, 3),
                'label': 'AMAN',
                'color': '#2ECC71',
                'icon': '✅',
                'recommendations': [
                    'Penggunaan AI dalam batas wajar dan sehat',
                    'Frekuensi penggunaan sesuai kebutuhan belajar',
                    'AI digunakan sebagai alat bantu yang tepat',
                    'Pertahankan keseimbangan antara AI dan belajar mandiri'
                ],
                'actions': [
                    'Monitor berkala (bulanan)',
                    'Berikan apresiasi untuk penggunaan bijak',
                    'Dukung eksplorasi fitur AI yang bermanfaat',
                    'Tetap dorong pembelajaran konvensional'
                ]
            },
            'SEDANG': {
                'score_range': (4, 7),
                'label': 'PERLU PERHATIAN',
                'color': '#F39C12',
                'icon': '⚠️',
                'recommendations': [
                    'Penggunaan AI mulai intensif',
                    'Perlu evaluasi ketergantungan pada AI',
                    'Batasi penggunaan untuk tugas tertentu saja',
                    'Tingkatkan kesadaran etika penggunaan AI'
                ],
                'actions': [
                    'Konsultasi dengan dosen pembimbing',
                    'Buat jadwal penggunaan AI yang terstruktur',
                    'Ikuti workshop penggunaan AI yang bertanggung jawab',
                    'Laporan penggunaan mingguan',
                    'Evaluasi tugas mandiri tanpa bantuan AI'
                ]
            },
            'TINGGI': {
                'score_range': (8, 10),
                'label': 'BUTUH PENGAWASAN',
                'color': '#E74C3C',
                'icon': '🚨',
                'recommendations': [
                    'Penggunaan AI sangat intensif dan berpotensi mengganggu',
                    'Tingkat ketergantungan tinggi terhadap AI',
                    'Perlu intervensi segera untuk mencegah plagiarisme',
                    'Risiko rendahnya pemahaman konsep dasar'
                ],
                'actions': [
                    'Pembatasan akses tools AI di lingkungan kampus',
                    'Konsultasi wajib dengan pembimbing akademik',
                    'Monitoring ketat setiap minggu',
                    'Program pengurangan ketergantungan AI',
                    'Evaluasi ulang semua tugas yang menggunakan AI',
                    'Pendampingan khusus oleh tutor'
                ]
            }
        }
    
    def get_frequency_interpretation(self, score):
        """Dapatkan interpretasi frekuensi berdasarkan skor"""
        if isinstance(score, str) and score == '10+':
            score = 10
        score = int(float(score))
        
        if score in self.score_interpretation:
            return self.score_interpretation[score]
        elif score > 10:
            return {"frequency": ">20 kali/minggu", "description": "Penggunaan ekstrem"}
        else:
            return {"frequency": "Tidak terdefinisi", "description": "Skor tidak valid"}
    
    def classify_usage(self, score):
        """Klasifikasi berdasarkan skor penggunaan"""
        if isinstance(score, str) and score == '10+':
            score = 10
        
        try:
            score = float(score)
            if 1 <= score <= 3:
                return 'RENDAH'
            elif 4 <= score <= 7:
                return 'SEDANG'
            elif 8 <= score <= 10:
                return 'TINGGI'
            else:
                return 'TIDAK_VALID'
        except:
            return 'TIDAK_VALID'
    
    def get_recommendation(self, classification, student_data):
        """Dapatkan rekomendasi berdasarkan klasifikasi"""
        if classification == 'TIDAK_VALID':
            return {
                'classification': 'TIDAK_VALID',
                'label': 'DATA TIDAK VALID',
                'color': '#95A5A6',
                'icon': '❓',
                'student_name': student_data.get('Nama', ''),
                'jurusan': student_data.get('Studi_Jurusan', ''),
                'semester': student_data.get('Semester', ''),
                'ai_tool': student_data.get('AI_Tools', ''),
                'trust_level': student_data.get('Trust_Level', ''),
                'usage_score': student_data.get('Usage_Intensity_Score', ''),
                'main_recommendation': "Data tidak valid",
                'details': ['Skor penggunaan tidak valid', 'Perlu verifikasi ulang data'],
                'actions': ['Verifikasi data mahasiswa', 'Koreksi input skor'],
                'monitoring_level': 'TIDAK_TERDEFINISI',
                'frequency_interpretation': {'frequency': 'Tidak valid', 'description': 'Skor tidak valid'}
            }
        
        rule = self.rules[classification]
        
        # Get frequency interpretation
        score = student_data.get('Usage_Intensity_Score', '')
        freq_interpretation = self.get_frequency_interpretation(score)
        
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
            }[classification],
            'frequency_interpretation': freq_interpretation
        }
        
        return recommendation
    
    def generate_summary_report(self, classifications):
        """Hasilkan laporan ringkasan"""
        total = len(classifications)
        counts = {
            'RENDAH': classifications.count('RENDAH'),
            'SEDANG': classifications.count('SEDANG'),
            'TINGGI': classifications.count('TINGGI'),
            'TIDAK_VALID': classifications.count('TIDAK_VALID')
        }
        
        # Calculate percentages (excluding invalid)
        valid_total = total - counts['TIDAK_VALID']
        if valid_total > 0:
            percentages = {k: (v/valid_total)*100 for k, v in counts.items() if k != 'TIDAK_VALID'}
        else:
            percentages = {}
        
        dominant_class = max([k for k in counts.keys() if k != 'TIDAK_VALID'], key=lambda x: counts[x], default='TIDAK_VALID')
        
        return {
            'total_students': total,
            'valid_students': valid_total,
            'invalid_students': counts['TIDAK_VALID'],
            'counts': counts,
            'percentages': percentages,
            'dominant_class': dominant_class,
            'risk_level': 'TINGGI' if percentages.get('TINGGI', 0) > 30 else 'SEDANG' if percentages.get('TINGGI', 0) > 10 else 'RENDAH'
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
        df['Usage_Intensity_Score'] = df['Usage_Intensity_Score'].astype(str).apply(
            lambda x: 10 if x.strip() == '10+' else float(x)
        )
    except Exception as e:
        return False, f"Error konversi tipe data: {str(e)}"
    
    # Check for empty values
    if df[required_columns].isnull().any().any():
        return False, "Dataset mengandung nilai kosong (null)"
    
    # Validate score range
    invalid_scores = df[~df['Usage_Intensity_Score'].between(1, 10)]
    if not invalid_scores.empty:
        return False, f"Terdapat {len(invalid_scores)} data dengan skor di luar range 1-10"
    
    return True, "Dataset valid"

def clean_data(df):
    """Pembersihan data"""
    df_clean = df.copy()
    
    # Handle duplicate entries (keep first)
    df_clean = df_clean.drop_duplicates(subset=['Nama'], keep='first')
    
    # Convert Usage_Intensity_Score to numeric
    df_clean['Usage_Intensity_Score'] = df_clean['Usage_Intensity_Score'].apply(
        lambda x: 10 if str(x).strip() == '10+' else float(x)
    )
    
    # Handle missing values
    for col in ['Semester', 'Trust_Level', 'Usage_Intensity_Score']:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Validate score range
    df_clean = df_clean[df_clean['Usage_Intensity_Score'].between(1, 10)]
    
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

def show_score_interpretation_table():
    """Tampilkan tabel interpretasi skor"""
    st.markdown("""
    <div class="interpretation-table">
        <h4>📊 Interpretasi Usage_Intensity_Score (Self-Report)</h4>
        <p><em>Skor intensitas penggunaan AI berbasis laporan mandiri</em></p>
        <table style="width:100%; border-collapse: collapse; margin-top: 10px;">
            <tr style="background-color: #2E86C1; color: white;">
                <th style="padding: 10px; border: 1px solid #ddd;">Skor</th>
                <th style="padding: 10px; border: 1px solid #ddd;">Frekuensi Penggunaan</th>
                <th style="padding: 10px; border: 1px solid #ddd;">Interpretasi</th>
                <th style="padding: 10px; border: 1px solid #ddd;">Kategori</th>
            </tr>
            <tr style="background-color: #D5F4E6;">
                <td style="padding: 8px; border: 1px solid #ddd; text-align: center;"><strong>1-2</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">1-2 kali/minggu</td>
                <td style="padding: 8px; border: 1px solid #ddd;">Penggunaan sangat jarang</td>
                <td style="padding: 8px; border: 1px solid #ddd; text-align: center;"><span style="background-color: #2ECC71; color: white; padding: 3px 8px; border-radius: 3px;">RENDAH</span></td>
            </tr>
            <tr style="background-color: #D5F4E6;">
                <td style="padding: 8px; border: 1px solid #ddd; text-align: center;"><strong>3</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">3-5 kali/minggu</td>
                <td style="padding: 8px; border: 1px solid #ddd;">Penggunaan cukup sering</td>
                <td style="padding: 8px; border: 1px solid #ddd; text-align: center;"><span style="background-color: #2ECC71; color: white; padding: 3px 8px; border-radius: 3px;">RENDAH</span></td>
            </tr>
            <tr style="background-color: #FCF3CF;">
                <td style="padding: 8px; border: 1px solid #ddd; text-align: center;"><strong>4</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">3-5 kali/minggu</td>
                <td style="padding: 8px; border: 1px solid #ddd;">Penggunaan cukup sering</td>
                <td style="padding: 8px; border: 1px solid #ddd; text-align: center;"><span style="background-color: #F39C12; color: white; padding: 3px 8px; border-radius: 3px;">SEDANG</span></td>
            </tr>
            <tr style="background-color: #FCF3CF;">
                <td style="padding: 8px; border: 1px solid #ddd; text-align: center;"><strong>5-6</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">6-10 kali/minggu</td>
                <td style="padding: 8px; border: 1px solid #ddd;">Penggunaan sering</td>
                <td style="padding: 8px; border: 1px solid #ddd; text-align: center;"><span style="background-color: #F39C12; color: white; padding: 3px 8px; border-radius: 3px;">SEDANG</span></td>
            </tr>
            <tr style="background-color: #FCF3CF;">
                <td style="padding: 8px; border: 1px solid #ddd; text-align: center;"><strong>7</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">11-20 kali/minggu</td>
                <td style="padding: 8px; border: 1px solid #ddd;">Penggunaan sangat sering</td>
                <td style="padding: 8px; border: 1px solid #ddd; text-align: center;"><span style="background-color: #F39C12; color: white; padding: 3px 8px; border-radius: 3px;">SEDANG</span></td>
            </tr>
            <tr style="background-color: #FADBD8;">
                <td style="padding: 8px; border: 1px solid #ddd; text-align: center;"><strong>8</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">11-20 kali/minggu</td>
                <td style="padding: 8px; border: 1px solid #ddd;">Penggunaan sangat sering</td>
                <td style="padding: 8px; border: 1px solid #ddd; text-align: center;"><span style="background-color: #E74C3C; color: white; padding: 3px 8px; border-radius: 3px;">TINGGI</span></td>
            </tr>
            <tr style="background-color: #FADBD8;">
                <td style="padding: 8px; border: 1px solid #ddd; text-align: center;"><strong>9-10+</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">>20 kali/minggu</td>
                <td style="padding: 8px; border: 1px solid #ddd;">Penggunaan ekstrem</td>
                <td style="padding: 8px; border: 1px solid #ddd; text-align: center;"><span style="background-color: #E74C3C; color: white; padding: 3px 8px; border-radius: 3px;">TINGGI</span></td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

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
        
        **Interpretasi Skor Penggunaan (Self-Report):**
        - **Rendah (1-3):** 1-5 kali/minggu
        - **Sedang (4-7):** 6-20 kali/minggu  
        - **Tinggi (8-10+):** >20 kali/minggu
        
        **Format Dataset:**
        - File CSV dengan kolom: Nama, Studi_Jurusan, Semester, AI_Tools, Trust_Level, Usage_Intensity_Score
        - Usage_Intensity_Score: Skala 1-10 (10+ = 10)
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
            ["📁 Upload Dataset", "📋 Interpretasi Skor", "🧹 Data Cleaning", "🔧 Data Processing", 
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
        
        # Show score interpretation in sidebar
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🎯 Klasifikasi Skor")
        st.markdown("""
        **RENDAH (1-3):** Aman  
        **SEDANG (4-7):** Perlu Perhatian  
        **TINGGI (8-10+):** Butuh Pengawasan
        """)
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
    elif menu == "📋 Interpretasi Skor":
        score_interpretation_section()
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
            <li>Usage_Intensity_Score: skala 1-10 (10+ akan dikonversi ke 10)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Show score interpretation table
    show_score_interpretation_table()
    
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
                
                # Show score distribution
                st.markdown("**📈 Distribusi Skor:**")
                score_counts = df['Usage_Intensity_Score'].value_counts().sort_index()
                for score, count in score_counts.items():
                    st.markdown(f"  - Skor {score}: {count} mahasiswa")
                
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
                    'Usage_Intensity_Score': [3, 6, 9]
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
        
        # Show score distribution
        st.markdown("**📈 Distribusi Skor:**")
        score_counts = df['Usage_Intensity_Score'].value_counts().sort_index()
        for score, count in score_counts.items():
            st.markdown(f"  - Skor {score}: {count} mahasiswa")
        
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
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sumber Data", source)
        with col2:
            st.metric("Jumlah Data", len(df))
        with col3:
            st.metric("Rentang Skor", f"{df['Usage_Intensity_Score'].min()}-{df['Usage_Intensity_Score'].max()}")
        with col4:
            st.metric("Skor Rata-rata", f"{df['Usage_Intensity_Score'].mean():.1f}")
        
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

def score_interpretation_section():
    """Section untuk interpretasi skor"""
    st.header("📋 Interpretasi Usage_Intensity_Score")
    
    # Show detailed interpretation table
    show_score_interpretation_table()
    
    # Additional information
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📝 Keterangan")
    st.markdown("""
    **Usage_Intensity_Score** adalah skor self-report (laporan mandiri) yang diberikan mahasiswa 
    berdasarkan frekuensi penggunaan AI dalam aktivitas akademik mereka.
    
    **Metode Pengumpulan Data:**
    1. Mahasiswa mengisi kuesioner online
    2. Melaporkan frekuensi penggunaan AI per minggu
    3. Skor dikonversi ke skala 1-10
    4. Data divalidasi melalui wawancara acak
    
    **Validitas Skor:**
    - **Skor 1-3:** Penggunaan terbatas, hanya untuk tugas spesifik
    - **Skor 4-7:** Penggunaan rutin, sebagai alat bantu belajar
    - **Skor 8-10+:** Penggunaan intensif, berpotensi ketergantungan
    
    **Implikasi Akademik:**
    - Skor tinggi tidak selalu negatif, perlu konteks penggunaan
    - Penting untuk membedakan antara penggunaan produktif vs ketergantungan
    - Monitoring diperlukan untuk mencegah plagiarisme
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Interactive score interpretation
    st.subheader("🔍 Cek Interpretasi Skor")
    
    col1, col2 = st.columns(2)
    
    with col1:
        score_input = st.number_input(
            "Masukkan skor (1-10):",
            min_value=1,
            max_value=10,
            value=5,
            step=1
        )
    
    with col2:
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        check_button = st.button("📊 Interpretasi Skor")
    
    if check_button:
        kb = KnowledgeBaseSystem()
        interpretation = kb.get_frequency_interpretation(score_input)
        classification = kb.classify_usage(score_input)
        
        # Get color and icon based on classification
        colors = {
            'RENDAH': '#2ECC71',
            'SEDANG': '#F39C12',
            'TINGGI': '#E74C3C'
        }
        
        icons = {
            'RENDAH': '✅',
            'SEDANG': '⚠️',
            'TINGGI': '🚨'
        }
        
        color = colors.get(classification, '#95A5A6')
        icon = icons.get(classification, '❓')
        
        # Display interpretation
        st.markdown(f"""
        <div style="background-color: {color}; color: white; padding: 20px; border-radius: 10px; margin-top: 20px;">
            <h3>{icon} Skor {score_input}: {classification}</h3>
            <p><strong>Frekuensi:</strong> {interpretation['frequency']}</p>
            <p><strong>Deskripsi:</strong> {interpretation['description']}</p>
            <p><strong>Rekomendasi Umum:</strong> {kb.rules.get(classification, {}).get('label', 'Data tidak valid')}</p>
        </div>
        """, unsafe_allow_html=True)

def data_cleaning_section():
    """Data cleaning section"""
    st.header("🧹 Data Cleaning")
    
    if st.session_state.df is None:
        st.warning("Harap upload dataset terlebih dahulu di menu 'Upload Dataset'!")
        return
    
    df = st.session_state.df
    
    # Show score interpretation info
    st.info("""
    **Proses Data Cleaning untuk Usage_Intensity_Score:**
    1. Konversi nilai '10+' menjadi 10
    2. Validasi skor dalam range 1-10
    3. Penanganan data outlier
    4. Normalisasi data untuk analisis
    """)
    
    # Show data before cleaning
    st.subheader("📊 Data Sebelum Cleaning")
    
    # Score distribution visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram of scores
    scores = df['Usage_Intensity_Score'].apply(lambda x: 10 if str(x) == '10+' else float(x))
    ax1.hist(scores, bins=10, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.set_title('Distribusi Skor Penggunaan AI')
    ax1.set_xlabel('Skor (1-10)')
    ax1.set_ylabel('Frekuensi')
    ax1.set_xticks(range(1, 11))
    
    # Box plot for outlier detection
    ax2.boxplot(scores, vert=False)
    ax2.set_title('Deteksi Outlier Skor')
    ax2.set_xlabel('Skor')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Statistics before cleaning
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        duplicates = df.duplicated(subset=['Nama']).sum()
        st.metric("Duplikat Nama", duplicates)
    with col2:
        missing = df.isnull().sum().sum()
        st.metric("Missing Values", missing)
    with col3:
        invalid_scores = (~df['Usage_Intensity_Score'].apply(
            lambda x: 1 <= float(10 if str(x) == '10+' else x) <= 10
        )).sum()
        st.metric("Skor Tidak Valid", invalid_scores)
    with col4:
        avg_score = scores.mean()
        st.metric("Skor Rata-rata", f"{avg_score:.2f}")
    
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
                st.write(f"- Skor tidak valid: {invalid_scores}")
                st.write(f"- Skor rata-rata: {avg_score:.2f}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='card success-card'>", unsafe_allow_html=True)
                st.markdown("**Setelah Cleaning:**")
                st.write(f"- Jumlah data: {len(df_clean)}")
                st.write(f"- Data duplikat: {df_clean.duplicated().sum()}")
                st.write(f"- Missing values: {df_clean.isnull().sum().sum()}")
                st.write(f"- Skor tidak valid: 0")
                st.write(f"- Skor rata-rata: {df_clean['Usage_Intensity_Score'].mean():.2f}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Show cleaned data with classification
            st.subheader("📋 Data Setelah Cleaning dengan Klasifikasi")
            
            # Add classification to cleaned data
            kb = KnowledgeBaseSystem()
            df_display = df_clean.copy()
            df_display['Klasifikasi'] = df_display['Usage_Intensity_Score'].apply(kb.classify_usage)
            df_display['Frekuensi'] = df_display['Usage_Intensity_Score'].apply(
                lambda x: kb.get_frequency_interpretation(x)['frequency']
            )
            
            # Color code the classification
            def color_classification(val):
                if val == 'RENDAH':
                    return 'background-color: #D5F4E6'
                elif val == 'SEDANG':
                    return 'background-color: #FCF3CF'
                elif val == 'TINGGI':
                    return 'background-color: #FADBD8'
                else:
                    return ''
            
            styled_df = df_display.style.applymap(color_classification, subset=['Klasifikasi'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Data quality metrics
            st.subheader("📈 Metrik Kualitas Data")
            
            # Calculate data quality score
            total_rows = len(df_clean)
            quality_metrics = {
                'Kompleteness': 1 - (df_clean.isnull().sum().sum() / (total_rows * len(df_clean.columns))),
                'Uniqueness': 1 - (df_clean.duplicated().sum() / total_rows),
                'Validity': 1 - (df_clean['Usage_Intensity_Score'].apply(
                    lambda x: not (1 <= float(x) <= 10)
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
        
        # Add classification to cleaned data
        kb = KnowledgeBaseSystem()
        df_display = df_clean.copy()
        df_display['Klasifikasi'] = df_display['Usage_Intensity_Score'].apply(kb.classify_usage)
        df_display['Frekuensi'] = df_display['Usage_Intensity_Score'].apply(
            lambda x: kb.get_frequency_interpretation(x)['frequency']
        )
        
        st.dataframe(df_display.head(10), use_container_width=True)
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Data", len(df_clean))
        with col2:
            avg_score = df_clean['Usage_Intensity_Score'].mean()
            st.metric("Skor Rata-rata", f"{avg_score:.2f}")
        with col3:
            high_users = (df_clean['Usage_Intensity_Score'] >= 8).sum()
            st.metric("Pengguna Tinggi", high_users)
        
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
    
    **Target Variable:** Klasifikasi berdasarkan Usage_Intensity_Score
    - RENDAH (1-3): 1-5 kali/minggu
    - SEDANG (4-7): 6-20 kali/minggu
    - TINGGI (8-10+): >20 kali/minggu
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
            
            # Add frequency interpretation
            kb = KnowledgeBaseSystem()
            df_display = df_encoded.copy()
            df_display['Frekuensi'] = df_display['Usage_Intensity_Score'].apply(
                lambda x: kb.get_frequency_interpretation(x)['frequency']
            )
            
            st.dataframe(df_display[['Nama', 'Studi_Jurusan', 'Semester', 'AI_Tools', 
                                    'Trust_Level', 'Usage_Intensity_Score', 'Frekuensi', 
                                    'Usage_Level']].head(), use_container_width=True)
            
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
            
            # Show statistics with frequency interpretation
            st.subheader("📈 Statistik Klasifikasi")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                rendah_count = level_counts.get('RENDAH', 0)
                st.metric("Level Rendah", rendah_count)
                if rendah_count > 0:
                    st.caption("1-5 kali/minggu")
            
            with col2:
                sedang_count = level_counts.get('SEDANG', 0)
                st.metric("Level Sedang", sedang_count)
                if sedang_count > 0:
                    st.caption("6-20 kali/minggu")
            
            with col3:
                tinggi_count = level_counts.get('TINGGI', 0)
                st.metric("Level Tinggi", tinggi_count)
                if tinggi_count > 0:
                    st.caption(">20 kali/minggu")
            
            # Show feature correlation
            st.subheader("📊 Korelasi Fitur dengan Skor Penggunaan")
            
            # Calculate correlation
            correlation_df = df_encoded[['Semester', 'Trust_Level', 'Usage_Intensity_Score']].corr()
            
            fig = px.imshow(correlation_df,
                           labels=dict(x="Variabel", y="Variabel", color="Korelasi"),
                           x=['Semester', 'Trust Level', 'Usage Score'],
                           y=['Semester', 'Trust Level', 'Usage Score'],
                           title="Matriks Korelasi",
                           color_continuous_scale='RdBu',
                           zmin=-1, zmax=1,
                           text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

def model_training():
    """Model training section"""
    st.header("🤖 Model Training - Random Forest")
    
    if 'processed_data' not in st.session_state:
        st.warning("Harap proses data terlebih dahulu di menu Data Processing!")
        return
    
    st.info("""
    **Algoritma Random Forest untuk Klasifikasi Penggunaan AI:**
    - Ensemble learning method untuk klasifikasi 3 level
    - Multiple decision trees untuk meningkatkan akurasi
    - Bagging technique untuk mengurangi overfitting
    - Robust terhadap data noise dan outlier
    
    **Target Klasifikasi:** 
    - RENDAH: Skor 1-3 (1-5 kali/minggu)
    - SEDANG: Skor 4-7 (6-20 kali/minggu) 
    - TINGGI: Skor 8-10+ (>20 kali/minggu)
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
                'Importance': importances,
                'Interpretasi': [
                    'Semester studi',
                    'Tingkat kepercayaan terhadap AI',
                    'Program studi/jurusan',
                    'Jenis tools AI yang digunakan'
                ]
            }).sort_values('Importance', ascending=False)
            
            st.dataframe(importance_df, use_container_width=True)
            
            # Show prediction distribution
            st.subheader("📈 Distribusi Prediksi")
            
            # Decode predictions
            le_target = st.session_state.encoders['Usage_Level']
            y_pred_decoded = le_target.inverse_transform(y_pred)
            
            pred_counts = pd.Series(y_pred_decoded).value_counts()
            
            fig = px.bar(x=pred_counts.index, y=pred_counts.values,
                        title="Distribusi Hasil Prediksi",
                        labels={'x': 'Klasifikasi', 'y': 'Jumlah'},
                        color=pred_counts.index,
                        color_discrete_map={
                            'RENDAH': '#2ECC71',
                            'SEDANG': '#F39C12',
                            'TINGGI': '#E74C3C'
                        })
            st.plotly_chart(fig, use_container_width=True)

def model_evaluation():
    """Model evaluation section"""
    st.header("📈 Evaluasi Model")
    
    if 'model' not in st.session_state:
        st.warning("Harap train model terlebih dahulu!")
        return
    
    model_data = st.session_state.model
    
    st.info("""
    **Metrik Evaluasi untuk Klasifikasi Multi-Kelas:**
    - **Accuracy:** Proporsi prediksi benar dari total prediksi
    - **Precision:** Proporsi prediksi positif yang benar (per kelas)
    - **Recall:** Proporsi data positif yang terdeteksi (per kelas)
    - **F1-Score:** Rata-rata harmonik precision dan recall (per kelas)
    - **Weighted Avg:** Rata-rata terbobot berdasarkan support tiap kelas
    """)
    
    # Calculate metrics
    accuracy = accuracy_score(model_data['y_test'], model_data['y_pred'])
    report = classification_report(model_data['y_test'], model_data['y_pred'], 
                                  target_names=['RENDAH', 'SEDANG', 'TINGGI'],
                                  output_dict=True)
    
    # Display metrics
    st.subheader("📊 Performa Model")
    
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
    st.subheader("📋 Classification Report Detail")
    
    # Create a styled dataframe
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.round(2)
    
    # Color code the metrics
    def color_metrics(val):
        if isinstance(val, (int, float)):
            if val >= 0.9:
                return 'background-color: #D5F4E6'
            elif val >= 0.7:
                return 'background-color: #FCF3CF'
            else:
                return 'background-color: #FADBD8'
        return ''
    
    styled_report = report_df.style.applymap(color_metrics, subset=pd.IndexSlice[:, ['precision', 'recall', 'f1-score']])
    st.dataframe(styled_report, use_container_width=True)
    
    # Confusion matrix
    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(model_data['y_test'], model_data['y_pred'])
    
    fig = px.imshow(cm,
                    labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                    x=['RENDAH', 'SEDANG', 'TINGGI'],
                    y=['RENDAH', 'SEDANG', 'TINGGI'],
                    title="Confusion Matrix - Perbandingan Prediksi vs Aktual",
                    text_auto=True,
                    color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed confusion matrix analysis
    st.subheader("🔍 Analisis Confusion Matrix")
    
    # Decode predictions
    le_target = st.session_state.encoders['Usage_Level']
    y_test_decoded = le_target.inverse_transform(model_data['y_test'])
    y_pred_decoded = le_target.inverse_transform(model_data['y_pred'])
    
    comparison_df = pd.DataFrame({
        'Aktual': y_test_decoded,
        'Prediksi': y_pred_decoded
    })
    
    # Calculate accuracy per class
    class_accuracy = {}
    for class_name in ['RENDAH', 'SEDANG', 'TINGGI']:
        mask = comparison_df['Aktual'] == class_name
        if mask.sum() > 0:
            class_acc = (comparison_df.loc[mask, 'Aktual'] == comparison_df.loc[mask, 'Prediksi']).mean()
            class_accuracy[class_name] = class_acc
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Akurasi RENDAH", f"{class_accuracy.get('RENDAH', 0):.2%}")
    with col2:
        st.metric("Akurasi SEDANG", f"{class_accuracy.get('SEDANG', 0):.2%}")
    with col3:
        st.metric("Akurasi TINGGI", f"{class_accuracy.get('TINGGI', 0):.2%}")
    
    # Show misclassifications
    st.subheader("📝 Contoh Kasus Salah Klasifikasi")
    
    misclassified = comparison_df[comparison_df['Aktual'] != comparison_df['Prediksi']]
    
    if not misclassified.empty:
        st.dataframe(misclassified.head(10), use_container_width=True)
        
        # Analyze common misclassification patterns
        st.write("**Pola Salah Klasifikasi yang Sering Terjadi:**")
        misclass_patterns = misclassified.groupby(['Aktual', 'Prediksi']).size().reset_index(name='Jumlah')
        misclass_patterns = misclass_patterns.sort_values('Jumlah', ascending=False)
        
        for _, row in misclass_patterns.iterrows():
            st.write(f"- {row['Aktual']} → {row['Prediksi']}: {row['Jumlah']} kasus")
    else:
        st.success("🎉 Semua prediksi sesuai dengan data aktual!")

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
        
        # Display summary with detailed statistics
        st.subheader("📊 Ringkasan Analisis")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Mahasiswa", summary['total_students'])
        with col2:
            st.metric("Data Valid", summary['valid_students'])
        with col3:
            st.metric("Level Dominan", summary['dominant_class'])
        with col4:
            st.metric("Tingkat Risiko", summary['risk_level'])
        
        # Display recommendations by category
        st.subheader("📋 Detail Rekomendasi per Kategori")
        
        tabs = st.tabs(["🔴 TINGGI", "🟡 SEDANG", "🟢 RENDAH", "❓ TIDAK VALID"])
        
        categories = ['TINGGI', 'SEDANG', 'RENDAH', 'TIDAK_VALID']
        for tab, category in zip(tabs, categories):
            with tab:
                category_students = [r for r in results['recommendations'] if r['classification'] == category]
                
                if category_students:
                    st.markdown(f"### {len(category_students)} Mahasiswa dengan Level {category}")
                    
                    for student_rec in category_students:
                        color_class = {
                            'TINGGI': 'danger-card',
                            'SEDANG': 'warning-card',
                            'RENDAH': 'success-card',
                            'TIDAK_VALID': 'card'
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
                            if 'frequency_interpretation' in student_rec:
                                st.markdown(f"**Frekuensi:** {student_rec['frequency_interpretation']['frequency']}")
                        
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
        
        col1, col2, col3 = st.columns(3)
        
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
                        'Frekuensi': rec.get('frequency_interpretation', {}).get('frequency', ''),
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
                    'Data_Valid': [summary['valid_students']],
                    'Data_Tidak_Valid': [summary['invalid_students']],
                    'Level_Rendah': [summary['counts'].get('RENDAH', 0)],
                    'Level_Sedang': [summary['counts'].get('SEDANG', 0)],
                    'Level_Tinggi': [summary['counts'].get('TINGGI', 0)],
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
        
        with col3:
            # Export detailed report
            if st.button("📋 Export Laporan Detail"):
                # Create detailed report
                report_lines = []
                report_lines.append("LAPORAN ANALISIS PENGGUNAAN AI MAHASISWA")
                report_lines.append("=" * 50)
                report_lines.append(f"Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                report_lines.append(f"Total Mahasiswa: {summary['total_students']}")
                report_lines.append(f"Data Valid: {summary['valid_students']}")
                report_lines.append(f"Level Dominan: {summary['dominant_class']}")
                report_lines.append(f"Tingkat Risiko: {summary['risk_level']}")
                report_lines.append("")
                report_lines.append("DISTRIBUSI KLASIFIKASI:")
                for level in ['RENDAH', 'SEDANG', 'TINGGI']:
                    count = summary['counts'].get(level, 0)
                    percentage = summary['percentages'].get(level, 0)
                    report_lines.append(f"- {level}: {count} mahasiswa ({percentage:.1f}%)")
                report_lines.append("")
                report_lines.append("REKOMENDASI PER KATEGORI:")
                
                for level in ['TINGGI', 'SEDANG', 'RENDAH']:
                    level_students = [r for r in results['recommendations'] if r['classification'] == level]
                    if level_students:
                        report_lines.append(f"\n{level} ({len(level_students)} mahasiswa):")
                        for student in level_students[:5]:  # Limit to 5 per category
                            report_lines.append(f"  • {student['student_name']} - Skor: {student['usage_score']}")
                
                report_text = "\n".join(report_lines)
                
                # Convert to bytes for download
                b64 = base64.b64encode(report_text.encode()).decode()
                href = f'<a href="data:file/txt;base64,{b64}" download="laporan_analisis_ai.txt" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; border-radius: 5px;">📄 Download Laporan Detail</a>'
                st.markdown(href, unsafe_allow_html=True)

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
        avg_usage = df['Usage_Intensity_Score'].mean()
        st.metric("Rata-rata Penggunaan", f"{avg_usage:.2f}")
    
    with col2:
        avg_trust = df['Trust_Level'].mean()
        st.metric("Rata-rata Trust Level", f"{avg_trust:.2f}")
    
    with col3:
        high_users = (df['Usage_Intensity_Score'] >= 8).sum()
        st.metric("Pengguna Intensif", high_users)
    
    with col4:
        low_users = (df['Usage_Intensity_Score'] <= 3).sum()
        st.metric("Pengguna Rendah", low_users)
    
    # Interactive visualizations
    st.subheader("📊 Visualisasi Interaktif")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Distribusi Skor", "Analisis Frekuensi", "Tools AI", "Korelasi"])
    
    with tab1:
        # Distribution visualization with interpretation
        fig = px.histogram(df, x='Usage_Intensity_Score',
                          title='Distribusi Skor Penggunaan AI',
                          nbins=10,
                          color_discrete_sequence=['#2E86C1'],
                          labels={'Usage_Intensity_Score': 'Skor Penggunaan (1-10)'})
        
        # Add vertical lines for classification boundaries
        fig.add_vline(x=3.5, line_dash="dash", line_color="green", 
                     annotation_text="RENDAH-SEDANG", annotation_position="top")
        fig.add_vline(x=7.5, line_dash="dash", line_color="red",
                     annotation_text="SEDANG-TINGGI", annotation_position="top")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show interpretation of distribution
        st.markdown("**Interpretasi Distribusi:**")
        kb = KnowledgeBaseSystem()
        for score_range in [(1, 3), (4, 7), (8, 10)]:
            count = ((df['Usage_Intensity_Score'] >= score_range[0]) & 
                    (df['Usage_Intensity_Score'] <= score_range[1])).sum()
            percentage = (count / len(df)) * 100
            classification = kb.classify_usage(score_range[0])
            freq = kb.get_frequency_interpretation(score_range[0])['frequency']
            
            st.markdown(f"- **{classification}** (Skor {score_range[0]}-{score_range[1]}): {count} mahasiswa ({percentage:.1f}%) - {freq}")
    
    with tab2:
        # Frequency analysis
        kb = KnowledgeBaseSystem()
        df_freq = df.copy()
        df_freq['Frekuensi'] = df_freq['Usage_Intensity_Score'].apply(
            lambda x: kb.get_frequency_interpretation(x)['frequency']
        )
        
        freq_counts = df_freq['Frekuensi'].value_counts().reset_index()
        freq_counts.columns = ['Frekuensi', 'Jumlah']
        
        fig = px.bar(freq_counts, x='Frekuensi', y='Jumlah',
                    title='Distribusi Frekuensi Penggunaan AI',
                    color='Jumlah',
                    color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed frequency statistics
        st.markdown("**Statistik Frekuensi:**")
        for freq in freq_counts['Frekuensi'].unique():
            count = freq_counts[freq_counts['Frekuensi'] == freq]['Jumlah'].values[0]
            percentage = (count / len(df)) * 100
            st.markdown(f"- **{freq}:** {count} mahasiswa ({percentage:.1f}%)")
    
    with tab3:
        # AI Tools usage analysis
        tools_usage = df.groupby('AI_Tools').agg({
            'Usage_Intensity_Score': ['mean', 'count'],
            'Trust_Level': 'mean'
        }).reset_index()
        
        tools_usage.columns = ['AI_Tools', 'Avg_Usage', 'Count', 'Avg_Trust']
        
        # Create bubble chart
        fig = px.scatter(tools_usage, x='Avg_Usage', y='Avg_Trust',
                        size='Count', color='AI_Tools',
                        title='Tools AI vs Penggunaan & Kepercayaan',
                        hover_name='AI_Tools',
                        labels={'Avg_Usage': 'Rata-rata Skor Penggunaan',
                               'Avg_Trust': 'Rata-rata Trust Level'})
        
        # Add reference lines
        fig.add_hline(y=3, line_dash="dash", line_color="orange")
        fig.add_vline(x=7, line_dash="dash", line_color="red")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show tools statistics
        st.markdown("**Statistik Tools AI:**")
        for _, row in tools_usage.iterrows():
            st.markdown(f"- **{row['AI_Tools']}:** {row['Count']} pengguna, Skor rata-rata: {row['Avg_Usage']:.1f}, Trust: {row['Avg_Trust']:.1f}")
    
    with tab4:
        # Correlation matrix with additional variables
        numeric_df = df.copy()
        
        # Add classification for color coding
        kb = KnowledgeBaseSystem()
        numeric_df['Classification'] = numeric_df['Usage_Intensity_Score'].apply(kb.classify_usage)
        
        # Create scatter matrix
        fig = px.scatter_matrix(numeric_df,
                               dimensions=['Semester', 'Trust_Level', 'Usage_Intensity_Score'],
                               color='Classification',
                               title='Matriks Korelasi Antar Variabel',
                               color_discrete_map={
                                   'RENDAH': '#2ECC71',
                                   'SEDANG': '#F39C12',
                                   'TINGGI': '#E74C3C'
                               })
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate and display correlation coefficients
        correlation = numeric_df[['Semester', 'Trust_Level', 'Usage_Intensity_Score']].corr()
        
        st.markdown("**Koefisien Korelasi:**")
        st.dataframe(correlation.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1), 
                    use_container_width=True)
    
    # Detailed analysis
    if st.session_state.results:
        st.subheader("🎯 Analisis Rekomendasi")
        
        results = st.session_state.results
        summary = results['summary']
        
        # Pie chart of classifications
        fig = px.pie(values=[summary['counts'].get(k, 0) for k in ['RENDAH', 'SEDANG', 'TINGGI']],
                    names=['RENDAH', 'SEDANG', 'TINGGI'],
                    title='Distribusi Klasifikasi Penggunaan AI',
                    color=['RENDAH', 'SEDANG', 'TINGGI'],
                    color_discrete_map={
                        'RENDAH': '#2ECC71',
                        'SEDANG': '#F39C12',
                        'TINGGI': '#E74C3C'
                    })
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk assessment
        st.subheader("⚠️ Assessment Risiko")
        
        risk_level = summary['risk_level']
        risk_color = {
            'RENDAH': '#2ECC71',
            'SEDANG': '#F39C12',
            'TINGGI': '#E74C3C'
        }.get(risk_level, '#95A5A6')
        
        risk_icon = {
            'RENDAH': '✅',
            'SEDANG': '⚠️',
            'TINGGI': '🚨'
        }.get(risk_level, '❓')
        
        st.markdown(f"""
        <div style="background-color: {risk_color}; color: white; padding: 20px; border-radius: 10px; text-align: center;">
            <h2>{risk_icon} TINGKAT RISIKO: {risk_level}</h2>
            <p>Berdasarkan analisis {summary['valid_students']} mahasiswa</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendations based on risk level
        st.markdown("**Rekomendasi Institusional:**")
        
        if risk_level == 'TINGGI':
            st.markdown("""
            - **Segera lakukan intervensi institusional**
            - **Buat kebijakan penggunaan AI di kampus**
            - **Lakukan workshop etika penggunaan AI**
            - **Monitor penggunaan AI di laboratorium komputer**
            - **Sediakan layanan konseling teknologi**
            """)
        elif risk_level == 'SEDANG':
            st.markdown("""
            - **Tingkatkan awareness penggunaan AI yang sehat**
            - **Integrasikan etika AI dalam kurikulum**
            - **Buat panduan penggunaan AI untuk tugas akademik**
            - **Lakukan survey berkala tentang penggunaan AI**
            """)
        else:
            st.markdown("""
            - **Pertahankan kondisi saat ini**
            - **Lakukan monitoring rutin**
            - **Siapkan mekanisme responsif untuk perubahan trend**
            - **Dokumentasikan best practices**
            """)

def mahasiswa_dashboard():
    """Dashboard untuk Mahasiswa"""
    st.markdown(f"<h1 class='main-header'>👨‍🎓 Dashboard Mahasiswa</h1>", unsafe_allow_html=True)
    
    student_name = st.session_state.get('student_name', '')
    
    if student_name:
        st.markdown(f"<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"### 👋 Selamat datang, **{student_name}**!")
        st.markdown("Di sini Anda dapat melihat hasil analisis penggunaan AI Anda berdasarkan self-report.")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Show score interpretation table
        show_score_interpretation_table()
        
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
                    if 'frequency_interpretation' in recommendation:
                        st.markdown(f"**Frekuensi:** {recommendation['frequency_interpretation']['frequency']}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    # Display classification with appropriate color
                    color_class = {
                        'RENDAH': 'success-card',
                        'SEDANG': 'warning-card',
                        'TINGGI': 'danger-card',
                        'TIDAK_VALID': 'card'
                    }[classification]
                    
                    st.markdown(f"<div class='card {color_class}'>", unsafe_allow_html=True)
                    st.markdown(f"### 🎯 Hasil Analisis")
                    st.markdown(f"**{recommendation['icon']} Klasifikasi:** {classification}")
                    st.markdown(f"**🏷️ Label:** {recommendation['label']}")
                    st.markdown(f"**📊 Level Monitoring:** {recommendation['monitoring_level']}")
                    
                    # Show score interpretation
                    if 'frequency_interpretation' in recommendation:
                        st.markdown("---")
                        st.markdown(f"**📈 Interpretasi Skor {score}:**")
                        st.markdown(f"- **Frekuensi:** {recommendation['frequency_interpretation']['frequency']}")
                        st.markdown(f"- **Deskripsi:** {recommendation['frequency_interpretation']['description']}")
                    
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
                
                # Additional tips based on classification
                st.markdown("**💡 Tips Penggunaan AI yang Sehat:**")
                
                if classification == 'TINGGI':
                    tips = [
                        "Evaluasi ketergantungan Anda terhadap AI",
                        "Coba selesaikan tugas tanpa bantuan AI minimal 1x/minggu",
                        "Diskusikan dengan dosen tentang penggunaan AI yang tepat",
                        "Gunakan AI hanya untuk brainstorming, bukan untuk mengerjakan seluruh tugas",
                        "Catat penggunaan AI Anda untuk evaluasi mandiri"
                    ]
                elif classification == 'SEDANG':
                    tips = [
                        "Pertahankan keseimbangan antara AI dan belajar mandiri",
                        "Gunakan AI sebagai alat verifikasi, bukan sumber utama",
                        "Selalu cross-check informasi dari AI dengan sumber terpercaya",
                        "Tetap kembangkan kemampuan analisis dan pemecahan masalah Anda",
                        "Ikuti perkembangan etika penggunaan AI dalam akademik"
                    ]
                else:
                    tips = [
                        "Teruskan pola penggunaan AI yang sehat ini",
                        "Eksplorasi fitur AI yang dapat meningkatkan produktivitas",
                        "Bagikan tips penggunaan AI yang bijak dengan teman",
                        "Tetap waspada terhadap perkembangan teknologi AI",
                        "Pertahankan kemampuan belajar mandiri Anda"
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
                    title={'text': f"Skor Penggunaan AI: {classification}"},
                    gauge={
                        'axis': {'range': [1, 10], 'tickwidth': 1},
                        'bar': {'color': recommendation['color']},
                        'steps': [
                            {'range': [1, 3], 'color': '#2ECC71'},
                            {'range': [3, 7], 'color': '#F39C12'},
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
                
                # Comparison with peers
                st.markdown("### 📈 Perbandingan dengan Rekan")
                
                df_clean = st.session_state.df_clean
                
                # Calculate statistics
                same_jurusan = df_clean[df_clean['Studi_Jurusan'] == student['Studi_Jurusan']]
                same_semester = df_clean[df_clean['Semester'] == student['Semester']]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_all = df_clean['Usage_Intensity_Score'].mean()
                    st.metric("Rata-rata Semua", f"{avg_all:.1f}")
                with col2:
                    avg_jurusan = same_jurusan['Usage_Intensity_Score'].mean()
                    st.metric(f"Rata-rata {student['Studi_Jurusan']}", f"{avg_jurusan:.1f}")
                with col3:
                    avg_semester = same_semester['Usage_Intensity_Score'].mean()
                    st.metric(f"Rata-rata Semester {student['Semester']}", f"{avg_semester:.1f}")
                
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
                
                INTERPRETASI SKOR:
                - Klasifikasi: {classification}
                - Frekuensi Penggunaan: {recommendation.get('frequency_interpretation', {}).get('frequency', 'N/A')}
                - Deskripsi: {recommendation.get('frequency_interpretation', {}).get('description', 'N/A')}
                
                HASIL ANALISIS:
                - Label: {recommendation['label']}
                - Level Monitoring: {recommendation['monitoring_level']}
                
                REKOMENDASI:
                {chr(10).join(['- ' + rec for rec in recommendation['details']])}
                
                TINDAKAN YANG DISARANKAN:
                {chr(10).join(['- ' + action for action in recommendation['actions']])}
                
                TIPS PENGGUNAAN AI YANG SEHAT:
                {chr(10).join(['- ' + tip for tip in tips])}
                
                PERBANDINGAN DENGAN REKAN:
                - Rata-rata semua mahasiswa: {avg_all:.1f}
                - Rata-rata jurusan {student['Studi_Jurusan']}: {avg_jurusan:.1f}
                - Rata-rata semester {student['Semester']}: {avg_semester:.1f}
                
                CATATAN:
                Laporan ini berdasarkan data self-report. Untuk konsultasi lebih lanjut,
                silakan hubungi dosen pembimbing atau unit konseling kampus.
                """
                
                # Convert to bytes for download
                b64 = base64.b64encode(personal_report.encode()).decode()
                href = f'<a href="data:file/txt;base64,{b64}" download="laporan_ai_{student["Nama"].replace(" ", "_")}.txt" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; border-radius: 5px;">📄 Download Laporan Pribadi</a>'
                st.markdown(href, unsafe_allow_html=True)
                
            else:
                st.warning(f"Data tidak ditemukan untuk nama '{student_name}'.")
                st.info("""
                **Kemungkinan penyebab:**
                1. Nama tidak terdaftar dalam sistem
                2. Data Anda belum diproses oleh administrator
                3. Terdapat kesalahan penulisan nama
                
                Silakan hubungi administrator atau pastikan nama Anda sesuai dengan data akademik.
                """)
        else:
            st.info("🔄 Analisis sedang dipersiapkan oleh administrator. Silakan coba lagi nanti.")
    
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
