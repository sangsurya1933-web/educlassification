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
import csv


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
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

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
                'icon': '',
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
                'icon': '',
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
                'icon': '',
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
        try:
            score = int(float(score))
            if score in self.score_interpretation:
                return self.score_interpretation[score]
            elif score > 10:
                return {"frequency": ">20 kali/minggu", "description": "Penggunaan ekstrem"}
            else:
                return {"frequency": "Tidak terdefinisi", "description": "Skor tidak valid"}
        except:
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
                'icon': '',
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
        
        dominant_class = max([k for k in counts.keys() if k != 'TIDAK_VALID'], 
                            key=lambda x: counts[x], default='TIDAK_VALID')
        
        return {
            'total_students': total,
            'valid_students': valid_total,
            'invalid_students': counts['TIDAK_VALID'],
            'counts': counts,
            'percentages': percentages,
            'dominant_class': dominant_class,
            'risk_level': 'TINGGI' if percentages.get('TINGGI', 0) > 30 else 'SEDANG' if percentages.get('TINGGI', 0) > 10 else 'RENDAH'
        }

def read_csv_with_various_delimiters(uploaded_file):
    """Membaca file CSV dengan berbagai delimiter"""
    content = uploaded_file.getvalue().decode('utf-8')
    
    # Coba dengan titik koma (;) terlebih dahulu
    try:
        # Gunakan csv reader untuk mendeteksi delimiter
        dialect = csv.Sniffer().sniff(content[:1024])
        # Baca dengan pandas menggunakan delimiter yang terdeteksi
        if hasattr(dialect, 'delimiter'):
            df = pd.read_csv(uploaded_file, sep=dialect.delimiter)
            st.info(f"Delimiter terdeteksi: '{dialect.delimiter}'")
            return df
    except:
        pass
    
    # Coba delimiter yang umum
    delimiters = [';', ',', '\t', '|']
    
    for delimiter in delimiters:
        try:
            uploaded_file.seek(0)  # Reset file pointer
            df = pd.read_csv(uploaded_file, sep=delimiter)
            if len(df.columns) > 1:
                st.info(f"Menggunakan delimiter: '{delimiter}'")
                return df
        except:
            continue
    
    # Jika semua gagal, coba baca sebagai file dengan titik koma
    try:
        uploaded_file.seek(0)
        # Baca sebagai string dan proses manual
        lines = content.strip().split('\n')
        if lines:
            # Cari delimiter dengan menghitung kemunculan karakter
            first_line = lines[0]
            delimiter_counts = {
                ';': first_line.count(';'),
                ',': first_line.count(','),
                '\t': first_line.count('\t'),
                '|': first_line.count('|')
            }
            best_delimiter = max(delimiter_counts, key=delimiter_counts.get)
            if delimiter_counts[best_delimiter] > 0:
                df = pd.read_csv(io.StringIO(content), sep=best_delimiter)
                st.info(f"Menggunakan delimiter terbaik: '{best_delimiter}'")
                return df
    except Exception as e:
        st.error(f"Gagal membaca file: {str(e)}")
    
    return None

def load_default_dataset():
    """Load dataset default dari data yang diberikan"""
    try:
        # Membaca dari data yang disediakan dalam format asli
        data_text = """Nama;Studi_Jurusan;Semester;AI_Tools;Trust_Level;Usage_Intensity_Score
Althaf Rayyan Putra;Teknologi Informasi;7;Gemini;4;8
Ayesha Kinanti;Teknologi Informasi;3;Gemini;4;9
Salsabila Nurfadila;Teknik Informatika;1;Gemini;5;3
Anindya Safira;Teknik Informatika;5;Gemini;4;6
Iqbal Ramadhan;Farmasi;1;Gemini;5;10+
Muhammad Rizky Pratama;Teknologi Informasi;5;Gemini;4;4
Fikri Alfarizi;Teknologi Informasi;1;ChatGPT;4;7
Iqbal Ramadhan;Farmasi;1;Gemini;5;9
Citra Maharani;Keperawatan;5;Multiple;4;2
Iqbal Ramadhan;Farmasi;3;Gemini;5;9
Zidan Harits;Farmasi;7;Gemini;4;4
Rizky Kurniawan Putra;Teknik Informatika;5;ChatGPT;4;3
Raka Bimantara;Farmasi;3;ChatGPT;4;4
Zahra Alya Safitri;Teknik Informatika;3;Gemini;4;10+
Muhammad Naufal Haidar;Farmasi;1;Gemini;4;3
Citra Maharani;Keperawatan;1;Copilot;5;8
Ammar Zaky Firmansyah;Farmasi;3;ChatGPT;5;7
Ilham Nurhadi;Teknologi Informasi;1;Gemini;5;9
Muhammad Rizky Pratama;Teknologi Informasi;1;Multiple;5;3
Nayla Syakira;Teknologi Informasi;7;ChatGPT;4;8
Zidan Harits;Farmasi;3;Gemini;5;10+
Citra Maharani;Keperawatan;5;ChatGPT;2;8
Arfan Maulana;Teknik Informatika;7;ChatGPT;1;2
Nabila Khairunnisa;Teknologi Informasi;1;Gemini;5;9
Safira Azzahra Putri;Teknologi Informasi;1;Gemini;5;7
Farah Amalia;Teknologi Informasi;1;ChatGPT;5;6
Muhammad Reza Ananda;Teknologi Informasi;1;ChatGPT;5;4
Citra Maharani;Keperawatan;5;ChatGPT;1;3
Aulia Rahma;Teknik Informatika;3;Gemini;1;2
Yusuf Al Hakim;Teknik Informatika;5;ChatGPT;3;8
Salsabila Nurfadila;Teknik Informatika;1;Multiple;4;3
Aulia Rahma;Teknik Informatika;5;ChatGPT;4;10+
Ayesha Kinanti;Teknologi Informasi;1;Copilot;3;4
Damar Alif Prakoso;Teknologi Informasi;7;ChatGPT;4;3
Zidan Harits;Farmasi;5;ChatGPT;2;7
Ammar Zaky Firmansyah;Farmasi;1;ChatGPT;4;10+
Citra Maharani;Keperawatan;3;Gemini;4;8
Nabila Khairunnisa;Teknologi Informasi;1;Copilot;5;6
Farah Amalia;Teknologi Informasi;7;ChatGPT;4;6
Ammar Zaky Firmansyah;Farmasi;1;Gemini;4;2
Zidan Harits;Farmasi;7;ChatGPT;2;1
Farah Amalia;Teknologi Informasi;5;ChatGPT;3;5
Ahmad Fauzan Maulana;Farmasi;1;Copilot;4;7
Khansa Humaira Zahira;Teknik Informatika;1;ChatGPT;4;7
Salsabila Nurfadila;Teknik Informatika;1;Multiple;3;5"""
        
        # Parse data dengan pandas
        from io import StringIO
        df = pd.read_csv(StringIO(data_text), sep=';')
        
        # Konversi tipe data
        df['Semester'] = pd.to_numeric(df['Semester'], errors='coerce')
        df['Trust_Level'] = pd.to_numeric(df['Trust_Level'], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading default dataset: {str(e)}")
        # Fallback ke data minimal
        data = {
            'Nama': [
                'Althaf Rayyan Putra', 'Ayesha Kinanti', 'Salsabila Nurfadila', 
                'Anindya Safira', 'Iqbal Ramadhan', 'Muhammad Rizky Pratama'
            ],
            'Studi_Jurusan': [
                'Teknologi Informasi', 'Teknologi Informasi', 'Teknik Informatika',
                'Teknik Informatika', 'Farmasi', 'Teknologi Informasi'
            ],
            'Semester': [7, 3, 1, 5, 1, 5],
            'AI_Tools': ['Gemini', 'Gemini', 'Gemini', 'Gemini', 'Gemini', 'Gemini'],
            'Trust_Level': [4, 4, 5, 4, 5, 4],
            'Usage_Intensity_Score': [8, 9, 3, 6, 10, 4]
        }
        
        df = pd.DataFrame(data)
        return df

def validate_dataset(df):
    """Validasi dataset yang diupload"""
    required_columns = ['Nama', 'Studi_Jurusan', 'Semester', 'AI_Tools', 'Trust_Level', 'Usage_Intensity_Score']
    
    # Check if dataframe is empty
    if df is None or len(df) == 0:
        return False, "Dataset kosong"
    
    # Normalize column names: strip whitespace, lower case
    df.columns = [str(col).strip() for col in df.columns]
    
    # Case-insensitive column matching
    df_columns_lower = [col.lower().strip() for col in df.columns]
    required_columns_lower = [col.lower().strip() for col in required_columns]
    
    # Check for matching columns
    missing_columns = []
    column_mapping = {}
    
    for req_col, req_lower in zip(required_columns, required_columns_lower):
        found = False
        for df_col, df_lower in zip(df.columns, df_columns_lower):
            if req_lower == df_lower:
                column_mapping[req_col] = df_col
                found = True
                break
        
        if not found:
            # Try partial match
            for df_col, df_lower in zip(df.columns, df_columns_lower):
                if req_lower in df_lower or df_lower in req_lower:
                    column_mapping[req_col] = df_col
                    found = True
                    st.info(f"Kolom '{df_col}' dianggap sebagai '{req_col}'")
                    break
            
            if not found:
                missing_columns.append(req_col)
    
    if missing_columns:
        return False, f"Kolom yang hilang: {', '.join(missing_columns)}. Kolom yang ditemukan: {', '.join(df.columns)}"
    
    # Rename columns to standard names
    for req_col, df_col in column_mapping.items():
        if req_col != df_col:
            df.rename(columns={df_col: req_col}, inplace=True)
    
    # Check data types and convert
    try:
        # Convert Semester to numeric
        df['Semester'] = pd.to_numeric(df['Semester'], errors='coerce')
        
        # Convert Trust_Level to numeric
        df['Trust_Level'] = pd.to_numeric(df['Trust_Level'], errors='coerce')
        
        # Handle Usage_Intensity_Score - convert '10+' to 10
        def convert_score(x):
            if pd.isna(x):
                return np.nan
            x_str = str(x).strip()
            if x_str == '10+':
                return 10
            try:
                return float(x_str)
            except:
                return np.nan
        
        df['Usage_Intensity_Score'] = df['Usage_Intensity_Score'].apply(convert_score)
        
        # Check for missing values
        missing_info = {}
        for col in required_columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_info[col] = missing_count
        
        if missing_info:
            missing_msg = ", ".join([f"{col}: {count}" for col, count in missing_info.items()])
            st.warning(f"Terdapat nilai kosong: {missing_msg}")
        
        # Check score range
        valid_scores = df['Usage_Intensity_Score'].dropna()
        if len(valid_scores) > 0:
            min_score = valid_scores.min()
            max_score = valid_scores.max()
            if min_score < 1 or max_score > 10:
                st.warning(f"Skor berada di luar range normal (1-10): {min_score} - {max_score}")
        
        return True, "Dataset valid"
        
    except Exception as e:
        return False, f"Error konversi tipe data: {str(e)}"

def clean_data(df):
    """Pembersihan data"""
    if df is None or len(df) == 0:
        return df
    
    df_clean = df.copy()
    
    # Remove duplicate entries (keep first)
    df_clean = df_clean.drop_duplicates(subset=['Nama'], keep='first')
    
    # Convert Usage_Intensity_Score to numeric
    def convert_score(x):
        if pd.isna(x):
            return np.nan
        x_str = str(x).strip()
        if x_str == '10+':
            return 10
        try:
            return float(x_str)
        except:
            return np.nan
    
    df_clean['Usage_Intensity_Score'] = df_clean['Usage_Intensity_Score'].apply(convert_score)
    
    # Handle missing values
    for col in ['Semester', 'Trust_Level', 'Usage_Intensity_Score']:
        if col in df_clean.columns and df_clean[col].isnull().any():
            if col in ['Semester', 'Trust_Level']:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            elif col == 'Usage_Intensity_Score':
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    
    # Validate score range (1-10) and filter out invalid
    mask = df_clean['Usage_Intensity_Score'].between(1, 10) | df_clean['Usage_Intensity_Score'].isna()
    df_clean = df_clean[mask].copy()
    
    # Fill any remaining NaN in scores with median
    if df_clean['Usage_Intensity_Score'].isnull().any():
        df_clean['Usage_Intensity_Score'] = df_clean['Usage_Intensity_Score'].fillna(
            df_clean['Usage_Intensity_Score'].median()
        )
    
    # Reset index
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean

def encode_categorical_data(df):
    """Encode data kategorikal"""
    if df is None or len(df) == 0:
        return df, {}
    
    df_encoded = df.copy()
    encoders = {}
    
    # Encode Studi_Jurusan
    if 'Studi_Jurusan' in df_encoded.columns:
        le_jurusan = LabelEncoder()
        df_encoded['Studi_Jurusan_Encoded'] = le_jurusan.fit_transform(df_encoded['Studi_Jurusan'])
        encoders['Studi_Jurusan'] = le_jurusan
    
    # Encode AI_Tools
    if 'AI_Tools' in df_encoded.columns:
        le_tools = LabelEncoder()
        df_encoded['AI_Tools_Encoded'] = le_tools.fit_transform(df_encoded['AI_Tools'])
        encoders['AI_Tools'] = le_tools
    
    # Create target variable (klasifikasi penggunaan)
    if 'Usage_Intensity_Score' in df_encoded.columns:
        kb = KnowledgeBaseSystem()
        df_encoded['Usage_Level'] = df_encoded['Usage_Intensity_Score'].apply(kb.classify_usage)
        
        # Encode target variable
        le_target = LabelEncoder()
        df_encoded['Usage_Level_Encoded'] = le_target.fit_transform(df_encoded['Usage_Level'])
        encoders['Usage_Level'] = le_target
    
    return df_encoded, encoders

def prepare_features(df_encoded):
    """Persiapkan fitur untuk model"""
    if df_encoded is None or len(df_encoded) == 0:
        return None, None, None, []
    
    features = []
    
    # Add basic features
    if 'Semester' in df_encoded.columns:
        features.append('Semester')
    if 'Trust_Level' in df_encoded.columns:
        features.append('Trust_Level')
    
    # Add encoded features if they exist
    if 'Studi_Jurusan_Encoded' in df_encoded.columns:
        features.append('Studi_Jurusan_Encoded')
    if 'AI_Tools_Encoded' in df_encoded.columns:
        features.append('AI_Tools_Encoded')
    
    if not features:
        return None, None, None, []
    
    X = df_encoded[features]
    
    # Check if target exists
    if 'Usage_Level_Encoded' in df_encoded.columns:
        y = df_encoded['Usage_Level_Encoded']
    else:
        # If no target, create simple target based on score
        kb = KnowledgeBaseSystem()
        y = df_encoded['Usage_Intensity_Score'].apply(kb.classify_usage)
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, features

def train_random_forest(X, y):
    """Train model Random Forest"""
    if X is None or y is None or len(X) == 0:
        return None
    
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

def get_download_link(df, filename="dataset.csv", text="Download CSV"):
    """Generate download link for CSV file"""
    try:
        csv = df.to_csv(index=False, sep=';')
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; border-radius: 5px;">{text}</a>'
        return href
    except:
        return ""

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
    st.markdown("<h1 class='main-header'>Sistem Analisis Penggunaan AI Mahasiswa</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        role = st.selectbox("Pilih Peran", ["Guru/Dosen", "Mahasiswa"])
        
        if role == "Guru":
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
        st.markdown("### Informasi Sistem")
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
        
        

def upload_dataset_section():
    """Section untuk upload dataset"""
    st.header(" Upload Dataset CSV")
    
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
            <li>Pemisah kolom: titik koma (;) atau koma (,)</li>
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
            help="Upload dataset dalam format CSV. Sistem akan mendeteksi delimiter otomatis."
        )
    
    with col2:
        st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
        use_default = st.button(" Gunakan Dataset Default", use_container_width=True)
    
    # Handle file upload
    if uploaded_file is not None:
        try:
            # Read the CSV file with automatic delimiter detection
            df = read_csv_with_various_delimiters(uploaded_file)
            
            if df is None:
                st.error("Gagal membaca file CSV. Pastikan format file benar.")
                return
            
            # Show detected columns
            st.info(f"Kolom yang terdeteksi: {', '.join(df.columns.tolist())}")
            
            # Validate the dataset
            is_valid, message = validate_dataset(df)
            
            if is_valid:
                st.session_state.df = df
                st.session_state.uploaded_file = uploaded_file.name
                st.session_state.data_source = "uploaded"
                
                # Clear previous processed data
                st.session_state.df_clean = None
                st.session_state.model = None
                st.session_state.results = None
                st.session_state.predictions = None
                st.session_state.knowledge_base = None
                
                st.success(f"✅ Dataset berhasil diupload: {uploaded_file.name}")
                
                # Show dataset info
                st.markdown("<div class='file-info'>", unsafe_allow_html=True)
                st.markdown(f"**📊 Info Dataset:**")
                st.markdown(f"- **Nama file:** {uploaded_file.name}")
                st.markdown(f"- **Jumlah data:** {len(df)} baris")
                st.markdown(f"- **Jumlah kolom:** {len(df.columns)}")
                st.markdown(f"- **Kolom:** {', '.join(df.columns.tolist())}")
                
                # Show basic statistics
                if 'Usage_Intensity_Score' in df.columns:
                    # Convert scores for statistics
                    def get_score_val(x):
                        try:
                            x_str = str(x).strip()
                            if x_str == '10+':
                                return 10
                            return float(x_str)
                        except:
                            return np.nan
                    
                    scores = df['Usage_Intensity_Score'].apply(get_score_val).dropna()
                    if len(scores) > 0:
                        st.markdown("**📈 Statistik Skor:**")
                        st.markdown(f"- Rata-rata: {scores.mean():.2f}")
                        st.markdown(f"- Minimum: {scores.min():.0f}")
                        st.markdown(f"- Maximum: {scores.max():.0f}")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Show data preview
                st.subheader("👀 Preview Data (10 baris pertama)")
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
                st.info("""
                **Tips untuk memperbaiki dataset:**
                1. Pastikan file CSV memiliki 6 kolom: Nama, Studi_Jurusan, Semester, AI_Tools, Trust_Level, Usage_Intensity_Score
                2. Gunakan titik koma (;) sebagai pemisah kolom
                3. Pastikan tidak ada baris kosong di awal file
                4. Usage_Intensity_Score harus angka 1-10 atau '10+'
                5. Download template di atas untuk contoh format yang benar
                """)
                
                # Show what was read
                st.subheader("Data yang terbaca:")
                st.dataframe(df.head(), use_container_width=True)
                
        except Exception as e:
            st.error(f"Error membaca file: {str(e)}")
            st.info("Pastikan file CSV menggunakan format yang benar. Coba download template di atas.")
    
    # Handle default dataset
    elif use_default:
        with st.spinner("Memuat dataset default..."):
            df = load_default_dataset()
            
            if df is not None and len(df) > 0:
                st.session_state.df = df
                st.session_state.data_source = "default"
                st.session_state.uploaded_file = None
                
                # Clear previous processed data
                st.session_state.df_clean = None
                st.session_state.model = None
                st.session_state.results = None
                st.session_state.predictions = None
                st.session_state.knowledge_base = None
                
                st.success("✅ Dataset default berhasil dimuat!")
                
                # Show dataset info
                st.markdown("<div class='file-info'>", unsafe_allow_html=True)
                st.markdown(f"**Info Dataset:**")
                st.markdown(f"- **Sumber:** Dataset Default")
                st.markdown(f"- **Jumlah data:** {len(df)} baris")
                st.markdown(f"- **Jumlah kolom:** {len(df.columns)}")
                st.markdown(f"- **Kolom:** {', '.join(df.columns.tolist())}")
                
                # Show basic statistics
                if 'Usage_Intensity_Score' in df.columns:
                    # Convert scores for statistics
                    def get_score_val(x):
                        try:
                            x_str = str(x).strip()
                            if x_str == '10+':
                                return 10
                            return float(x_str)
                        except:
                            return np.nan
                    
                    scores = df['Usage_Intensity_Score'].apply(get_score_val).dropna()
                    if len(scores) > 0:
                        st.markdown("**Statistik Skor:**")
                        st.markdown(f"- Rata-rata: {scores.mean():.2f}")
                        st.markdown(f"- Minimum: {scores.min():.0f}")
                        st.markdown(f"- Maximum: {scores.max():.0f}")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Show data preview
                st.subheader("👀 Preview Data")
                st.dataframe(df.head(10), use_container_width=True)
            else:
                st.error("Gagal memuat dataset default")
    
    # Show current dataset if exists
    elif st.session_state.df is not None:
        st.subheader("Dataset Saat Ini")
        
        # Show dataset info
        df = st.session_state.df
        source = "Uploaded File" if st.session_state.data_source == "uploaded" else "Default"
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sumber Data", source)
        with col2:
            st.metric("Jumlah Data", len(df))
        with col3:
            if 'Usage_Intensity_Score' in df.columns:
                # Convert scores for min/max calculation
                def get_score_val(x):
                    try:
                        x_str = str(x).strip()
                        if x_str == '10+':
                            return 10
                        return float(x_str)
                    except:
                        return np.nan
                
                scores = df['Usage_Intensity_Score'].apply(get_score_val).dropna()
                if len(scores) > 0:
                    st.metric("Rentang Skor", f"{scores.min():.0f}-{scores.max():.0f}")
                else:
                    st.metric("Rentang Skor", "N/A")
        with col4:
            if 'Usage_Intensity_Score' in df.columns:
                scores = df['Usage_Intensity_Score'].apply(get_score_val).dropna()
                if len(scores) > 0:
                    st.metric("Skor Rata-rata", f"{scores.mean():.1f}")
                else:
                    st.metric("Skor Rata-rata", "N/A")
        
        # Data preview
        st.dataframe(df.head(10), use_container_width=True)
        
        # Option to clear dataset
        if st.button("Hapus Dataset Saat Ini", use_container_width=True):
            st.session_state.df = None
            st.session_state.df_clean = None
            st.session_state.model = None
            st.session_state.results = None
            st.session_state.predictions = None
            st.session_state.knowledge_base = None
            st.session_state.uploaded_file = None
            st.session_state.data_source = "default"
            st.rerun()
    
    else:
        st.info("👆 Silakan upload file CSV atau gunakan dataset default untuk memulai analisis.")

def score_interpretation_section():
    """Section untuk interpretasi skor"""
    st.header(" Interpretasi Usage_Intensity_Score")
    
    # Show detailed interpretation table
    show_score_interpretation_table()
    
    # Additional information
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Keterangan")
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
    st.subheader("Cek Interpretasi Skor")
    
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
        check_button = st.button("Interpretasi Skor", use_container_width=True)
    
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
            'RENDAH': '',
            'SEDANG': '',
            'TINGGI': ''
        }
        
        color = colors.get(classification, '#95A5A6')
        icon = icons.get(classification, '')
        
        # Display interpretation
        st.markdown(f"""
        <div style="background-color: {color}; color: white; padding: 20px; border-radius: 10px; margin-top: 20px;">
            <h3>{icon} Skor {score_input}: {classification}</h3>
            <p><strong>Frekuensi:</strong> {interpretation['frequency']}</p>
            <p><strong>Deskripsi:</strong> {interpretation['description']}</p>
        </div>
        """, unsafe_allow_html=True)

def data_cleaning_section():
    """Data cleaning section"""
    st.header("Data Cleaning")
    
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
    st.subheader("Data Sebelum Cleaning")
    
    # Convert scores for visualization
    def get_score_val(x):
        try:
            x_str = str(x).strip()
            if x_str == '10+':
                return 10
            return float(x_str)
        except:
            return np.nan
    
    scores = df['Usage_Intensity_Score'].apply(get_score_val) if 'Usage_Intensity_Score' in df.columns else pd.Series([])
    
    # Statistics before cleaning
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        duplicates = df.duplicated(subset=['Nama']).sum() if 'Nama' in df.columns else 0
        st.metric("Duplikat Nama", duplicates)
    with col2:
        missing = df.isnull().sum().sum()
        st.metric("Missing Values", missing)
    with col3:
        if 'Usage_Intensity_Score' in df.columns:
            valid_scores = scores.dropna()
            invalid_scores = len(df) - len(valid_scores)
            st.metric("Skor Tidak Valid", invalid_scores)
        else:
            st.metric("Skor Tidak Valid", "N/A")
    with col4:
        if 'Usage_Intensity_Score' in df.columns and len(scores.dropna()) > 0:
            avg_score = scores.dropna().mean()
            st.metric("Skor Rata-rata", f"{avg_score:.2f}")
        else:
            st.metric("Skor Rata-rata", "N/A")
    
    if st.button("🚀 Jalankan Data Cleaning", type="primary", use_container_width=True):
        with st.spinner("Melakukan data cleaning..."):
            df_clean = clean_data(df)
            st.session_state.df_clean = df_clean
            
            if df_clean is not None and len(df_clean) > 0:
                st.success("ata cleaning selesai!")
                
                # Show cleaning results
                st.subheader("Hasil Data Cleaning")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("**Sebelum Cleaning:**")
                    st.write(f"- Jumlah data: {len(df)}")
                    st.write(f"- Missing values: {missing}")
                    if 'Usage_Intensity_Score' in df.columns:
                        st.write(f"- Skor tidak valid: {invalid_scores}")
                        if len(scores.dropna()) > 0:
                            st.write(f"- Skor rata-rata: {scores.dropna().mean():.2f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='card success-card'>", unsafe_allow_html=True)
                    st.markdown("**Setelah Cleaning:**")
                    st.write(f"- Jumlah data: {len(df_clean)}")
                    st.write(f"- Missing values: {df_clean.isnull().sum().sum()}")
                    if 'Usage_Intensity_Score' in df_clean.columns:
                        st.write(f"- Skor tidak valid: 0")
                        st.write(f"- Skor rata-rata: {df_clean['Usage_Intensity_Score'].mean():.2f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Show cleaned data with classification
                st.subheader(" Data Setelah Cleaning dengan Klasifikasi")
                
                # Add classification to cleaned data
                kb = KnowledgeBaseSystem()
                df_display = df_clean.copy()
                
                if 'Usage_Intensity_Score' in df_display.columns:
                    df_display['Klasifikasi'] = df_display['Usage_Intensity_Score'].apply(kb.classify_usage)
                    df_display['Frekuensi'] = df_display['Usage_Intensity_Score'].apply(
                        lambda x: kb.get_frequency_interpretation(x)['frequency']
                    )
                
                # Display with classification colors
                def highlight_classification(val):
                    if val == 'RENDAH':
                        return 'background-color: #D5F4E6'
                    elif val == 'SEDANG':
                        return 'background-color: #FCF3CF'
                    elif val == 'TINGGI':
                        return 'background-color: #FADBD8'
                    else:
                        return ''
                
                # Show only selected columns
                display_cols = []
                for col in ['Nama', 'Studi_Jurusan', 'Semester', 'AI_Tools', 'Trust_Level', 'Usage_Intensity_Score', 'Klasifikasi', 'Frekuensi']:
                    if col in df_display.columns:
                        display_cols.append(col)
                
                if display_cols:
                    display_df = df_display[display_cols]
                    
                    # Apply styling if Klasifikasi exists
                    if 'Klasifikasi' in display_cols:
                        styled_df = display_df.style.applymap(highlight_classification, subset=['Klasifikasi'])
                        st.dataframe(styled_df, use_container_width=True, height=400)
                    else:
                        st.dataframe(display_df, use_container_width=True, height=400)
                else:
                    st.dataframe(df_display, use_container_width=True, height=400)
            else:
                st.error("Data cleaning gagal menghasilkan data yang valid")
    
    elif st.session_state.df_clean is not None:
        st.success("Data sudah dibersihkan sebelumnya!")
        
        df_clean = st.session_state.df_clean
        
        # Show cleaned data
        st.subheader("Data Setelah Cleaning")
        
        # Add classification to cleaned data
        kb = KnowledgeBaseSystem()
        df_display = df_clean.copy()
        
        if 'Usage_Intensity_Score' in df_display.columns:
            df_display['Klasifikasi'] = df_display['Usage_Intensity_Score'].apply(kb.classify_usage)
            df_display['Frekuensi'] = df_display['Usage_Intensity_Score'].apply(
                lambda x: kb.get_frequency_interpretation(x)['frequency']
            )
        
        # Show only selected columns
        display_cols = []
        for col in ['Nama', 'Studi_Jurusan', 'Semester', 'AI_Tools', 'Trust_Level', 'Usage_Intensity_Score', 'Klasifikasi', 'Frekuensi']:
            if col in df_display.columns:
                display_cols.append(col)
        
        if display_cols:
            st.dataframe(df_display[display_cols].head(10), use_container_width=True)
        else:
            st.dataframe(df_display.head(10), use_container_width=True)
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Data", len(df_clean))
        with col2:
            if 'Usage_Intensity_Score' in df_clean.columns:
                avg_score = df_clean['Usage_Intensity_Score'].mean()
                st.metric("Skor Rata-rata", f"{avg_score:.2f}")
        with col3:
            if 'Usage_Intensity_Score' in df_clean.columns:
                high_users = (df_clean['Usage_Intensity_Score'] >= 8).sum()
                st.metric("Pengguna Tinggi", high_users)
        
        # Option to re-run cleaning
        if st.button("Jalankan Ulang Data Cleaning", use_container_width=True):
            st.session_state.df_clean = None
            st.rerun()

def data_processing_section():
    """Data processing section"""
    st.header("Data Processing & Encoding")
    
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
    
    if st.button("⚙️ Proses Data", type="primary", use_container_width=True):
        with st.spinner("Memproses data..."):
            # Encode categorical data
            df_encoded, encoders = encode_categorical_data(st.session_state.df_clean)
            st.session_state.encoders = encoders
            
            # Prepare features
            X_scaled, y, scaler, features = prepare_features(df_encoded)
            
            if X_scaled is not None and y is not None:
                # Save processed data
                st.session_state.processed_data = {
                    'X': X_scaled,
                    'y': y,
                    'df_encoded': df_encoded,
                    'scaler': scaler,
                    'features': features
                }
                
                st.success("✅ Data processing selesai!")
                
                # Show encoding results
                st.subheader(" Hasil Encoding")
                
                # Show encoded values
                if 'Studi_Jurusan' in encoders:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        st.markdown("**Mapping Jurusan:**")
                        jurusan_mapping = dict(zip(
                            encoders['Studi_Jurusan'].classes_,
                            range(len(encoders['Studi_Jurusan'].classes_))
                        ))
                        for key, value in jurusan_mapping.items():
                            st.write(f"{key} → {value}")
                        st.markdown("</div>", unsafe_allow_html=True)
                
                # Show processed data
                st.subheader("Data Setelah Processing")
                
                # Add frequency interpretation
                kb = KnowledgeBaseSystem()
                df_display = df_encoded.copy()
                
                if 'Usage_Intensity_Score' in df_display.columns:
                    df_display['Frekuensi'] = df_display['Usage_Intensity_Score'].apply(
                        lambda x: kb.get_frequency_interpretation(x)['frequency']
                    )
                
                # Show selected columns
                display_cols = []
                for col in ['Nama', 'Studi_Jurusan', 'Semester', 'AI_Tools', 'Trust_Level', 'Usage_Intensity_Score', 'Frekuensi']:
                    if col in df_display.columns:
                        display_cols.append(col)
                
                if 'Usage_Level' in df_display.columns:
                    display_cols.append('Usage_Level')
                
                if display_cols:
                    st.dataframe(df_display[display_cols].head(), use_container_width=True)
                
                # Show target distribution
                if 'Usage_Level' in df_encoded.columns:
                    st.subheader("Distribusi Target (Usage Level)")
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
                    st.subheader("Statistik Klasifikasi")
                    
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
            else:
                st.error("Gagal memproses data. Pastikan dataset memiliki kolom yang diperlukan.")

def model_training_section():
    """Model training section"""
    st.header("Model Training - Random Forest")
    
    if 'processed_data' not in st.session_state:
        st.warning("Harap proses data terlebih dahulu di menu Data Processing!")
        return
    
    st.info("""
    **Algoritma Random Forest untuk Klasifikasi Penggunaan AI:**
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
    
    if st.button("Train Model", type="primary", use_container_width=True):
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
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save model and results
            st.session_state.model = {
                'model': model,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'y_pred': y_pred,
                'params': model_params,
                'accuracy': accuracy
            }
            
            st.success("✅ Model berhasil dilatih!")
            
            # Show training results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{accuracy:.2%}")
            with col2:
                st.metric("Jumlah Trees", n_estimators)
            with col3:
                st.metric("Max Depth", max_depth)
            
            # Feature importance
            st.subheader(" Feature Importance")
            
            feature_names = st.session_state.processed_data['features']
            importances = model.feature_importances_
            
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'Fitur': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Plot feature importance
            fig = px.bar(importance_df, x='Fitur', y='Importance',
                        title="Importance Fitur dalam Model",
                        labels={'Importance': 'Importance', 'Fitur': 'Fitur'},
                        color='Importance',
                        color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)

def model_evaluation_section():
    """Model evaluation section"""
    st.header(" Evaluasi Model")
    
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
    """)
    
    # Calculate metrics
    accuracy = accuracy_score(model_data['y_test'], model_data['y_pred'])
    
    # Get classification report
    try:
        # Get class names
        if 'Usage_Level' in st.session_state.encoders:
            le_target = st.session_state.encoders['Usage_Level']
            target_names = le_target.classes_
        else:
            target_names = ['RENDAH', 'SEDANG', 'TINGGI']
        
        report = classification_report(model_data['y_test'], model_data['y_pred'], 
                                      target_names=target_names,
                                      output_dict=True)
    except:
        # Fallback report
        report = {
            'accuracy': accuracy,
            'weighted avg': {
                'precision': accuracy,
                'recall': accuracy,
                'f1-score': accuracy
            }
        }
    
    # Display metrics
    st.subheader(" Performa Model")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{accuracy:.2%}")
    with col2:
        precision = report.get('weighted avg', {}).get('precision', accuracy)
        st.metric("Precision", f"{precision:.2%}")
    with col3:
        recall = report.get('weighted avg', {}).get('recall', accuracy)
        st.metric("Recall", f"{recall:.2%}")
    with col4:
        f1 = report.get('weighted avg', {}).get('f1-score', accuracy)
        st.metric("F1-Score", f"{f1:.2%}")
    
    # Classification report
    st.subheader(" Classification Report Detail")
    
    # Create a styled dataframe
    if isinstance(report, dict) and 'RENDAH' in report:
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
        
        styled_report = report_df.style.applymap(color_metrics, 
                                               subset=pd.IndexSlice[['RENDAH', 'SEDANG', 'TINGGI', 'weighted avg'], 
                                                                    ['precision', 'recall', 'f1-score']])
        st.dataframe(styled_report, use_container_width=True)

def recommendations_section():
    """Recommendations section - DIPERBAIKI"""
    st.header("🎯 Rekomendasi Berdasarkan Analisis")
    
    # Initialize knowledge base if not exists
    if st.session_state.knowledge_base is None:
        st.session_state.knowledge_base = KnowledgeBaseSystem()
    
    # Check if we have clean data
    if st.session_state.df_clean is None:
        st.warning("Harap lakukan data cleaning terlebih dahulu!")
        return
    
    df_clean = st.session_state.df_clean
    
    # Button to generate recommendations
    if st.button("🔄 Generate Rekomendasi", type="primary", use_container_width=True):
        with st.spinner("Membuat rekomendasi..."):
            kb = st.session_state.knowledge_base
            
            # Generate recommendations for each student
            recommendations_list = []
            classifications_list = []
            
            for idx, student in df_clean.iterrows():
                score = student['Usage_Intensity_Score'] if 'Usage_Intensity_Score' in student else 5
                classification = kb.classify_usage(score)
                classifications_list.append(classification)
                
                # Get recommendation
                recommendation = kb.get_recommendation(classification, student)
                recommendations_list.append(recommendation)
            
            # Generate summary report
            summary = kb.generate_summary_report(classifications_list)
            
            # Store results
            st.session_state.results = {
                'recommendations': recommendations_list,
                'summary': summary,
                'classifications': classifications_list
            }
            
            st.success(f"✅ Rekomendasi berhasil dibuat untuk {len(recommendations_list)} mahasiswa!")
    
    # Display results if available
    if st.session_state.results:
        results = st.session_state.results
        summary = results['summary']
        
        # Display summary
        st.subheader(" Ringkasan Analisis")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Mahasiswa", summary['total_students'])
        with col2:
            st.metric("Data Valid", summary['valid_students'])
        with col3:
            st.metric("Level Dominan", summary['dominant_class'])
        with col4:
            st.metric("Tingkat Risiko", summary['risk_level'])
        
        # Display distribution chart
        st.subheader("📈 Distribusi Klasifikasi")
        
        # Create distribution dataframe
        dist_data = {
            'Kategori': ['RENDAH', 'SEDANG', 'TINGGI'],
            'Jumlah': [
                summary['counts'].get('RENDAH', 0),
                summary['counts'].get('SEDANG', 0),
                summary['counts'].get('TINGGI', 0)
            ]
        }
        
        dist_df = pd.DataFrame(dist_data)
        
        fig = px.bar(dist_df, x='Kategori', y='Jumlah',
                    title="Distribusi Mahasiswa Berdasarkan Klasifikasi",
                    color='Kategori',
                    color_discrete_map={
                        'RENDAH': '#2ECC71',
                        'SEDANG': '#F39C12',
                        'TINGGI': '#E74C3C'
                    })
        st.plotly_chart(fig, use_container_width=True)
        
        # Display detailed recommendations by category
        st.subheader("Detail Rekomendasi per Kategori")
        
        # Create tabs for each category
        tab1, tab2, tab3 = st.tabs(["RENDAH", "🟡 SEDANG", "🔴 TINGGI"])
        
        # Filter recommendations by category
        rendah_rec = [r for r in results['recommendations'] if r['classification'] == 'RENDAH']
        sedang_rec = [r for r in results['recommendations'] if r['classification'] == 'SEDANG']
        tinggi_rec = [r for r in results['recommendations'] if r['classification'] == 'TINGGI']
        
        with tab1:
            if rendah_rec:
                st.markdown(f"###  Level RENDAH ({len(rendah_rec)} mahasiswa)")
                for rec in rendah_rec[:5]:  # Show first 5
                    with st.expander(f"{rec['student_name']} - {rec['jurusan']}"):
                        st.write(f"**Skor:** {rec['usage_score']}")
                        st.write(f"**Trust Level:** {rec['trust_level']}")
                        st.write(f"**AI Tool:** {rec['ai_tool']}")
                        st.write(f"**Frekuensi:** {rec['frequency_interpretation']['frequency']}")
                        st.write("**Rekomendasi:**")
                        for detail in rec['details']:
                            st.write(f"- {detail}")
            else:
                st.info("Tidak ada mahasiswa dengan level RENDAH")
        
        with tab2:
            if sedang_rec:
                st.markdown(f"###  Level SEDANG ({len(sedang_rec)} mahasiswa)")
                for rec in sedang_rec[:5]:  # Show first 5
                    with st.expander(f"{rec['student_name']} - {rec['jurusan']}"):
                        st.write(f"**Skor:** {rec['usage_score']}")
                        st.write(f"**Trust Level:** {rec['trust_level']}")
                        st.write(f"**AI Tool:** {rec['ai_tool']}")
                        st.write(f"**Frekuensi:** {rec['frequency_interpretation']['frequency']}")
                        st.write("**Rekomendasi:**")
                        for detail in rec['details']:
                            st.write(f"- {detail}")
            else:
                st.info("Tidak ada mahasiswa dengan level SEDANG")
        
        with tab3:
            if tinggi_rec:
                st.markdown(f"###  Level TINGGI ({len(tinggi_rec)} mahasiswa)")
                for rec in tinggi_rec[:5]:  # Show first 5
                    with st.expander(f"{rec['student_name']} - {rec['jurusan']}"):
                        st.write(f"**Skor:** {rec['usage_score']}")
                        st.write(f"**Trust Level:** {rec['trust_level']}")
                        st.write(f"**AI Tool:** {rec['ai_tool']}")
                        st.write(f"**Frekuensi:** {rec['frequency_interpretation']['frequency']}")
                        st.write("**Rekomendasi:**")
                        for detail in rec['details']:
                            st.write(f"- {detail}")
            else:
                st.info("Tidak ada mahasiswa dengan level TINGGI")
        
        # Export options
        st.subheader("Export Hasil")
        
        if st.button("xport ke CSV", use_container_width=True):
            # Create export dataframe
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
                    'Rekomendasi_1': rec['details'][0] if len(rec['details']) > 0 else '',
                    'Rekomendasi_2': rec['details'][1] if len(rec['details']) > 1 else '',
                    'Tindakan_1': rec['actions'][0] if len(rec['actions']) > 0 else '',
                    'Tindakan_2': rec['actions'][1] if len(rec['actions']) > 1 else ''
                })
            
            export_df = pd.DataFrame(export_data)
            
            # Convert to CSV
            csv = export_df.to_csv(index=False, sep=';')
            
            # Create download button
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="rekomendasi_ai_mahasiswa.csv" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; border-radius: 5px;">📥 Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    else:
        st.info("Klik tombol 'Generate Rekomendasi' untuk membuat analisis rekomendasi.")

def analytics_dashboard_section():
    """Analytics dashboard section"""
    st.header("📊 Dashboard Analitik")
    
    if st.session_state.df_clean is None:
        st.warning("Data belum tersedia untuk analisis!")
        return
    
    df = st.session_state.df_clean
    
    # Key metrics
    st.subheader("📈 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'Usage_Intensity_Score' in df.columns:
            avg_usage = df['Usage_Intensity_Score'].mean()
            st.metric("Rata-rata Penggunaan", f"{avg_usage:.2f}")
        else:
            st.metric("Rata-rata Penggunaan", "N/A")
    
    with col2:
        if 'Trust_Level' in df.columns:
            avg_trust = df['Trust_Level'].mean()
            st.metric("Rata-rata Trust Level", f"{avg_trust:.2f}")
        else:
            st.metric("Rata-rata Trust Level", "N/A")
    
    with col3:
        if 'Usage_Intensity_Score' in df.columns:
            high_users = (df['Usage_Intensity_Score'] >= 8).sum()
            st.metric("Pengguna Intensif", high_users)
        else:
            st.metric("Pengguna Intensif", "N/A")
    
    with col4:
        if 'Usage_Intensity_Score' in df.columns:
            low_users = (df['Usage_Intensity_Score'] <= 3).sum()
            st.metric("Pengguna Rendah", low_users)
        else:
            st.metric("Pengguna Rendah", "N/A")
    
    # Show data table
    st.subheader("📋 Data Analisis")
    
    # Add classification if knowledge base exists
    if st.session_state.knowledge_base:
        kb = st.session_state.knowledge_base
        df_display = df.copy()
        
        if 'Usage_Intensity_Score' in df_display.columns:
            df_display['Klasifikasi'] = df_display['Usage_Intensity_Score'].apply(kb.classify_usage)
            df_display['Frekuensi'] = df_display['Usage_Intensity_Score'].apply(
                lambda x: kb.get_frequency_interpretation(x)['frequency']
            )
        
        # Show selected columns
        display_cols = []
        for col in ['Nama', 'Studi_Jurusan', 'Semester', 'AI_Tools', 'Trust_Level', 'Usage_Intensity_Score', 'Klasifikasi', 'Frekuensi']:
            if col in df_display.columns:
                display_cols.append(col)
        
        if display_cols:
            st.dataframe(df_display[display_cols], use_container_width=True, height=400)
        else:
            st.dataframe(df_display, use_container_width=True, height=400)
    else:
        st.dataframe(df, use_container_width=True, height=400)

def mahasiswa_dashboard():
    """Dashboard untuk Mahasiswa"""
    st.markdown(f"<h1 class='main-header'> Dashboard Mahasiswa</h1>", unsafe_allow_html=True)
    
    student_name = st.session_state.get('student_name', '')
    
    if student_name:
        st.markdown(f"<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"###  Selamat datang, **{student_name}**!")
        st.markdown("Di sini Anda dapat melihat hasil analisis penggunaan AI Anda berdasarkan self-report.")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Show score interpretation table
        show_score_interpretation_table()
        
        # Check if analysis is available
        if st.session_state.df_clean is not None and st.session_state.knowledge_base is not None:
            # Find student data
            df_clean = st.session_state.df_clean
            
            # Case-insensitive search
            student_data = df_clean[
                df_clean['Nama'].str.contains(student_name, case=False, na=False)
            ]
            
            if not student_data.empty:
                student = student_data.iloc[0]
                
                # Classify student
                kb = st.session_state.knowledge_base
                score = student['Usage_Intensity_Score'] if 'Usage_Intensity_Score' in student else 5
                classification = kb.classify_usage(score)
                recommendation = kb.get_recommendation(classification, student)
                
                # Display student info
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("###  Profil Akademik")
                    st.markdown(f"**Nama:** {student['Nama']}")
                    if 'Studi_Jurusan' in student:
                        st.markdown(f"**Jurusan:** {student['Studi_Jurusan']}")
                    if 'Semester' in student:
                        st.markdown(f"**Semester:** {student['Semester']}")
                    if 'AI_Tools' in student:
                        st.markdown(f"**Tools AI:** {student['AI_Tools']}")
                    if 'Trust_Level' in student:
                        st.markdown(f"**Trust Level:** {student['Trust_Level']}/5")
                    st.markdown(f"**Usage Score:** {score}/10")
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
                    if 'monitoring_level' in recommendation:
                        st.markdown(f"** Level Monitoring:** {recommendation['monitoring_level']}")
                    
                    # Show score interpretation
                    if 'frequency_interpretation' in recommendation:
                        st.markdown("---")
                        st.markdown(f"**📈 Interpretasi Skor {score}:**")
                        st.markdown(f"- **Frekuensi:** {recommendation['frequency_interpretation']['frequency']}")
                        st.markdown(f"- **Deskripsi:** {recommendation['frequency_interpretation']['description']}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Display recommendations
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("###  Rekomendasi untuk Anda")
                
                if 'details' in recommendation:
                    st.markdown("**Detail Rekomendasi:**")
                    for rec in recommendation['details']:
                        st.markdown(f"• {rec}")
                
                if 'actions' in recommendation:
                    st.markdown("**Tindakan yang Disarankan:**")
                    for action in recommendation['actions']:
                        st.markdown(f"▶️ {action}")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
            else:
                st.warning(f"Data tidak ditemukan untuk mahasiswa: '{student_name}'.")
                st.info("""
                **Kemungkinan penyebab:**
                1. Nama tidak terdaftar dalam dataset
                2. Data Anda belum diproses oleh administrator
                3. Terdapat perbedaan penulisan nama
                
                **Mahasiswa yang tersedia dalam dataset:**
                """)
                
                # Show available students
                if len(df_clean) > 0 and 'Nama' in df_clean.columns:
                    st.dataframe(df_clean[['Nama', 'Studi_Jurusan']].head(10) if 'Studi_Jurusan' in df_clean.columns else df_clean[['Nama']].head(10), 
                                use_container_width=True)
        else:
            st.info("🔄 Analisis sedang dipersiapkan oleh administrator. Silakan coba lagi nanti.")
    
    # Logout button
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.user_role = None
        st.rerun()

def guru_dashboard():
    """Dashboard untuk Guru"""
    st.markdown("<h1 class='main-header'> Dashboard Guru</h1>", unsafe_allow_html=True)
    
    # Initialize knowledge base
    if st.session_state.knowledge_base is None:
        st.session_state.knowledge_base = KnowledgeBaseSystem()
    
    # Sidebar menu
    with st.sidebar:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Menu Analisis")
        menu = st.radio(
            "Pilih Menu:",
            [" Upload Dataset", " Interpretasi Skor", " Data Cleaning", 
             "Data Processing", " Model Training", " Evaluasi Model", 
             " Rekomendasi", " Dashboard Analitik"]
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Data source info
        if st.session_state.data_source:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 📂 Sumber Data")
            if st.session_state.data_source == "uploaded":
                if st.session_state.uploaded_file:
                    st.success(f"✅ {st.session_state.uploaded_file}")
                else:
                    st.success("✅ Dataset dari file upload")
            else:
                st.info("📊 Dataset default")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Data status
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📈 Status Data")
        
        if st.session_state.df is not None:
            st.success("Data tersedia")
            if st.session_state.df_clean is not None:
                st.success("Data sudah dibersihkan")
            else:
                st.warning("Data belum dibersihkan")
            
            if st.session_state.model is not None:
                st.success("Model sudah dilatih")
            else:
                st.info("ℹ️ Model belum dilatih")
        else:
            st.error("Data belum dimuat")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Logout button
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_role = None
            st.rerun()
    
    # Main content based on menu
    if menu == "Upload Dataset":
        upload_dataset_section()
    elif menu == "Interpretasi Skor":
        score_interpretation_section()
    elif menu == "Data Cleaning":
        data_cleaning_section()
    elif menu == "Data Processing":
        data_processing_section()
    elif menu == "Model Training":
        model_training_section()
    elif menu == "Evaluasi Model":
        model_evaluation_section()
    elif menu == "Rekomendasi":
        recommendations_section()
    elif menu == "Dashboard Analitik":
        analytics_dashboard_section()

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
