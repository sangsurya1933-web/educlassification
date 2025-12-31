import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import io
import base64

# ============================================
# KONFIGURASI HALAMAN
# ============================================
st.set_page_config(
    page_title="AI Academic Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# STYLE CUSTOM
# ============================================
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Card styling */
    .card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-left: 4px solid #4e54c8;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    
    /* Status badges */
    .status-safe {
        background-color: #d4edda;
        color: #155724;
        padding: 5px 15px;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
    }
    
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 5px 15px;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
    }
    
    .status-danger {
        background-color: #f8d7da;
        color: #721c24;
        padding: 5px 15px;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f3f4;
        border-radius: 10px 10px 0px 0px;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4e54c8;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# KNOWLEDGE BASE SYSTEM
# ============================================
class AIKnowledgeBase:
    def __init__(self):
        self.categories = {
            'USAGE_LEVEL': {
                'VERY_LOW': {
                    'label': 'Sangat Rendah',
                    'level': 'AMAN',
                    'icon': '✅',
                    'color': '#28a745',
                    'score': 1,
                    'description': 'Penggunaan AI dalam batas normal',
                    'actions': [
                        'Tidak diperlukan tindakan khusus',
                        'Pertahankan pola penggunaan saat ini',
                        'AI digunakan secara optimal sebagai alat bantu'
                    ],
                    'recommendations': [
                        'Lanjutkan penggunaan AI yang sehat',
                        'Eksplorasi tools AI untuk efisiensi',
                        'Bagikan pengalaman positif dengan teman'
                    ]
                },
                'LOW': {
                    'label': 'Rendah',
                    'level': 'AMAN',
                    'icon': '🟢',
                    'color': '#20c997',
                    'score': 2,
                    'description': 'Penggunaan AI masih wajar',
                    'actions': [
                        'Monitoring rutin',
                        'Edukasi penggunaan optimal',
                        'Evaluasi berkala'
                    ],
                    'recommendations': [
                        'Tetap waspada terhadap peningkatan ketergantungan',
                        'Gunakan AI untuk tugas kompleks',
                        'Jaga keseimbangan antara teknologi dan kemampuan manusia'
                    ]
                },
                'MEDIUM': {
                    'label': 'Sedang',
                    'level': 'PERHATIAN',
                    'icon': '🟡',
                    'color': '#ffc107',
                    'score': 3,
                    'description': 'Perlu evaluasi penggunaan AI',
                    'actions': [
                        'Konsultasi dengan dosen',
                        'Pembatasan penggunaan',
                        'Workshop pengurangan ketergantungan'
                    ],
                    'recommendations': [
                        'Batasi penggunaan AI pada tugas tertentu',
                        'Kembangkan kemampuan problem-solving mandiri',
                        'Ikuti workshop keterampilan belajar'
                    ]
                },
                'HIGH': {
                    'label': 'Tinggi',
                    'level': 'WASPADA',
                    'icon': '🟠',
                    'color': '#fd7e14',
                    'score': 4,
                    'description': 'Butuh pembatasan penggunaan AI',
                    'actions': [
                        'Program pembatasan AI',
                        'Konsultasi akademik intensif',
                        'Monitoring harian'
                    ],
                    'recommendations': [
                        'Kurangi penggunaan AI secara bertahap',
                        'Fokus pada pengembangan kemampuan analitis',
                        'Cari alternatif metode belajar'
                    ]
                },
                'VERY_HIGH': {
                    'label': 'Sangat Tinggi',
                    'level': 'KRITIS',
                    'icon': '🔴',
                    'color': '#dc3545',
                    'score': 5,
                    'description': 'Diperlukan intervensi segera',
                    'actions': [
                        'Intervensi langsung oleh pembimbing',
                        'Program rehabilitasi khusus',
                        'Monitoring ketat'
                    ],
                    'recommendations': [
                        'Hentikan penggunaan AI untuk tugas akademik',
                        'Ikuti program khusus pengurangan ketergantungan',
                        'Konsultasi dengan psikolog pendidikan'
                    ]
                }
            },
            'IMPACT_TYPES': {
                'POSITIVE': {
                    'icon': '📈',
                    'description': 'AI meningkatkan performa akademik',
                    'guidance': 'Pertahankan dengan pengawasan'
                },
                'NEUTRAL': {
                    'icon': '➖',
                    'description': 'AI tidak berpengaruh signifikan',
                    'guidance': 'Evaluasi manfaat penggunaan'
                },
                'NEGATIVE': {
                    'icon': '📉',
                    'description': 'AI menurunkan kemampuan mandiri',
                    'guidance': 'Kurangi penggunaan segera'
                }
            }
        }
    
    def get_category_info(self, category_key):
        """Mendapatkan informasi kategori"""
        return self.categories['USAGE_LEVEL'].get(category_key, {})
    
    def get_all_categories(self):
        """Mendapatkan semua kategori"""
        return list(self.categories['USAGE_LEVEL'].keys())
    
    def get_recommendation_summary(self, category_key):
        """Ringkasan rekomendasi untuk kategori tertentu"""
        info = self.get_category_info(category_key)
        if not info:
            return ""
        
        return f"""
        ### {info['icon']} {info['label']} - Level: {info['level']}
        
        **Deskripsi:** {info['description']}
        
        **Skor Risiko:** {info['score']}/5
        
        **Tindakan yang Disarankan:**
        {chr(10).join(['• ' + action for action in info['actions']])}
        
        **Rekomendasi untuk Mahasiswa:**
        {chr(10).join(['• ' + rec for rec in info['recommendations']])}
        """

# ============================================
# KOMPONEN UI
# ============================================
def create_metric_card(title, value, change=None, icon="📊"):
    """Membuat kartu metrik"""
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown(f"<h1 style='font-size: 2.5rem; margin: 0;'>{icon}</h1>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<h3 style='margin: 0; color: #666;'>{title}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='margin: 0;'>{value}</h1>", unsafe_allow_html=True)
        if change:
            st.markdown(f"<p style='margin: 0; color: #28a745;'>{change}</p>", unsafe_allow_html=True)

def create_status_badge(level, text):
    """Membuat badge status"""
    if level == 'AMAN':
        badge_class = "status-safe"
    elif level == 'PERHATIAN':
        badge_class = "status-warning"
    else:
        badge_class = "status-danger"
    
    return f'<span class="{badge_class}">{text}</span>'

def create_progress_chart(labels, values, colors):
    """Membuat chart progress"""
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=values,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, range=[0, max(values)*1.1])
    )
    
    return fig

# ============================================
# FUNGSI UTILITAS
# ============================================
@st.cache_data
def load_sample_data():
    """Memuat data contoh"""
    np.random.seed(42)
    n_samples = 150
    
    data = {
        'ID': [f'STD{i:03d}' for i in range(1, n_samples+1)],
        'Nama': [f'Mahasiswa_{i}' for i in range(1, n_samples+1)],
        'Fakultas': np.random.choice(['Teknik', 'Sains', 'Ekonomi', 'Kedokteran', 'Hukum'], n_samples),
        'IPK': np.round(np.random.normal(3.2, 0.5, n_samples), 2),
        'Jam_Belajar': np.random.randint(10, 50, n_samples),
        'Penggunaan_AI_Jam': np.random.randint(0, 40, n_samples),
        'Frekuensi_AI': np.random.choice(['Sangat Rendah', 'Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi'], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1]),
        'Ketergantungan': np.random.choice(['Rendah', 'Sedang', 'Tinggi'], n_samples),
        'Nilai_Rata': np.random.randint(60, 100, n_samples),
        'Status': np.random.choice(['Aktif', 'Lulus', 'Cuti'], n_samples, p=[0.85, 0.1, 0.05])
    }
    
    df = pd.DataFrame(data)
    return df

def preprocess_data(df):
    """Preprocessing data sederhana"""
    df_clean = df.copy()
    
    # Encoding
    le = LabelEncoder()
    df_clean['Frekuensi_AI_Encoded'] = le.fit_transform(df_clean['Frekuensi_AI'])
    df_clean['Ketergantungan_Encoded'] = le.fit_transform(df_clean['Ketergantungan'])
    
    return df_clean, le

def train_ai_model(X_train, X_test, y_train, y_test):
    """Training model AI"""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, y_pred, accuracy

# ============================================
# HALAMAN LOGIN
# ============================================
def login_page():
    """Halaman login"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<h1 style='font-size: 3rem; color: #4e54c8;'>🧠 AI Academic Analyzer</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #666;'>Analisis Penggunaan AI terhadap Performa Akademik Mahasiswa</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; 
                    padding: 30px; 
                    border-radius: 15px; 
                    margin: 20px 0;'>
            <h3>🏆 Fitur Utama</h3>
            <p>• Analisis tingkat penggunaan AI mahasiswa</p>
            <p>• Sistem rekomendasi berbasis knowledge base</p>
            <p>• Dashboard monitoring real-time</p>
            <p>• Laporan analisis otomatis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🔐 Login")
        
        role = st.radio(
            "Masuk sebagai:",
            ["🎓 Mahasiswa", "👨‍🏫 Dosen/Pengajar"],
            horizontal=True
        )
        
        if "Dosen" in role:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Login", use_container_width=True):
                if username == "dosen" and password == "123":
                    st.session_state.logged_in = True
                    st.session_state.role = "dosen"
                    st.rerun()
                else:
                    st.error("Username atau password salah")
        else:
            student_id = st.text_input("NIM/Nama")
            if st.button("Masuk sebagai Mahasiswa", use_container_width=True):
                if student_id:
                    st.session_state.logged_in = True
                    st.session_state.role = "mahasiswa"
                    st.session_state.student_name = student_id
                    st.rerun()
        
        st.markdown("---")
        st.markdown("**Demo Credentials:**")
        st.markdown("- Dosen: `dosen` / `123`")
        st.markdown("- Mahasiswa: masukkan nama/NIM")
        st.markdown("</div>", unsafe_allow_html=True)

# ============================================
# DASHBOARD DOSEN
# ============================================
def dosen_dashboard():
    """Dashboard dosen"""
    # Sidebar
    with st.sidebar:
        st.markdown("<h2>📊 Menu Dosen</h2>", unsafe_allow_html=True)
        
        menu_option = st.selectbox(
            "Navigasi",
            ["🏠 Dashboard Utama", "📁 Kelola Data", "🤖 Analisis AI", "📈 Visualisasi", "ℹ️ Knowledge Base", "⚙️ Pengaturan"]
        )
        
        st.markdown("---")
        st.markdown("### 📋 Quick Stats")
        
        if 'df' in st.session_state:
            df = st.session_state.df
            st.metric("Total Mahasiswa", len(df))
            st.metric("Rata-rata IPK", f"{df['IPK'].mean():.2f}")
            st.metric("Penggunaan AI", f"{df['Penggunaan_AI_Jam'].mean():.1f} jam/minggu")
        
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()
    
    # Main content
    st.markdown(f"<h1>{menu_option.split(' ')[1]} Dosen</h1>", unsafe_allow_html=True)
    
    if menu_option == "🏠 Dashboard Utama":
        show_dosen_dashboard()
    elif menu_option == "📁 Kelola Data":
        show_data_management()
    elif menu_option == "🤖 Analisis AI":
        show_ai_analysis()
    elif menu_option == "📈 Visualisasi":
        show_visualization()
    elif menu_option == "ℹ️ Knowledge Base":
        show_knowledge_base()
    elif menu_option == "⚙️ Pengaturan":
        show_settings()

def show_dosen_dashboard():
    """Dashboard utama dosen"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_metric_card("Total Mahasiswa", "150", "+5%", "👨‍🎓")
    with col2:
        create_metric_card("Rata-rata IPK", "3.42", "+0.1", "📚")
    with col3:
        create_metric_card("Penggunaan AI", "18.5 jam", "-2%", "🤖")
    with col4:
        create_metric_card("Akurasi Model", "89%", "+3%", "🎯")
    
    # Quick actions
    st.markdown("<h3>⚡ Quick Actions</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📥 Import Data", use_container_width=True):
            st.session_state.show_import = True
    with col2:
        if st.button("🔍 Analisis Cepat", use_container_width=True):
            st.session_state.quick_analysis = True
    with col3:
        if st.button("📊 Generate Report", use_container_width=True):
            st.session_state.generate_report = True
    with col4:
        if st.button("👁️ Preview Data", use_container_width=True):
            st.session_state.preview_data = True
    
    # Recent activities
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📋 Aktivitas Terkini")
    
    activities = [
        {"time": "10:30", "action": "Data mahasiswa diupdate", "user": "Admin"},
        {"time": "09:15", "action": "Model AI ditraining ulang", "user": "System"},
        {"time": "Yesterday", "action": "Laporan bulanan digenerate", "user": "Admin"},
        {"time": "2 days ago", "action": "5 mahasiswa dikategorikan 'Waspada'", "user": "System"}
    ]
    
    for act in activities:
        col1, col2, col3 = st.columns([2, 3, 2])
        with col1:
            st.write(f"**{act['time']}**")
        with col2:
            st.write(act['action'])
        with col3:
            st.write(f"👤 {act['user']}")
    st.markdown("</div>", unsafe_allow_html=True)

def show_data_management():
    """Manajemen data"""
    tab1, tab2, tab3 = st.tabs(["📥 Import Data", "🧹 Preprocessing", "📋 Preview Data"])
    
    with tab1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📥 Import Data Mahasiswa")
        
        col1, col2 = st.columns(2)
        
        with col1:
            upload_method = st.radio(
                "Pilih metode import:",
                ["📁 Upload File", "📝 Manual Input", "🔄 Gunakan Sample Data"]
            )
            
            if upload_method == "📁 Upload File":
                uploaded_file = st.file_uploader("Upload file CSV atau Excel", type=['csv', 'xlsx'])
                if uploaded_file:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    st.session_state.df = df
                    st.success(f"✅ Data berhasil diimport: {len(df)} baris")
            
            elif upload_method == "🔄 Gunakan Sample Data":
                if st.button("Load Sample Data", use_container_width=True):
                    st.session_state.df = load_sample_data()
                    st.success("✅ Sample data loaded!")
        
        with col2:
            if 'df' in st.session_state:
                st.metric("Total Data", len(st.session_state.df))
                st.metric("Kolom", len(st.session_state.df.columns))
                st.metric("Missing Values", st.session_state.df.isnull().sum().sum())
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        if 'df' not in st.session_state:
            st.warning("Silakan import data terlebih dahulu")
            return
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🧹 Data Preprocessing")
        
        df = st.session_state.df
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Data Cleaning")
            if st.button("🔍 Cek Missing Values", use_container_width=True):
                missing = df.isnull().sum()
                if missing.sum() > 0:
                    st.warning(f"Ditemukan {missing.sum()} missing values")
                    st.dataframe(missing[missing > 0])
                else:
                    st.success("✅ Tidak ada missing values")
            
            if st.button("🧼 Clean Data", use_container_width=True):
                df_clean = df.dropna()
                st.session_state.df = df_clean
                st.success(f"✅ Data cleaned: {len(df_clean)} baris tersisa")
        
        with col2:
            st.markdown("#### Encoding")
            if st.button("🔢 Encode Categorical", use_container_width=True):
                df_encoded, encoder = preprocess_data(df)
                st.session_state.df_encoded = df_encoded
                st.session_state.encoder = encoder
                st.success("✅ Data berhasil di-encode")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        if 'df' in st.session_state:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 📋 Preview Data")
            
            # Filter dan pencarian
            col1, col2 = st.columns(2)
            with col1:
                search_term = st.text_input("🔍 Cari nama/NIM")
            with col2:
                rows_to_show = st.slider("Baris ditampilkan", 10, 100, 20)
            
            # Filter data
            df_display = st.session_state.df
            if search_term:
                df_display = df_display[df_display.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)]
            
            st.dataframe(df_display.head(rows_to_show), use_container_width=True)
            
            # Statistik
            if st.checkbox("Tampilkan Statistik"):
                st.write(df_display.describe())
            
            st.markdown("</div>", unsafe_allow_html=True)

def show_ai_analysis():
    """Analisis AI"""
    tab1, tab2, tab3 = st.tabs(["🎯 Training Model", "📊 Hasil Analisis", "💡 Rekomendasi"])
    
    with tab1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🎯 Training Model AI")
        
        if 'df' not in st.session_state:
            st.warning("Silakan import data terlebih dahulu")
            return
        
        # Parameter model
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Parameter Model")
            n_estimators = st.slider("Jumlah Tree", 10, 200, 100)
            test_size = st.slider("Test Size (%)", 10, 40, 20)
        
        with col2:
            st.markdown("#### Target Variable")
            target_options = ['Frekuensi_AI', 'Ketergantungan', 'Nilai_Rata']
            target = st.selectbox("Pilih target", target_options)
            
            st.markdown("#### Fitur")
            feature_options = ['IPK', 'Jam_Belajar', 'Penggunaan_AI_Jam']
            selected_features = st.multiselect("Pilih fitur", feature_options, default=feature_options)
        
        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            with st.spinner("Training model..."):
                # Prepare data
                df = st.session_state.df
                if 'df_encoded' in st.session_state:
                    df = st.session_state.df_encoded
                    target_col = f"{target}_Encoded" if f"{target}_Encoded" in df.columns else target
                else:
                    target_col = target
                
                X = df[selected_features]
                y = df[target_col]
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=42
                )
                
                # Train model
                model, y_pred, accuracy = train_ai_model(X_train, X_test, y_train, y_test)
                
                # Save results
                st.session_state.model = model
                st.session_state.y_pred = y_pred
                st.session_state.y_test = y_test
                st.session_state.accuracy = accuracy
                st.session_state.X_test = X_test
                
                st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.2%}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        if 'model' not in st.session_state:
            st.warning("Silakan train model terlebih dahulu")
            return
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Hasil Analisis")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{st.session_state.accuracy:.2%}")
        with col2:
            st.metric("Total Predictions", len(st.session_state.y_pred))
        with col3:
            unique_preds = len(np.unique(st.session_state.y_pred))
            st.metric("Unique Categories", unique_preds)
        
        # Classification report
        st.markdown("#### 📋 Classification Report")
        report = classification_report(st.session_state.y_test, st.session_state.y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}"))
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 💡 Rekomendasi Otomatis")
        
        kb = AIKnowledgeBase()
        
        # Distribution analysis
        pred_counts = pd.Series(st.session_state.y_pred).value_counts().sort_index()
        
        # Map numeric predictions to categories
        category_map = {
            0: 'VERY_LOW',
            1: 'LOW',
            2: 'MEDIUM',
            3: 'HIGH',
            4: 'VERY_HIGH'
        }
        
        # Display recommendations for each category
        for pred_val, count in pred_counts.items():
            category_key = category_map.get(pred_val, 'MEDIUM')
            category_info = kb.get_category_info(category_key)
            
            if category_info:
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.markdown(f"<h1 style='color: {category_info['color']}; text-align: center;'>{category_info['icon']}</h1>", 
                                unsafe_allow_html=True)
                    st.markdown(f"<h3 style='text-align: center;'>{count} mahasiswa</h3>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"**{category_info['label']}** - Level: {category_info['level']}")
                    st.markdown(f"*{category_info['description']}*")
                    
                    with st.expander("Lihat detail rekomendasi"):
                        st.markdown("**Tindakan:**")
                        for action in category_info['actions']:
                            st.write(f"• {action}")
                        
                        st.markdown("**Rekomendasi:**")
                        for rec in category_info['recommendations']:
                            st.write(f"• {rec}")
                
                st.markdown("---")
        
        st.markdown("</div>", unsafe_allow_html=True)

def show_visualization():
    """Visualisasi data"""
    if 'df' not in st.session_state:
        st.warning("Silakan import data terlebih dahulu")
        return
    
    df = st.session_state.df
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Distribusi Penggunaan AI")
        
        fig = px.histogram(df, x='Frekuensi_AI', 
                         color='Frekuensi_AI',
                         title='Distribusi Frekuensi Penggunaan AI',
                         color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📈 Hubungan AI vs IPK")
        
        fig = px.scatter(df, x='Penggunaan_AI_Jam', y='IPK',
                       color='Frekuensi_AI',
                       size='Jam_Belajar',
                       hover_data=['Nama'],
                       title='Hubungan Penggunaan AI dengan IPK')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🏫 Analisis per Fakultas")
        
        faculty_stats = df.groupby('Fakultas').agg({
            'IPK': 'mean',
            'Penggunaan_AI_Jam': 'mean',
            'Jam_Belajar': 'mean'
        }).round(2)
        
        fig = go.Figure(data=[
            go.Bar(name='Rata IPK', x=faculty_stats.index, y=faculty_stats['IPK']),
            go.Bar(name='Penggunaan AI', x=faculty_stats.index, y=faculty_stats['Penggunaan_AI_Jam']/10)
        ])
        
        fig.update_layout(barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🔍 Heatmap Korelasi")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(corr_matrix,
                          labels=dict(color="Korelasi"),
                          x=corr_matrix.columns,
                          y=corr_matrix.columns,
                          color_continuous_scale='RdBu')
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

def show_knowledge_base():
    """Knowledge Base System"""
    kb = AIKnowledgeBase()
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### ℹ️ Knowledge Base System")
    st.markdown("Sistem pengetahuan untuk klasifikasi dan rekomendasi otomatis")
    
    # Tampilkan semua kategori
    for category_key in kb.get_all_categories():
        category_info = kb.get_category_info(category_key)
        
        st.markdown(f"""
        <div style='border-left: 4px solid {category_info["color"]}; 
                    padding-left: 15px; 
                    margin: 20px 0;'>
            <h3>{category_info['icon']} {category_info['label']}</h3>
            <p><strong>Level:</strong> {category_info['level']}</p>
            <p><strong>Skor Risiko:</strong> {category_info['score']}/5</p>
            <p><strong>Deskripsi:</strong> {category_info['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Tindakan yang Disarankan:**")
            for action in category_info['actions']:
                st.write(f"• {action}")
        
        with col2:
            st.markdown("**Rekomendasi untuk Mahasiswa:**")
            for rec in category_info['recommendations']:
                st.write(f"• {rec}")
        
        st.markdown("---")
    
    # Risk scale visualization
    st.markdown("### 📊 Skala Risiko")
    
    categories = kb.get_all_categories()
    colors = [kb.get_category_info(cat)['color'] for cat in categories]
    labels = [kb.get_category_info(cat)['label'] for cat in categories]
    scores = [kb.get_category_info(cat)['score'] for cat in categories]
    
    fig = create_progress_chart(labels, scores, colors)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def show_settings():
    """Pengaturan"""
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### ⚙️ Pengaturan Aplikasi")
    
    tab1, tab2 = st.tabs(["Umum", "Notifikasi"])
    
    with tab1:
        st.markdown("#### Pengaturan Umum")
        
        col1, col2 = st.columns(2)
        
        with col1:
            language = st.selectbox("Bahasa", ["Indonesia", "English"])
            theme = st.selectbox("Tema", ["Light", "Dark", "Auto"])
            rows_per_page = st.slider("Baris per halaman", 10, 100, 20)
        
        with col2:
            auto_save = st.checkbox("Auto-save changes", value=True)
            show_tutorial = st.checkbox("Show tutorial on startup", value=False)
            enable_analytics = st.checkbox("Enable analytics", value=True)
        
        if st.button("💾 Simpan Pengaturan", use_container_width=True):
            st.success("Pengaturan berhasil disimpan!")
    
    with tab2:
        st.markdown("#### Pengaturan Notifikasi")
        
        email_notif = st.checkbox("Email notifications", value=True)
        push_notif = st.checkbox("Push notifications", value=True)
        
        st.markdown("**Jenis notifikasi:**")
        new_data = st.checkbox("New data uploaded", value=True)
        model_trained = st.checkbox("Model training completed", value=True)
        high_risk = st.checkbox("High-risk students detected", value=True)
        
        frequency = st.selectbox("Frequency", ["Real-time", "Daily", "Weekly"])
        
        if st.button("💾 Simpan Notifikasi", use_container_width=True):
            st.success("Pengaturan notifikasi berhasil disimpan!")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================
# DASHBOARD MAHASISWA
# ============================================
def mahasiswa_dashboard():
    """Dashboard mahasiswa"""
    # Sidebar mahasiswa
    with st.sidebar:
        if 'student_name' in st.session_state:
            st.markdown(f"<h2>👨‍🎓 {st.session_state.student_name}</h2>", unsafe_allow_html=True)
        
        st.markdown("### 📊 Menu Mahasiswa")
        
        menu_option = st.selectbox(
            "Navigasi",
            ["🏠 Dashboard Saya", "📈 Analisis AI", "📋 Rekomendasi", "🎯 Tips & Panduan", "⚙️ Profil"]
        )
        
        st.markdown("---")
        st.markdown("### 📅 Quick Info")
        st.info("**Deadline:** Tugas AI Ethics - 30 Nov 2024")
        st.warning("**Warning:** Penggunaan AI Anda dalam kategori Waspada")
        
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()
    
    # Main content
    if menu_option == "🏠 Dashboard Saya":
        show_student_dashboard()
    elif menu_option == "📈 Analisis AI":
        show_student_analysis()
    elif menu_option == "📋 Rekomendasi":
        show_student_recommendations()
    elif menu_option == "🎯 Tips & Panduan":
        show_student_guides()
    elif menu_option == "⚙️ Profil":
        show_student_profile()

def show_student_dashboard():
    """Dashboard utama mahasiswa"""
    st.markdown(f"<h1>👋 Selamat Datang, {st.session_state.get('student_name', 'Mahasiswa')}!</h1>", unsafe_allow_html=True)
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_metric_card("IPK Saat Ini", "3.42", "+0.15", "📊")
    with col2:
        create_metric_card("Penggunaan AI", "22 jam", "↑ 5%", "🤖")
    with col3:
        create_metric_card("Jam Belajar", "28 jam", "↑ 2%", "📚")
    with col4:
        create_metric_card("Level Risiko", "Waspada", "", "⚠️")
    
    # Progress cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📈 Progress Akademik")
        
        # Create progress chart
        labels = ['Sem 1', 'Sem 2', 'Sem 3', 'Sem 4']
        values = [2.8, 3.0, 3.2, 3.4]
        
        fig = go.Figure(data=[
            go.Scatter(
                x=labels,
                y=values,
                mode='lines+markers',
                line=dict(color='#4e54c8', width=3),
                marker=dict(size=10)
            )
        ])
        
        fig.update_layout(
            height=200,
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🤖 Penggunaan AI")
        
        # AI usage by purpose
        data = {
            'Purpose': ['Tugas', 'Penelitian', 'Belajar', 'Lainnya'],
            'Hours': [12, 5, 3, 2]
        }
        
        fig = px.pie(data, values='Hours', names='Purpose', 
                     color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(height=200, showlegend=False)
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Recent activities
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📋 Aktivitas Terkini")
    
    activities = [
        {"time": "Hari ini", "action": "Menggunakan AI untuk tugas matematika", "duration": "2 jam"},
        {"time": "Kemarin", "action": "Konsultasi dengan dosen tentang penggunaan AI", "status": "✅ Selesai"},
        {"time": "2 hari lalu", "action": "Workshop AI Ethics", "status": "🎓 Diikuti"},
        {"time": "Minggu lalu", "action": "Evaluasi penggunaan AI pribadi", "status": "📊 Dilaporkan"}
    ]
    
    for act in activities:
        col1, col2, col3 = st.columns([2, 3, 2])
        with col1:
            st.write(f"**{act['time']}**")
        with col2:
            st.write(act['action'])
        with col3:
            st.write(act.get('duration', act.get('status', '')))
    st.markdown("</div>", unsafe_allow_html=True)

def show_student_analysis():
    """Analisis untuk mahasiswa"""
    st.markdown("<h1>📈 Analisis Penggunaan AI Saya</h1>", unsafe_allow_html=True)
    
    # Input data pribadi
    with st.form("student_analysis_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            ipk = st.slider("IPK Saat Ini", 2.0, 4.0, 3.4, 0.1)
            study_hours = st.slider("Jam Belajar/Minggu", 5, 60, 25)
        
        with col2:
            ai_hours = st.slider("Jam Penggunaan AI/Minggu", 0, 40, 15)
            dependency = st.select_slider(
                "Tingkat Ketergantungan pada AI",
                options=['Sangat Rendah', 'Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi'],
                value='Sedang'
            )
        
        submitted = st.form_submit_button("🔍 Analisis Profil Saya", use_container_width=True)
    
    if submitted:
        # Analyze based on inputs
        kb = AIKnowledgeBase()
        
        # Determine category based on AI hours
        if ai_hours < 5:
            category = 'VERY_LOW'
        elif ai_hours < 15:
            category = 'LOW'
        elif ai_hours < 25:
            category = 'MEDIUM'
        elif ai_hours < 35:
            category = 'HIGH'
        else:
            category = 'VERY_HIGH'
        
        category_info = kb.get_category_info(category)
        
        # Display results
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
            <div style='background: {category_info['color']}20; 
                        padding: 30px; 
                        border-radius: 15px; 
                        text-align: center;'>
                <h1 style='font-size: 4rem; margin: 0;'>{category_info['icon']}</h1>
                <h2 style='color: {category_info['color']}; margin: 10px 0;'>{category_info['label']}</h2>
                <h3>Level: {category_info['level']}</h3>
                <p>Skor Risiko: {category_info['score']}/5</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"### 📋 Hasil Analisis")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("IPK", f"{ipk:.2f}")
            with col2:
                st.metric("Jam Belajar", f"{study_hours} jam")
            with col3:
                st.metric("Jam AI", f"{ai_hours} jam")
            
            st.markdown(f"**Deskripsi:** {category_info['description']}")
            
            with st.expander("📝 Detail Rekomendasi"):
                st.markdown("**Tindakan yang Disarankan:**")
                for action in category_info['actions']:
                    st.write(f"• {action}")
                
                st.markdown("**Rekomendasi Khusus untuk Anda:**")
                for rec in category_info['recommendations']:
                    st.write(f"• {rec}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Visualization
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Visualisasi Data")
        
        fig = go.Figure()
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=ai_hours,
            title={'text': "Jam AI/Minggu"},
            domain={'x': [0, 0.33], 'y': [0, 1]},
            gauge={'axis': {'range': [0, 40]},
                   'bar': {'color': category_info['color']},
                   'steps': [
                       {'range': [0, 10], 'color': "#28a74520"},
                       {'range': [10, 20], 'color': "#ffc10720"},
                       {'range': [20, 40], 'color': "#dc354520"}
                   ]}
        ))
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=ipk,
            title={'text': "IPK"},
            domain={'x': [0.34, 0.66], 'y': [0, 1]},
            gauge={'axis': {'range': [2, 4]},
                   'bar': {'color': '#4e54c8'}}
        ))
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=study_hours,
            title={'text': "Jam Belajar"},
            domain={'x': [0.67, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [0, 60]},
                   'bar': {'color': '#20c997'}}
        ))
        
        fig.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

def show_student_recommendations():
    """Rekomendasi untuk mahasiswa"""
    st.markdown("<h1>💡 Rekomendasi Personal</h1>", unsafe_allow_html=True)
    
    kb = AIKnowledgeBase()
    
    # Simulate student data
    student_category = 'MEDIUM'  # This would come from actual analysis
    
    # Get recommendations
    category_info = kb.get_category_info(student_category)
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {category_info['color']}20 0%, #ffffff 100%);
                padding: 30px;
                border-radius: 15px;
                margin-bottom: 20px;'>
        <h2>{category_info['icon']} Status Anda: {category_info['label']}</h2>
        <p style='font-size: 1.2rem;'>{category_info['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Action plan
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🎯 Rencana Aksi")
    
    action_plan = [
        {
            "title": "Minggu 1-2",
            "actions": [
                "Kurangi penggunaan AI untuk tugas sederhana",
                "Catat semua penggunaan AI dalam jurnal",
                "Ikuti workshop 'Belajar Tanpa AI'"
            ]
        },
        {
            "title": "Minggu 3-4",
            "actions": [
                "Batasi penggunaan AI maksimal 2 jam/hari",
                "Konsultasi dengan dosen pembimbing",
                "Coba metode belajar tradisional"
            ]
        },
        {
            "title": "Minggu 5-8",
            "actions": [
                "Evaluasi progress pengurangan ketergantungan",
                "Tingkatkan kemampuan problem-solving mandiri",
                "Bagikan pengalaman dengan teman"
            ]
        }
    ]
    
    for week in action_plan:
        with st.expander(f"📅 {week['title']}"):
            for action in week['actions']:
                st.write(f"• {action}")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Resources
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📚 Sumber Daya yang Direkomendasikan")
    
    resources = [
        {"title": "📘 Buku: 'Digital Minimalism'", "type": "Bacaan", "duration": "2 minggu"},
        {"title": "🎥 Video: 'Ethical AI Usage'", "type": "Video", "duration": "45 menit"},
        {"title": "👨‍🏫 Workshop: 'Critical Thinking Skills'", "type": "Event", "date": "15 Nov 2024"},
        {"title": "🔄 App: 'Digital Detox Tracker'", "type": "Aplikasi", "platform": "Android/iOS"}
    ]
    
    for res in resources:
        col1, col2, col3 = st.columns([3, 2, 2])
        with col1:
            st.write(f"**{res['title']}**")
        with col2:
            st.write(res['type'])
        with col3:
            st.write(res.get('duration', res.get('date', res.get('platform', ''))))
    
    st.markdown("</div>", unsafe_allow_html=True)

def show_student_guides():
    """Tips dan panduan"""
    st.markdown("<h1>🎯 Tips & Panduan</h1>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📖 Panduan AI", "💡 Tips Belajar", "🛡️ Keamanan Digital"])
    
    with tab1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📖 Panduan Penggunaan AI yang Sehat")
        
        guides = [
            {
                "title": "1. Gunakan AI sebagai Asisten, Bukan Pengganti",
                "content": "AI seharusnya membantu proses belajar, bukan menggantikan pemikiran kritis Anda."
            },
            {
                "title": "2. Verifikasi Semua Hasil AI",
                "content": "Selalu cek kebenaran informasi dari AI dengan sumber terpercaya."
            },
            {
                "title": "3. Tetap Kembangkan Skill Dasar",
                "content": "Jangan biarkan AI membuat Anda kehilangan kemampuan fundamental."
            },
            {
                "title": "4. Atur Batas Waktu",
                "content": "Tetapkan waktu maksimal penggunaan AI per hari untuk mencegah ketergantungan."
            },
            {
                "title": "5. Diskusikan dengan Dosen",
                "content": "Konsultasikan penggunaan AI Anda dengan dosen untuk mendapatkan panduan yang tepat."
            }
        ]
        
        for guide in guides:
            st.markdown(f"**{guide['title']}**")
            st.write(guide['content'])
            st.write("")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 💡 Tips Belajar Efektif")
        
        tips = [
            "🎯 **Fokus pada Pemahaman**, bukan hafalan",
            "⏰ **Gunakan Teknik Pomodoro**: 25 menit belajar, 5 menit istirahat",
            "📝 **Buat Catatan Manual** untuk meningkatkan daya ingat",
            "🔄 **Review Materi** secara berkala",
            "👥 **Belajar Kelompok** untuk diskusi dan sharing",
            "🏃 **Jaga Kesehatan** dengan olahraga teratur",
            "💤 **Istirahat Cukup** untuk optimalisasi memori"
        ]
        
        for tip in tips:
            st.write(tip)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🛡️ Keamanan Digital")
        
        st.markdown("**Privasi Data:**")
        st.write("• Jangan bagikan data pribadi ke AI")
        st.write("• Gunakan akun terpisah untuk aktivitas akademik")
        st.write("• Periksa kebijakan privasi tools AI yang digunakan")
        
        st.markdown("**Keamanan Akademik:**")
        st.write("• Pahami kebijakan plagiarisme kampus")
        st.write("• Simpan semua percakapan dengan AI sebagai bukti")
        st.write("• Laporkan penyalahgunaan AI yang Anda temui")
        
        st.markdown("**Etika Digital:**")
        st.write("• Hormati hak cipta dan kekayaan intelektual")
        st.write("• Gunakan AI untuk kebaikan, bukan kecurangan")
        st.write("• Berkontribusi positif pada komunitas digital")
        
        st.markdown("</div>", unsafe_allow_html=True)

def show_student_profile():
    """Profil mahasiswa"""
    st.markdown("<h1>⚙️ Profil Saya</h1>", unsafe_allow_html=True)
    
    with st.form("student_profile_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            nama = st.text_input("Nama Lengkap", value=st.session_state.get('student_name', ''))
            nim = st.text_input("NIM", value="20230001")
            email = st.text_input("Email", value="mahasiswa@example.com")
        
        with col2:
            fakultas = st.selectbox("Fakultas", ["Teknik", "Sains", "Ekonomi", "Kedokteran", "Hukum"])
            program_studi = st.text_input("Program Studi", value="Informatika")
            semester = st.slider("Semester", 1, 8, 5)
        
        # Preferences
        st.markdown("### ⚙️ Preferensi")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            notif_email = st.checkbox("Notifikasi Email", value=True)
            notif_push = st.checkbox("Notifikasi Push", value=True)
        
        with col2:
            dark_mode = st.checkbox("Mode Gelap", value=False)
            auto_save = st.checkbox("Auto-save", value=True)
        
        with col3:
            language = st.selectbox("Bahasa", ["Indonesia", "English"])
            timezone = st.selectbox("Zona Waktu", ["WIB", "WITA", "WIT"])
        
        submitted = st.form_submit_button("💾 Simpan Perubahan", use_container_width=True)
    
    if submitted:
        st.success("✅ Profil berhasil diperbarui!")
        
        # Save to session state
        st.session_state.student_name = nama
    
    # Account security
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🔒 Keamanan Akun")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Ganti Password", use_container_width=True):
            st.info("Fitur ganti password akan segera tersedia")
    
    with col2:
        if st.button("📱 Verifikasi 2FA", use_container_width=True):
            st.info("Fitur 2FA akan segera tersedia")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Data export
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📤 Ekspor Data")
    
    if st.button("📥 Download Data Saya", use_container_width=True):
        # Create sample data
        data = {
            'Nama': [nama],
            'NIM': [nim],
            'Fakultas': [fakultas],
            'Program Studi': [program_studi],
            'Semester': [semester],
            'Email': [email]
        }
        
        df = pd.DataFrame(data)
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="⬇️ Download sebagai CSV",
            data=csv,
            file_name="profil_mahasiswa.csv",
            mime="text/csv"
        )
    
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================
# APLIKASI UTAMA
# ============================================
def main():
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'role' not in st.session_state:
        st.session_state.role = None
    
    # Show appropriate page
    if not st.session_state.logged_in:
        login_page()
    else:
        if st.session_state.role == "dosen":
            dosen_dashboard()
        else:
            mahasiswa_dashboard()

if __name__ == "__main__":
    main()
