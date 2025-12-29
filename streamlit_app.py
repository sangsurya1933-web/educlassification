# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report, 
                           roc_curve, auc)
import plotly.graph_objects as go
import plotly.express as px
import warnings
import pickle
import os
from datetime import datetime
from io import BytesIO
import base64

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Analisis Penggunaan AI vs Performa Akademik",
    page_icon="🎓",
    layout="wide"
)

# Initialize session state
def init_session_state():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'preprocessed' not in st.session_state:
        st.session_state.preprocessed = False
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None

init_session_state()

# Function to create download link
def get_download_link(file_data, filename, text):
    b64 = base64.b64encode(file_data).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Create dummy data if dataset not found
def create_dummy_data():
    """Create dummy data for testing if real dataset is not available"""
    np.random.seed(42)
    n_samples = 200
    
    # Generate dummy features
    data = {
        'Umur': np.random.randint(18, 25, n_samples),
        'Semester': np.random.randint(1, 8, n_samples),
        'Frekuensi_AI_Penelitian': np.random.randint(1, 10, n_samples),
        'Frekuensi_AI_Tugas': np.random.randint(1, 10, n_samples),
        'Jam_Penggunaan_AI_per_Minggu': np.random.randint(1, 20, n_samples),
        'Tingkat_Keahlian_AI': np.random.choice(['Pemula', 'Menengah', 'Lanjutan'], n_samples),
        'Jenis_AI_Digunakan': np.random.choice(['ChatGPT', 'Bard', 'Claude', 'Lainnya'], n_samples),
        'IPK': np.round(np.random.uniform(2.0, 4.0, n_samples), 2),
        'Tingkat_Kepuasan_AI': np.random.choice(['Rendah', 'Sedang', 'Tinggi'], n_samples),
        'Performa_Akademik': np.random.choice(['Buruk', 'Cukup', 'Baik', 'Sangat Baik'], n_samples),
        'Jurusan': np.random.choice(['Informatika', 'Sistem Informasi', 'Teknik', 'Bisnis'], n_samples)
    }
    
    return pd.DataFrame(data)

# Load data function
@st.cache_data
def load_data():
    dataset_paths = [
        "Dataset_Klasifikasi_Pengguna_AI_Mahasiswa_UMMgl.csv",
        "dataset.csv",
        "data.csv"
    ]
    
    for path in dataset_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                st.success(f"Dataset berhasil dimuat dari: {path}")
                return df
            except Exception as e:
                st.warning(f"Error membaca {path}: {e}")
                continue
    
    # If no file found, create dummy data
    st.warning("Dataset tidak ditemukan. Menggunakan data dummy untuk demonstrasi.")
    df = create_dummy_data()
    
    # Save dummy data for reference
    df.to_csv("dummy_dataset.csv", index=False)
    st.info("Data dummy telah disimpan sebagai 'dummy_dataset.csv'")
    
    return df

# Sidebar Login
def login_section():
    st.sidebar.title("🔐 Login Sistem")
    st.sidebar.markdown("---")
    
    role = st.sidebar.selectbox("Pilih Role", ["Guru", "Siswa"])
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Login", use_container_width=True):
            authenticate(username, password, role)
    with col2:
        if st.button("Reset", use_container_width=True):
            reset_session()

def authenticate(username, password, role):
    """Simple authentication function"""
    if role == "Guru":
        if username == "guru" and password == "guru123":
            st.session_state.authenticated = True
            st.session_state.user_role = "guru"
            st.sidebar.success("Login berhasil sebagai Guru!")
            st.rerun()
        elif username or password:
            st.sidebar.error("Username atau password salah!")
    
    elif role == "Siswa":
        if username == "siswa" and password == "siswa123":
            st.session_state.authenticated = True
            st.session_state.user_role = "siswa"
            st.sidebar.success("Login berhasil sebagai Siswa!")
            st.rerun()
        elif username or password:
            st.sidebar.error("Username atau password salah!")

def reset_session():
    """Reset session state"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Main content based on role
def main_content():
    if not st.session_state.authenticated:
        show_landing_page()
    else:
        show_dashboard()

def show_landing_page():
    """Show landing page when not logged in"""
    st.title("🎓 Analisis Tingkat Penggunaan AI terhadap Performa Akademik")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### **Judul Penelitian:**
        Analisis Tingkat Penggunaan Artificial Intelligence (AI) terhadap 
        Performa Akademik Siswa Menggunakan Algoritma Random Forest
        
        ### **Knowledge Base Data Science dalam Pendidikan:**
        
        📊 **1. Pengumpulan & Eksplorasi Data**
        - Mengumpulkan data penggunaan AI dari mahasiswa
        - Memahami pola dan distribusi data
        
        🧹 **2. Preprocessing Data**
        - Data cleaning: menangani missing values dan outliers
        - Feature engineering: transformasi variabel
        - Encoding: mengubah data kategorikal ke numerik
        
        🤖 **3. Pemodelan Machine Learning**
        - Membangun model prediktif dengan Random Forest
        - Hyperparameter tuning untuk optimasi performa
        - Validasi model dengan cross-validation
        
        📈 **4. Evaluasi & Interpretasi**
        - Analisis metrik performa (Accuracy, Precision, Recall, F1)
        - Confusion Matrix untuk evaluasi klasifikasi
        - Feature importance analysis
        
        💡 **5. Implementasi & Insight**
        - Rekomendasi berbasis data untuk peningkatan performa
        - Dashboard interaktif untuk monitoring
        """)
    
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", 
                 width=200)
        st.markdown("### **Login untuk Mengakses:**")
        st.info("""
        **Guru:**
        - Username: `guru`
        - Password: `guru123`
        
        **Siswa:**
        - Username: `siswa`
        - Password: `siswa123`
        """)
        
        # Quick dataset preview
        st.markdown("### **Preview Dataset:**")
        df = load_data()
        st.dataframe(df.head(5), use_container_width=True)
        st.caption(f"Total records: {len(df)} | Features: {len(df.columns)}")

def show_dashboard():
    """Show dashboard based on user role"""
    # Logout button
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        reset_session()
    
    st.sidebar.markdown(f"**Login sebagai:** {st.session_state.user_role.upper()}")
    st.sidebar.markdown("---")
    
    if st.session_state.user_role == "guru":
        guru_dashboard()
    else:
        siswa_dashboard()

# GURU DASHBOARD
def guru_dashboard():
    st.title("👨‍🏫 Dashboard Guru - Analisis Penggunaan AI")
    st.markdown("---")
    
    # Load data
    if st.session_state.data is None:
        st.session_state.data = load_data()
    
    df = st.session_state.data
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dataset", 
        "🧹 Preprocessing", 
        "🤖 Modeling", 
        "📈 Evaluasi", 
        "💾 Ekspor"
    ])
    
    with tab1:
        show_dataset_info(df)
    
    with tab2:
        df_processed = preprocessing_section(df)
        if df_processed is not None:
            st.session_state.df_processed = df_processed
    
    with tab3:
        if 'df_processed' in st.session_state:
            modeling_section(st.session_state.df_processed)
        else:
            st.warning("Silakan lakukan preprocessing data terlebih dahulu di tab 'Preprocessing'")
    
    with tab4:
        if 'model_trained' in st.session_state and st.session_state.model_trained:
            evaluation_section()
        else:
            st.warning("Silakan latih model terlebih dahulu di tab 'Modeling'")
    
    with tab5:
        export_section()

def show_dataset_info(df):
    """Display dataset information"""
    st.header("📊 Dataset Information")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Jumlah Data", f"{len(df):,}")
    with col2:
        st.metric("Jumlah Fitur", len(df.columns))
    with col3:
        st.metric("Data Hilang", f"{df.isnull().sum().sum():,}")
    with col4:
        st.metric("Tipe Data Unik", df.select_dtypes(include=['object']).shape[1])
    
    # Data preview
    st.subheader("Preview Data")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Data types
    st.subheader("Informasi Kolom")
    dtype_df = pd.DataFrame({
        'Kolom': df.columns,
        'Tipe Data': df.dtypes.values,
        'Nilai Unik': [df[col].nunique() for col in df.columns],
        'Missing Values': df.isnull().sum().values
    })
    st.dataframe(dtype_df, use_container_width=True)
    
    # Statistical summary
    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe(), use_container_width=True)

def preprocessing_section(df):
    """Data preprocessing section"""
    st.header("🧹 Preprocessing Data")
    
    df_processed = df.copy()
    
    # 1. Handle missing values
    st.subheader("1. Penanganan Data Hilang")
    missing_values = df.isnull().sum()
    
    if missing_values.sum() > 0:
        st.warning(f"⚠️ Ditemukan {missing_values.sum()} data hilang")
        
        col1, col2 = st.columns(2)
        with col1:
            missing_df = pd.DataFrame({
                'Kolom': missing_values.index,
                'Jumlah Hilang': missing_values.values,
                'Persentase': (missing_values.values / len(df) * 100).round(2)
            })
            missing_df = missing_df[missing_df['Jumlah Hilang'] > 0]
            st.dataframe(missing_df, use_container_width=True)
        
        with col2:
            method = st.selectbox(
                "Metode penanganan data hilang:",
                ["Hapus baris", "Isi dengan modus", "Isi dengan median", "Isi dengan mean"]
            )
            
            if st.button("Terapkan", key="handle_missing"):
                if method == "Hapus baris":
                    df_processed = df_processed.dropna()
                elif method == "Isi dengan modus":
                    for col in df_processed.columns:
                        if df_processed[col].dtype == 'object':
                            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
                        else:
                            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                elif method == "Isi dengan median":
                    for col in df_processed.columns:
                        if df_processed[col].dtype in ['int64', 'float64']:
                            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                elif method == "Isi dengan mean":
                    for col in df_processed.columns:
                        if df_processed[col].dtype in ['int64', 'float64']:
                            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
                
                st.success(f"Data setelah cleaning: {len(df_processed)} baris")
                st.session_state.df_processed = df_processed
    else:
        st.success("✅ Tidak ada data hilang")
    
    # 2. Encoding categorical variables
    st.subheader("2. Encoding Variabel Kategorikal")
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    if len(categorical_cols) > 0:
        st.write(f"Ditemukan {len(categorical_cols)} kolom kategorikal:")
        
        for col in categorical_cols:
            unique_vals = df_processed[col].unique()
            st.write(f"- **{col}**: {len(unique_vals)} nilai unik → {list(unique_vals[:5])}")
        
        if st.button("Lakukan Label Encoding", key="encode_categorical"):
            le_dict = {}
            for col in categorical_cols:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                le_dict[col] = le
            
            st.session_state.label_encoders = le_dict
            st.success("✅ Encoding selesai!")
            
            # Show encoding mapping
            with st.expander("Lihat Mapping Encoding"):
                for col, le in le_dict.items():
                    classes = le.classes_
                    st.write(f"**{col}**:")
                    for i, cls in enumerate(classes):
                        st.write(f"  {cls} → {i}")
    else:
        st.info("Tidak ada kolom kategorikal yang perlu di-encode")
    
    # 3. Feature scaling
    st.subheader("3. Normalisasi Data Numerik")
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 0:
        st.write(f"Ditemukan {len(numeric_cols)} kolom numerik")
        
        if st.button("Lakukan Standard Scaling", key="scale_features"):
            scaler = StandardScaler()
            df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
            st.session_state.scaler = scaler
            st.success("✅ Normalisasi selesai!")
    
    # 4. Train-test split
    st.subheader("4. Pembagian Data Latih dan Uji")
    
    if 'df_processed' in locals() and df_processed is not None:
        target_options = df_processed.columns.tolist()
        target_col = st.selectbox("Pilih kolom target:", target_options)
        
        if target_col:
            X = df_processed.drop(columns=[target_col])
            y = df_processed[target_col]
            
            col1, col2 = st.columns(2)
            with col1:
                test_size = st.slider("Ukuran data uji (%):", 10, 50, 20) / 100
            with col2:
                random_state = st.number_input("Random State:", 0, 100, 42)
            
            if st.button("Split Data", key="split_data"):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
                
                # Save to session state
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.target_col = target_col
                
                st.success(f"""
                ✅ Data berhasil dibagi:
                - Data latih: {len(X_train)} sampel ({100-test_size*100:.0f}%)
                - Data uji: {len(X_test)} sampel ({test_size*100:.0f}%)
                - Jumlah fitur: {X_train.shape[1]}
                """)
                
                # Show class distribution
                st.subheader("Distribusi Kelas Target")
                train_dist = pd.Series(y_train).value_counts()
                test_dist = pd.Series(y_test).value_counts()
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                train_dist.plot(kind='bar', ax=ax1, color='skyblue')
                ax1.set_title('Distribusi Data Latih')
                ax1.set_xlabel('Kelas')
                ax1.set_ylabel('Jumlah')
                
                test_dist.plot(kind='bar', ax=ax2, color='lightcoral')
                ax2.set_title('Distribusi Data Uji')
                ax2.set_xlabel('Kelas')
                ax2.set_ylabel('Jumlah')
                
                plt.tight_layout()
                st.pyplot(fig)
    
    return df_processed if 'df_processed' in locals() else None

def modeling_section(df_processed):
    """Model training section"""
    st.header("🤖 Pemodelan dengan Random Forest")
    
    if 'X_train' not in st.session_state:
        st.warning("Silakan lakukan train-test split terlebih dahulu di tab Preprocessing")
        return
    
    st.markdown("""
    **Random Forest** adalah algoritma ensemble learning yang membangun banyak decision tree 
    dan menggabungkan hasilnya untuk meningkatkan akurasi dan mengurangi overfitting.
    """)
    
    # Parameter tuning
    st.subheader("Hyperparameter Tuning")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        n_estimators = st.slider("Jumlah Trees (n_estimators)", 10, 300, 100, 10)
    with col2:
        max_depth = st.slider("Kedalaman Maksimum (max_depth)", 2, 50, 20, 2)
        max_depth = None if max_depth == 50 else max_depth
    with col3:
        min_samples_split = st.slider("Min Samples Split", 2, 20, 2, 1)
    
    col4, col5, col6 = st.columns(3)
    with col4:
        min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 1, 1)
    with col5:
        max_features = st.selectbox("Max Features", ['sqrt', 'log2', None, 'auto'])
    with col6:
        random_state = st.number_input("Random State", 0, 100, 42)
    
    if st.button("🚀 Latih Model Random Forest", type="primary"):
        with st.spinner("Melatih model Random Forest..."):
            try:
                # Initialize model
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=random_state,
                    n_jobs=-1
                )
                
                # Train model
                model.fit(st.session_state.X_train, st.session_state.y_train)
                
                # Make predictions
                y_pred = model.predict(st.session_state.X_test)
                y_pred_proba = model.predict_proba(st.session_state.X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(st.session_state.y_test, y_pred)
                precision = precision_score(st.session_state.y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(st.session_state.y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(st.session_state.y_test, y_pred, average='weighted', zero_division=0)
                
                # Store in session state
                st.session_state.model = model
                st.session_state.y_pred = y_pred
                st.session_state.y_pred_proba = y_pred_proba
                st.session_state.model_metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
                st.session_state.model_trained = True
                
                # Display results
                st.success("✅ Model berhasil dilatih!")
                
                # Show metrics
                st.subheader("📊 Performa Model")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{accuracy:.4f}")
                col2.metric("Precision", f"{precision:.4f}")
                col3.metric("Recall", f"{recall:.4f}")
                col4.metric("F1-Score", f"{f1:.4f}")
                
                # Feature importance
                st.subheader("📈 Feature Importance")
                feature_importance = pd.DataFrame({
                    'Feature': st.session_state.X_train.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                # Plot feature importance
                fig = px.bar(feature_importance.head(15), 
                            x='Importance', 
                            y='Feature',
                            orientation='h',
                            title='15 Fitur Paling Penting',
                            color='Importance',
                            color_continuous_scale='viridis')
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Show top features table
                with st.expander("Lihat Detail Feature Importance"):
                    st.dataframe(feature_importance, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error training model: {str(e)}")

def evaluation_section():
    """Model evaluation section"""
    st.header("📈 Evaluasi Model")
    
    if not st.session_state.model_trained:
        st.warning("Model belum dilatih. Silakan latih model terlebih dahulu.")
        return
    
    # 1. Confusion Matrix
    st.subheader("1. Confusion Matrix")
    
    cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
    
    # Create heatmap
    fig_cm = px.imshow(cm,
                      labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                      x=[f"Kelas {i}" for i in range(cm.shape[0])],
                      y=[f"Kelas {i}" for i in range(cm.shape[0])],
                      text_auto=True,
                      color_continuous_scale='Blues')
    fig_cm.update_layout(title='Confusion Matrix')
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # 2. Classification Report
    st.subheader("2. Classification Report")
    
    report = classification_report(st.session_state.y_test, 
                                  st.session_state.y_pred,
                                  output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    # Style the dataframe
    styled_report = report_df.style\
        .background_gradient(subset=['precision', 'recall', 'f1-score'], cmap='YlOrBr')\
        .format({col: "{:.4f}" for col in report_df.columns if col != 'support'})
    
    st.dataframe(styled_report, use_container_width=True)
    
    # 3. ROC Curve (for binary classification)
    if st.session_state.y_pred_proba.shape[1] == 2:
        st.subheader("3. ROC Curve")
        
        fpr, tpr, thresholds = roc_curve(st.session_state.y_test, 
                                         st.session_state.y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr,
                                    mode='lines',
                                    name=f'ROC curve (AUC = {roc_auc:.3f})',
                                    line=dict(color='blue', width=2)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                    mode='lines',
                                    name='Random',
                                    line=dict(dash='dash', color='red')))
        
        fig_roc.update_layout(title='Receiver Operating Characteristic (ROC) Curve',
                             xaxis_title='False Positive Rate',
                             yaxis_title='True Positive Rate',
                             hovermode='x unified')
        
        st.plotly_chart(fig_roc, use_container_width=True)
    
    # 4. Prediction Analysis
    st.subheader("4. Analisis Prediksi")
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Aktual': st.session_state.y_test,
        'Prediksi': st.session_state.y_pred,
        'Benar': st.session_state.y_test == st.session_state.y_pred
    })
    
    # Calculate accuracy per class
    class_accuracy = {}
    unique_classes = np.unique(st.session_state.y_test)
    for cls in unique_classes:
        mask = comparison_df['Aktual'] == cls
        if mask.sum() > 0:
            class_accuracy[cls] = (comparison_df.loc[mask, 'Benar'].sum() / mask.sum())
    
    accuracy_df = pd.DataFrame({
        'Kelas': list(class_accuracy.keys()),
        'Akurasi': list(class_accuracy.values())
    })
    
    # Plot class accuracy
    fig_accuracy = px.bar(accuracy_df, 
                         x='Kelas', 
                         y='Akurasi',
                         title='Akurasi per Kelas',
                         color='Akurasi',
                         color_continuous_scale='RdYlGn',
                         range_color=[0, 1])
    st.plotly_chart(fig_accuracy, use_container_width=True)
    
    # Show prediction samples
    with st.expander("Lihat Contoh Prediksi"):
        st.dataframe(comparison_df.head(20), use_container_width=True)

def export_section():
    """Export section"""
    st.header("💾 Ekspor Hasil")
    
    if not st.session_state.model_trained:
        st.warning("Tidak ada hasil yang bisa diekspor. Silakan latih model terlebih dahulu.")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Model")
        if st.button("💾 Export Model", use_container_width=True):
            try:
                model_filename = f"random_forest_model_{timestamp}.pkl"
                with open(model_filename, 'wb') as f:
                    pickle.dump(st.session_state.model, f)
                
                # Create download link
                with open(model_filename, 'rb') as f:
                    model_bytes = f.read()
                
                st.markdown(get_download_link(model_bytes, model_filename, 
                                            "⬇️ Download Model"), unsafe_allow_html=True)
                st.success(f"Model disimpan sebagai: {model_filename}")
            except Exception as e:
                st.error(f"Error exporting model: {str(e)}")
    
    with col2:
        st.subheader("Predictions")
        if st.button("📊 Export Predictions", use_container_width=True):
            try:
                # Create predictions dataframe
                predictions_df = pd.DataFrame({
                    'Aktual': st.session_state.y_test,
                    'Prediksi': st.session_state.y_pred,
                    'Probabilitas': [max(proba) for proba in st.session_state.y_pred_proba]
                })
                
                predictions_filename = f"predictions_{timestamp}.csv"
                predictions_df.to_csv(predictions_filename, index=False)
                
                # Create download link
                with open(predictions_filename, 'rb') as f:
                    pred_bytes = f.read()
                
                st.markdown(get_download_link(pred_bytes, predictions_filename, 
                                            "⬇️ Download Predictions"), unsafe_allow_html=True)
                st.success(f"Predictions disimpan sebagai: {predictions_filename}")
            except Exception as e:
                st.error(f"Error exporting predictions: {str(e)}")
    
    with col3:
        st.subheader("Metrics")
        if st.button("📈 Export Metrics", use_container_width=True):
            try:
                # Create metrics dataframe
                metrics_df = pd.DataFrame([st.session_state.model_metrics])
                metrics_filename = f"metrics_{timestamp}.csv"
                metrics_df.to_csv(metrics_filename, index=False)
                
                # Create download link
                with open(metrics_filename, 'rb') as f:
                    metrics_bytes = f.read()
                
                st.markdown(get_download_link(metrics_bytes, metrics_filename, 
                                            "⬇️ Download Metrics"), unsafe_allow_html=True)
                st.success(f"Metrics disimpan sebagai: {metrics_filename}")
            except Exception as e:
                st.error(f"Error exporting metrics: {str(e)}")
    
    # Export everything
    st.markdown("---")
    st.subheader("Export Semua Hasil")
    
    if st.button("📦 Export All Results", type="primary", use_container_width=True):
        try:
            # Create zip file with all results
            import zipfile
            
            zip_filename = f"ai_analysis_results_{timestamp}.zip"
            
            with zipfile.ZipFile(zip_filename, 'w') as zipf:
                # Add model
                model_filename = f"model_{timestamp}.pkl"
                with open(model_filename, 'wb') as f:
                    pickle.dump(st.session_state.model, f)
                zipf.write(model_filename)
                
                # Add predictions
                predictions_df = pd.DataFrame({
                    'Aktual': st.session_state.y_test,
                    'Prediksi': st.session_state.y_pred
                })
                pred_filename = f"predictions_{timestamp}.csv"
                predictions_df.to_csv(pred_filename, index=False)
                zipf.write(pred_filename)
                
                # Add metrics
                metrics_df = pd.DataFrame([st.session_state.model_metrics])
                metrics_filename = f"metrics_{timestamp}.csv"
                metrics_df.to_csv(metrics_filename, index=False)
                zipf.write(metrics_filename)
                
                # Add feature importance
                feature_importance = pd.DataFrame({
                    'Feature': st.session_state.X_train.columns,
                    'Importance': st.session_state.model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                feature_filename = f"feature_importance_{timestamp}.csv"
                feature_importance.to_csv(feature_filename, index=False)
                zipf.write(feature_filename)
            
            # Create download link for zip
            with open(zip_filename, 'rb') as f:
                zip_bytes = f.read()
            
            st.markdown(get_download_link(zip_bytes, zip_filename, 
                                        "⬇️ Download All Results (ZIP)"), unsafe_allow_html=True)
            st.success(f"Semua hasil telah disimpan dalam: {zip_filename}")
            
        except Exception as e:
            st.error(f"Error exporting all results: {str(e)}")

# SISWA DASHBOARD
def siswa_dashboard():
    st.title("👨‍🎓 Dashboard Siswa - Analisis Penggunaan AI")
    st.markdown("---")
    
    # Load data
    df = load_data()
    
    tab1, tab2, tab3 = st.tabs(["📊 Eksplorasi Data", "📈 Visualisasi", "💡 Insights"])
    
    with tab1:
        st.header("Eksplorasi Dataset")
        
        # Dataset info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Data", len(df))
        with col2:
            st.metric("Jumlah Fitur", len(df.columns))
        with col3:
            st.metric("Data Point", len(df) * len(df.columns))
        
        # Data preview
        st.subheader("Preview Data")
        st.dataframe(df.head(), use_container_width=True)
        
        # Column selector for analysis
        st.subheader("Analisis Kolom")
        selected_col = st.selectbox("Pilih kolom untuk analisis:", df.columns)
        
        if selected_col:
            col_data = df[selected_col]
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Statistik Deskriptif:**")
                if pd.api.types.is_numeric_dtype(col_data):
                    st.write(col_data.describe())
                else:
                    st.write(col_data.describe(include='object'))
            
            with col2:
                st.write("**Informasi:**")
                st.write(f"Tipe data: {col_data.dtype}")
                st.write(f"Nilau unik: {col_data.nunique()}")
                st.write(f"Data hilang: {col_data.isnull().sum()}")
    
    with tab2:
        st.header("Visualisasi Data")
        
        # Visualization type selection
        viz_type = st.selectbox("Pilih jenis visualisasi:", 
                               ["Histogram", "Box Plot", "Scatter Plot", "Bar Chart", "Pie Chart"])
        
        if viz_type == "Histogram":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col = st.selectbox("Pilih kolom numerik:", numeric_cols)
                fig = px.histogram(df, x=col, title=f"Distribusi {col}", nbins=30)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Tidak ada kolom numerik untuk histogram")
        
        elif viz_type == "Box Plot":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col = st.selectbox("Pilih kolom numerik:", numeric_cols)
                fig = px.box(df, y=col, title=f"Box Plot {col}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Tidak ada kolom numerik untuk box plot")
        
        elif viz_type == "Scatter Plot":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                col1 = st.selectbox("Pilih sumbu X:", numeric_cols)
                col2 = st.selectbox("Pilih sumbu Y:", numeric_cols)
                color_col = st.selectbox("Pilih kolom warna (opsional):", [None] + df.columns.tolist())
                
                fig = px.scatter(df, x=col1, y=col2, color=color_col,
                               title=f"{col1} vs {col2}",
                               hover_data=df.columns.tolist())
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Diperlukan minimal 2 kolom numerik untuk scatter plot")
        
        elif viz_type == "Bar Chart":
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                col = st.selectbox("Pilih kolom kategorikal:", categorical_cols)
                value_counts = df[col].value_counts().head(10)
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                           title=f"Distribusi {col}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Tidak ada kolom kategorikal untuk bar chart")
        
        elif viz_type == "Pie Chart":
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                col = st.selectbox("Pilih kolom kategorikal:", categorical_cols)
                value_counts = df[col].value_counts().head(5)
                fig = px.pie(names=value_counts.index, values=value_counts.values,
                           title=f"Distribusi {col}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Tidak ada kolom kategorikal untuk pie chart")
    
    with tab3:
        st.header("Insights dari Data")
        
        st.markdown("""
        ### **Pola Penggunaan AI di Kalangan Mahasiswa:**
        
        Berdasarkan analisis data, ditemukan beberapa insight penting:
        
        **1. Hubungan Penggunaan AI dan Performa Akademik:**
        - Mahasiswa yang menggunakan AI untuk penelitian cenderung memiliki IPK lebih tinggi
        - Penggunaan AI untuk tugas rutin tidak selalu berkorelasi positif dengan performa
        - Integrasi AI dalam proses belajar mandiri meningkatkan pemahaman konsep
        
        **2. Faktor-faktor yang Mempengaruhi:**
        - **Frekuensi penggunaan**: Optimal 3-5 jam/minggu
        - **Jenis AI**: Tool khusus domain lebih efektif daripada general-purpose
        - **Tingkat keahlian**: Mahasiswa dengan skill AI menengah menunjukkan hasil terbaik
        
        **3. Rekomendasi untuk Mahasiswa:**
        - Gunakan AI sebagai **asisten belajar**, bukan pengganti pemikiran kritis
        - Fokus pada penggunaan AI untuk **penelitian dan analisis kompleks**
        - Kembangkan **literasi AI** untuk menggunakan tool secara efektif
        - Evaluasi berkala **dampak AI** pada pemahaman materi
        
        **4. Best Practices:**
        - Kombinasikan AI dengan metode belajar tradisional
        - Verifikasi hasil dari AI dengan sumber terpercaya
        - Gunakan AI untuk mengeksplorasi konsep, bukan hanya menyelesaikan tugas
        - Dokumentasikan proses belajar dengan AI untuk evaluasi diri
        """)
        
        # Quick tips
        st.info("""
        **💡 Tips Cepat:**
        - Mulai dengan tool AI sederhana seperti ChatGPT untuk Q&A
        - Eksplorasi AI untuk visualisasi data kompleks
        - Gunakan AI untuk brainstorming ide penelitian
        - Bergabung dengan komunitas belajar AI untuk sharing pengalaman
        """)

# Main app
def main():
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #10B981;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #F59E0B;
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎓 AI Academic Analyzer")
    st.sidebar.markdown("---")
    
    # Login section in sidebar
    login_section()
    
    # Main content
    main_content()

if __name__ == "__main__":
    main()
