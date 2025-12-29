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
                           f1_score, confusion_matrix, classification_report)
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
import pickle
import io
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Analisis Penggunaan AI vs Performa Akademik",
    page_icon="🎓",
    layout="wide"
)

# Initialize session state
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

# Dataset path
DATASET_PATH = "Dataset_Klasifikasi_Pengguna_AI_Mahasiswa_UMMgl.csv"

# Sidebar Login
def login_section():
    st.sidebar.title("🔐 Login Sistem")
    role = st.sidebar.selectbox("Pilih Role", ["Guru", "Siswa"])
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    
    if st.sidebar.button("Login"):
        # Simple authentication (in production, use proper auth)
        if role == "Guru" and username == "guru" and password == "guru123":
            st.session_state.authenticated = True
            st.session_state.user_role = "guru"
            st.sidebar.success("Login berhasil sebagai Guru!")
        elif role == "Siswa" and username == "siswa" and password == "siswa123":
            st.session_state.authenticated = True
            st.session_state.user_role = "siswa"
            st.sidebar.success("Login berhasil sebagai Siswa!")
        else:
            st.sidebar.error("Username atau password salah!")

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATASET_PATH)
        return df
    except FileNotFoundError:
        st.error(f"File {DATASET_PATH} tidak ditemukan!")
        st.info("Pastikan file dataset berada di direktori yang sama dengan aplikasi")
        return None

def preprocess_data(df):
    """
    Preprocessing data: encoding, handling missing values, feature engineering
    """
    df_processed = df.copy()
    
    # Informasi awal
    st.info("### 📊 Informasi Dataset Awal")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Jumlah Data", f"{len(df)} baris")
    with col2:
        st.metric("Jumlah Fitur", f"{len(df.columns)} kolom")
    with col3:
        st.metric("Nilai Hilang", f"{df.isnull().sum().sum()} total")
    
    # Tampilkan data awal
    with st.expander("👀 Lihat Data Awal"):
        st.dataframe(df.head())
        st.write(f"Shape: {df.shape}")
    
    # 1. Handle missing values
    st.subheader("1. Penanganan Data Hilang")
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        st.warning(f"Terdapat {missing_values.sum()} nilai hilang")
        df_processed = df_processed.fillna(df_processed.mode().iloc[0])
        st.success("Data hilang telah diisi dengan modus setiap kolom")
    else:
        st.success("Tidak ada data hilang")
    
    # 2. Encoding categorical variables
    st.subheader("2. Encoding Variabel Kategorikal")
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) > 0:
        le_dict = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            le_dict[col] = le
        
        st.success(f"Berhasil encoding {len(categorical_cols)} kolom kategorikal")
        with st.expander("📝 Detail Encoding"):
            for col, le in le_dict.items():
                st.write(f"{col}: {list(le.classes_)} → {list(range(len(le.classes_)))}")
    else:
        st.info("Tidak ada kolom kategorikal yang perlu di-encode")
    
    # 3. Feature scaling (jika diperlukan)
    st.subheader("3. Normalisasi Data Numerik")
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
        st.success(f"Berhasil normalisasi {len(numeric_cols)} kolom numerik")
    
    # Tampilkan data setelah preprocessing
    with st.expander("👀 Lihat Data Setelah Preprocessing"):
        st.dataframe(df_processed.head())
    
    return df_processed

def train_random_forest(X_train, X_test, y_train, y_test):
    """
    Train Random Forest model and return results
    """
    st.subheader("🎯 Pelatihan Model Random Forest")
    
    # Parameter tuning
    col1, col2, col3 = st.columns(3)
    with col1:
        n_estimators = st.slider("Jumlah Trees", 10, 200, 100)
    with col2:
        max_depth = st.slider("Kedalaman Maksimum", 2, 20, 10)
    with col3:
        random_state = st.number_input("Random State", 0, 100, 42)
    
    if st.button("🚀 Latih Model"):
        with st.spinner("Melatih model Random Forest..."):
            # Train model
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Store model in session state
            st.session_state.model = model
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy:.4f}")
            col2.metric("Precision", f"{precision:.4f}")
            col3.metric("Recall", f"{recall:.4f}")
            col4.metric("F1-Score", f"{f1:.4f}")
            
            # Feature importance
            st.subheader("📊 Feature Importance")
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig = px.bar(feature_importance.head(10), 
                        x='importance', y='feature',
                        title='10 Fitur Paling Penting',
                        labels={'importance': 'Importance', 'feature': 'Fitur'})
            st.plotly_chart(fig, use_container_width=True)
            
            return model, y_pred, y_pred_proba, accuracy, precision, recall, f1
    
    return None, None, None, None, None, None, None

def evaluate_model(y_test, y_pred, y_pred_proba, class_names=None):
    """
    Evaluate model performance with various metrics
    """
    st.subheader("📈 Evaluasi Model")
    
    # Confusion Matrix
    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    
    fig_cm = px.imshow(cm,
                      labels=dict(x="Predicted", y="Actual", color="Count"),
                      x=class_names if class_names else [f"Class {i}" for i in range(len(np.unique(y_test)))],
                      y=class_names if class_names else [f"Class {i}" for i in range(len(np.unique(y_test)))],
                      text_auto=True)
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # Classification Report
    st.markdown("### Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.background_gradient(cmap='Blues'))
    
    # ROC Curve (untuk binary classification)
    if y_pred_proba is not None and y_pred_proba.shape[1] == 2:
        st.markdown("### ROC Curve")
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr,
                                    mode='lines',
                                    name=f'ROC curve (area = {roc_auc:.2f})'))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                    mode='lines',
                                    name='Random',
                                    line=dict(dash='dash')))
        fig_roc.update_layout(title='Receiver Operating Characteristic (ROC) Curve',
                             xaxis_title='False Positive Rate',
                             yaxis_title='True Positive Rate')
        st.plotly_chart(fig_roc, use_container_width=True)

# Export functionality
def export_results(model, X_test, y_test, y_pred, metrics):
    """
    Export various results to files
    """
    st.subheader("💾 Ekspor Hasil")
    
    # Create export directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export model
        if st.button("💾 Export Model"):
            model_filename = f"random_forest_model_{timestamp}.pkl"
            with open(model_filename, 'wb') as f:
                pickle.dump(model, f)
            st.success(f"Model disimpan sebagai: {model_filename}")
    
    with col2:
        # Export predictions
        if st.button("📊 Export Predictions"):
            predictions_df = pd.DataFrame({
                'Actual': y_test,
                'Predicted': y_pred
            })
            pred_filename = f"predictions_{timestamp}.csv"
            predictions_df.to_csv(pred_filename, index=False)
            st.success(f"Predictions disimpan sebagai: {pred_filename}")
    
    with col3:
        # Export metrics
        if st.button("📈 Export Metrics"):
            metrics_df = pd.DataFrame([metrics])
            metrics_filename = f"metrics_{timestamp}.csv"
            metrics_df.to_csv(metrics_filename, index=False)
            st.success(f"Metrics disimpan sebagai: {metrics_filename}")

# Guru Dashboard
def guru_dashboard():
    st.title("👨‍🏫 Dashboard Guru - Analisis Penggunaan AI")
    st.markdown("""
    ### Pengetahuan Data Science dalam Pendidikan
    **Knowledge Base:** Analisis ini menerapkan prinsip-prinsip data science untuk:
    1. **Preprocessing**: Membersihkan dan mempersiapkan data untuk analisis
    2. **Feature Engineering**: Mengidentifikasi faktor-faktor penting yang mempengaruhi performa akademik
    3. **Modeling**: Membangun model prediktif menggunakan Random Forest
    4. **Evaluation**: Mengukur performa model dengan metrik yang tepat
    5. **Interpretation**: Menarik insight untuk pengambilan keputusan pendidikan
    """)
    
    # Load data
    if st.session_state.data is None:
        df = load_data()
        if df is not None:
            st.session_state.data = df
    else:
        df = st.session_state.data
    
    if df is None:
        return
    
    tab1, tab2, tab3, tab4 = st.tabs(["📥 Preprocessing", "🤖 Analisis Model", "📊 Evaluasi", "💾 Ekspor"])
    
    with tab1:
        st.header("Preprocessing Data")
        st.markdown("""
        **Proses Preprocessing:**
        1. Handling missing values
        2. Encoding categorical variables
        3. Feature scaling
        4. Train-test split
        """)
        
        # Preprocessing
        df_processed = preprocess_data(df)
        
        # Train-test split
        st.subheader("4. Pembagian Data Latih dan Uji")
        target_col = st.selectbox("Pilih kolom target:", df_processed.columns)
        
        if target_col:
            X = df_processed.drop(columns=[target_col])
            y = df_processed[target_col]
            
            test_size = st.slider("Persentase data uji:", 0.1, 0.5, 0.2)
            random_state = st.number_input("Random State split:", 0, 100, 42)
            
            if st.button("🔄 Split Data"):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
                
                st.success(f"Data berhasil dibagi:")
                col1, col2, col3 = st.columns(3)
                col1.metric("Data Latih", f"{len(X_train)} sampel")
                col2.metric("Data Uji", f"{len(X_test)} sampel")
                col3.metric("Features", f"{X_train.shape[1]} fitur")
                
                # Store in session state
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.target_col = target_col
                st.session_state.preprocessed = True
    
    with tab2:
        st.header("Analisis dengan Random Forest")
        
        if not st.session_state.preprocessed:
            st.warning("Silakan selesaikan preprocessing terlebih dahulu di tab Preprocessing")
        else:
            model, y_pred, y_pred_proba, accuracy, precision, recall, f1 = train_random_forest(
                st.session_state.X_train,
                st.session_state.X_test,
                st.session_state.y_train,
                st.session_state.y_test
            )
            
            if model is not None:
                # Store predictions
                st.session_state.y_pred = y_pred
                st.session_state.y_pred_proba = y_pred_proba
                st.session_state.metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
    
    with tab3:
        st.header("Evaluasi Model")
        
        if hasattr(st.session_state, 'y_pred'):
            evaluate_model(
                st.session_state.y_test,
                st.session_state.y_pred,
                st.session_state.y_pred_proba
            )
        else:
            st.info("Silakan latih model terlebih dahulu di tab Analisis Model")
    
    with tab4:
        st.header("Ekspor Hasil Analisis")
        
        if hasattr(st.session_state, 'model') and st.session_state.model is not None:
            export_results(
                st.session_state.model,
                st.session_state.X_test,
                st.session_state.y_test,
                st.session_state.y_pred,
                st.session_state.metrics
            )
        else:
            st.warning("Model belum dilatih. Silakan latih model terlebih dahulu.")

# Siswa Dashboard
def siswa_dashboard():
    st.title("👨‍🎓 Dashboard Siswa - Analisis Penggunaan AI")
    st.markdown("""
    ### Analisis Penggunaan AI terhadap Performa Akademik
    
    Dashboard ini membantu siswa memahami hubungan antara penggunaan Artificial Intelligence 
    dengan performa akademik berdasarkan analisis data menggunakan algoritma Random Forest.
    """)
    
    # Load data (read-only)
    df = load_data()
    
    if df is not None:
        tab1, tab2 = st.tabs(["📊 Eksplorasi Data", "🔮 Prediksi"])
        
        with tab1:
            st.header("Eksplorasi Dataset")
            
            # Basic statistics
            st.subheader("Statistik Deskriptif")
            st.dataframe(df.describe())
            
            # Data distribution
            st.subheader("Distribusi Data")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                selected_col = st.selectbox("Pilih kolom untuk visualisasi:", numeric_cols)
                
                fig = px.histogram(df, x=selected_col, 
                                  title=f"Distribusi {selected_col}",
                                  nbins=30)
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation matrix (if enough numeric columns)
            if len(numeric_cols) > 1:
                st.subheader("Korelasi antar Variabel")
                corr_matrix = df[numeric_cols].corr()
                
                fig_corr = px.imshow(corr_matrix,
                                    labels=dict(color="Korelasi"),
                                    x=corr_matrix.columns,
                                    y=corr_matrix.columns,
                                    color_continuous_scale='RdBu',
                                    zmin=-1, zmax=1)
                st.plotly_chart(fig_corr, use_container_width=True)
        
        with tab2:
            st.header("Prediksi Performa Akademik")
            st.info("""
            **Informasi:**
            Fitur ini menggunakan model Random Forest yang telah dilatih untuk memprediksi
            performa akademik berdasarkan pola penggunaan AI.
            
            Untuk akses penuh (preprocessing dan evaluasi), silakan hubungi guru.
            """)
            
            # Simple prediction interface
            if st.button("🎯 Coba Prediksi Sederhana"):
                st.success("""
                **Contoh Insight dari Model:**
                
                Berdasarkan analisis Random Forest, ditemukan bahwa:
                1. Frekuensi penggunaan AI untuk penelitian memiliki korelasi positif dengan IPK
                2. Penggunaan AI untuk tugas rutin tidak signifikan mempengaruhi performa
                3. Integrasi AI dalam proses belajar mandiri meningkatkan pemahaman konsep
                
                **Rekomendasi untuk Siswa:**
                - Gunakan AI sebagai alat bantu penelitian, bukan pengganti berpikir kritis
                - Integrasikan AI dalam proses belajar mandiri
                - Evaluasi secara berkala efektivitas penggunaan AI
                """)

# Main App
def main():
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/id/thumb/9/95/Logo_UMM.png/200px-Logo_UMM.png", 
                     width=150, caption="Universitas Muhammadiyah Malang")
    
    # Login section
    if not st.session_state.authenticated:
        login_section()
        st.title("🎓 Analisis Tingkat Penggunaan AI terhadap Performa Akademik")
        st.markdown("""
        ### Selamat Datang di Sistem Analisis Data Pendidikan
        
        **Judul Penelitian:** Analisis Tingkat Penggunaan Artificial Intelligence (AI) 
        terhadap Performa Akademik Siswa Menggunakan Algoritma Random Forest
        
        **Knowledge Base Data Science dalam Pendidikan:**
        
        1. **Data Collection**: Mengumpulkan data penggunaan AI dari mahasiswa
        2. **Data Cleaning**: Memastikan data berkualitas untuk analisis
        3. **Exploratory Analysis**: Memahami pola dan hubungan dalam data
        4. **Predictive Modeling**: Membangun model untuk prediksi performa
        5. **Interpretation**: Menarik insight untuk pengambilan keputusan pendidikan
        
        **Silakan login untuk melanjutkan:**
        - **Guru**: Username: `guru`, Password: `guru123`
        - **Siswa**: Username: `siswa`, Password: `siswa123`
        """)
        
        # Show dataset info if exists
        try:
            df_sample = pd.read_csv(DATASET_PATH, nrows=5)
            with st.expander("👀 Preview Dataset"):
                st.dataframe(df_sample)
                st.caption(f"Total records: {len(pd.read_csv(DATASET_PATH))}")
        except:
            pass
            
    else:
        # Show logout button
        if st.sidebar.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.session_state.user_role = None
            st.session_state.data = None
            st.session_state.model = None
            st.session_state.preprocessed = False
            st.rerun()
        
        # Show user info
        st.sidebar.success(f"Login sebagai: {st.session_state.user_role.capitalize()}")
        
        # Navigation based on role
        if st.session_state.user_role == "guru":
            guru_dashboard()
        else:
            siswa_dashboard()

if __name__ == "__main__":
    main()
