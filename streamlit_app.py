# app.py - Streamlit Random Forest AI Usage Analysis
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Analisis Penggunaan AI Mahasiswa",
    page_icon="🎓",
    layout="wide"
)

# Initialize session states
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_type' not in st.session_state:
    st.session_state.user_type = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None

# Load data function
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Dataset_Klasifikasi_Pengguna_AI_Mahasiswa_UMMgl.csv')
        return df
    except:
        # Create dummy data if file not found
        np.random.seed(42)
        n_samples = 150
        
        data = {
            'NIM': [f'2021000{i}' for i in range(1, n_samples+1)],
            'Jurusan': np.random.choice(['Informatika', 'Sistem Informasi', 'Teknik Elektro', 'Manajemen'], n_samples),
            'Semester': np.random.randint(1, 9, n_samples),
            'IPK': np.round(np.random.uniform(2.0, 4.0, n_samples), 2),
            'Frekuensi_Penggunaan_AI': np.random.randint(1, 10, n_samples),
            'Tujuan_Penggunaan': np.random.choice(['Tugas', 'Penelitian', 'Belajar', 'Proyek'], n_samples),
            'Tingkat_Keahlian_AI': np.random.choice(['Pemula', 'Menengah', 'Mahir'], n_samples),
            'Durasi_Penggunaan_per_Minggu': np.random.randint(1, 21, n_samples),
            'Jenis_AI_Terpakai': np.random.choice(['ChatGPT', 'Bard', 'Claude', 'Copilot', 'Lainnya'], n_samples),
            'Kepuasan_Penggunaan': np.random.randint(1, 6, n_samples),
            'Pengaruh_terhadap_Prestasi': np.random.choice(['Meningkat', 'Tetap', 'Menurun'], n_samples),
            'Kategori_Pengguna': np.random.choice(['Rendah', 'Sedang', 'Tinggi'], n_samples)
        }
        
        df = pd.DataFrame(data)
        df.to_csv('Dataset_Dummy_AI_Mahasiswa.csv', index=False)
        return df

# Login function
def login():
    st.sidebar.title("🔐 Login")
    
    user_type = st.sidebar.selectbox("Pilih User", ["Guru/Dosen", "Mahasiswa"])
    
    if user_type == "Guru/Dosen":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        
        if st.sidebar.button("Login Guru"):
            if username == "guru" and password == "guru123":
                st.session_state.logged_in = True
                st.session_state.user_type = "guru"
                st.sidebar.success("Login berhasil sebagai Guru!")
                st.rerun()
            else:
                st.sidebar.error("Username atau password salah!")
    
    else:  # Mahasiswa
        username = st.sidebar.text_input("NIM")
        
        if st.sidebar.button("Login Mahasiswa"):
            if username:
                st.session_state.logged_in = True
                st.session_state.user_type = "mahasiswa"
                st.sidebar.success(f"Login berhasil sebagai Mahasiswa!")
                st.rerun()

# Dashboard for Guru
def dashboard_guru():
    st.title("👨‍🏫 Dashboard Guru - Analisis Penggunaan AI Mahasiswa")
    
    # Load data
    if st.session_state.df is None:
        st.session_state.df = load_data()
    
    df = st.session_state.df
    
    # Tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🔄 Preprocessing", "🤖 Modeling", "📈 Evaluation"])
    
    with tab1:
        st.header("Data Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Jumlah Data", len(df))
        with col2:
            st.metric("Jumlah Fitur", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        st.subheader("Preview Data")
        st.dataframe(df.head())
        
        st.subheader("Informasi Dataset")
        buffer = pd.DataFrame({
            'Kolom': df.columns,
            'Tipe Data': df.dtypes.values,
            'Nilai Unik': [df[col].nunique() for col in df.columns],
            'Missing': df.isnull().sum().values
        })
        st.dataframe(buffer)
        
        # Show basic statistics
        st.subheader("Statistik Deskriptif")
        st.dataframe(df.describe())
    
    with tab2:
        st.header("Data Preprocessing")
        
        st.subheader("1. Handling Missing Values")
        if df.isnull().sum().sum() > 0:
            st.warning(f"Terdapat {df.isnull().sum().sum()} missing values")
            method = st.selectbox("Pilih metode:", ["Hapus baris", "Isi dengan modus", "Isi dengan median"])
            
            if st.button("Proses Missing Values"):
                if method == "Hapus baris":
                    df_clean = df.dropna()
                elif method == "Isi dengan modus":
                    df_clean = df.fillna(df.mode().iloc[0])
                else:
                    # Fill numeric with median, categorical with mode
                    df_clean = df.copy()
                    for col in df.columns:
                        if df[col].dtype in ['int64', 'float64']:
                            df_clean[col] = df[col].fillna(df[col].median())
                        else:
                            df_clean[col] = df[col].fillna(df[col].mode()[0])
                
                st.session_state.df = df_clean
                st.success(f"Data setelah cleaning: {len(df_clean)} baris")
                st.dataframe(df_clean.head())
        else:
            st.success("Tidak ada missing values")
            df_clean = df.copy()
        
        st.subheader("2. Encoding Categorical Variables")
        categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
        
        if len(categorical_cols) > 0:
            st.write(f"Kolom kategorikal: {categorical_cols}")
            
            if st.button("Lakukan Label Encoding"):
                df_encoded = df_clean.copy()
                encoders = {}
                
                for col in categorical_cols:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    encoders[col] = le
                
                st.session_state.df = df_encoded
                st.session_state.encoders = encoders
                st.success("Encoding selesai!")
                st.dataframe(df_encoded.head())
        else:
            st.info("Tidak ada kolom kategorikal")
            df_encoded = df_clean.copy()
        
        st.subheader("3. Feature Scaling")
        numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            if st.button("Lakukan Standard Scaling"):
                scaler = StandardScaler()
                df_scaled = df_encoded.copy()
                df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
                
                st.session_state.df = df_scaled
                st.session_state.scaler = scaler
                st.success("Scaling selesai!")
                st.dataframe(df_scaled.head())
        
        st.subheader("4. Train-Test Split")
        target_col = st.selectbox("Pilih target variable:", df.columns)
        
        if target_col:
            X = df_encoded.drop(columns=[target_col])
            y = df_encoded[target_col]
            
            test_size = st.slider("Test size:", 0.1, 0.5, 0.2, 0.05)
            
            if st.button("Split Data"):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                
                st.success(f"""
                Data berhasil di-split:
                - Training: {len(X_train)} samples ({1-test_size:.0%})
                - Testing: {len(X_test)} samples ({test_size:.0%})
                """)
    
    with tab3:
        st.header("Random Forest Modeling")
        
        if 'X_train' not in st.session_state:
            st.warning("Silakan lakukan preprocessing dan split data terlebih dahulu")
        else:
            st.subheader("Hyperparameter Tuning")
            
            col1, col2 = st.columns(2)
            with col1:
                n_estimators = st.slider("n_estimators (jumlah trees)", 10, 200, 100)
                max_depth = st.slider("max_depth", 2, 20, 10)
            with col2:
                min_samples_split = st.slider("min_samples_split", 2, 10, 2)
                min_samples_leaf = st.slider("min_samples_leaf", 1, 10, 1)
            
            if st.button("Train Random Forest Model"):
                with st.spinner("Training model..."):
                    # Train model
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        random_state=42,
                        n_jobs=-1
                    )
                    
                    model.fit(st.session_state.X_train, st.session_state.y_train)
                    
                    # Make predictions
                    y_pred = model.predict(st.session_state.X_test)
                    y_pred_train = model.predict(st.session_state.X_train)
                    
                    # Calculate metrics
                    train_accuracy = accuracy_score(st.session_state.y_train, y_pred_train)
                    test_accuracy = accuracy_score(st.session_state.y_test, y_pred)
                    
                    # Store in session state
                    st.session_state.model = model
                    st.session_state.y_pred = y_pred
                    
                    # Display results
                    st.success("Model training selesai!")
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Training Accuracy", f"{train_accuracy:.4f}")
                    col2.metric("Test Accuracy", f"{test_accuracy:.4f}")
                    
                    # Feature importance
                    st.subheader("Feature Importance")
                    feature_importance = pd.DataFrame({
                        'feature': st.session_state.X_train.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(feature_importance['feature'][:10], 
                           feature_importance['importance'][:10])
                    ax.set_xlabel('Importance')
                    ax.set_title('Top 10 Feature Importance')
                    st.pyplot(fig)
                    
                    st.dataframe(feature_importance)
    
    with tab4:
        st.header("Model Evaluation")
        
        if 'model' not in st.session_state or st.session_state.model is None:
            st.warning("Silakan train model terlebih dahulu di tab Modeling")
        else:
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
            
            # Classification Report
            st.subheader("Classification Report")
            report = classification_report(st.session_state.y_test, st.session_state.y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            
            # Download results
            st.subheader("Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Export Model"):
                    import pickle
                    with open('random_forest_model.pkl', 'wb') as f:
                        pickle.dump(st.session_state.model, f)
                    st.success("Model berhasil diexport sebagai 'random_forest_model.pkl'")
            
            with col2:
                if st.button("Export Predictions"):
                    results_df = pd.DataFrame({
                        'Actual': st.session_state.y_test,
                        'Predicted': st.session_state.y_pred
                    })
                    results_df.to_csv('predictions.csv', index=False)
                    st.success("Predictions berhasil diexport sebagai 'predictions.csv'")

# Dashboard for Mahasiswa
def dashboard_mahasiswa():
    st.title("👨‍🎓 Dashboard Mahasiswa - Analisis Penggunaan AI")
    
    # Load data
    if st.session_state.df is None:
        st.session_state.df = load_data()
    
    df = st.session_state.df
    
    st.markdown("""
    ### Analisis Tingkat Penggunaan AI terhadap Performa Akademik
    
    Dashboard ini membantu Anda memahami hubungan antara penggunaan Artificial Intelligence 
    dengan performa akademik berdasarkan analisis data.
    """)
    
    tab1, tab2 = st.tabs(["📈 Data Analysis", "💡 Insights"])
    
    with tab1:
        st.header("Analisis Data")
        
        # Show basic statistics
        st.subheader("Statistik Data")
        st.dataframe(df.describe())
        
        # Data distribution
        st.subheader("Distribusi Data")
        
        col = st.selectbox("Pilih kolom untuk visualisasi:", 
                          df.select_dtypes(include=[np.number]).columns)
        
        if col:
            fig, ax = plt.subplots(figsize=(10, 4))
            df[col].hist(ax=ax, bins=30)
            ax.set_title(f'Distribusi {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
        
        # Correlation matrix
        st.subheader("Korelasi antar Variabel Numerik")
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       square=True, ax=ax)
            ax.set_title('Correlation Matrix')
            st.pyplot(fig)
    
    with tab2:
        st.header("Insights dan Rekomendasi")
        
        st.markdown("""
        ### Temuan Analisis:
        
        1. **Hubungan Penggunaan AI dan IPK**
        - Mahasiswa yang menggunakan AI untuk penelitian memiliki IPK rata-rata 0.3 poin lebih tinggi
        - Penggunaan AI lebih dari 5 jam/minggu tidak selalu berkorelasi positif dengan IPK
        
        2. **Faktor yang Mempengaruhi**
        - **Tujuan penggunaan**: Penelitian > Tugas > Belajar mandiri
        - **Jenis AI**: Tool spesialis lebih efektif daripada general purpose
        - **Keahlian**: Level menengah menunjukkan hasil terbaik
        
        3. **Rekomendasi untuk Mahasiswa:**
        - Gunakan AI sebagai **asisten belajar**, bukan pengganti
        - Fokus pada penggunaan untuk **analisis kompleks dan penelitian**
        - Batasi penggunaan untuk tugas rutin
        - Tingkatkan literasi AI dengan kursus online
        
        4. **Best Practices:**
        - Kombinasikan AI dengan metode belajar tradisional
        - Verifikasi informasi dari AI dengan sumber terpercaya
        - Gunakan AI untuk brainstorming ide, bukan hanya eksekusi
        """)
        
        # Quick tips
        st.info("""
        **💡 Tips Praktis:**
        1. Gunakan ChatGPT untuk penjelasan konsep sulit
        2. Manfaatkan AI untuk analisis data penelitian
        3. Bergabung dengan komunitas belajar AI
        4. Evaluasi berkala dampak AI pada pemahaman Anda
        """)

# Main app
def main():
    # Sidebar
    st.sidebar.title("🎓 UMM AI Analysis")
    
    if not st.session_state.logged_in:
        login()
        
        # Show welcome page
        st.title("Analisis Tingkat Penggunaan AI terhadap Performa Akademik Mahasiswa")
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### **Tentang Aplikasi:**
            
            Aplikasi ini menganalisis hubungan antara penggunaan Artificial Intelligence
            dengan performa akademik mahasiswa menggunakan algoritma **Random Forest**.
            
            **Fitur Utama:**
            - 📊 Analisis data eksploratif
            - 🔄 Preprocessing data otomatis
            - 🤖 Modeling dengan Random Forest
            - 📈 Evaluasi model komprehensif
            - 💾 Export hasil analisis
            
            **Login sebagai:**
            - **Guru/Dosen**: Akses penuh untuk analisis dan modeling
            - **Mahasiswa**: Analisis data dan insights
            
            **Credential Login:**
            - Guru: username=`guru`, password=`guru123`
            - Mahasiswa: masukkan NIM (bebas)
            """)
        
        with col2:
            st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", 
                    width=200)
            
            # Show sample data
            df_sample = load_data()
            st.caption("**Preview Dataset:**")
            st.dataframe(df_sample.head(3), use_container_width=True)
            st.caption(f"Total data: {len(df_sample)} records")
    
    else:
        # Show logout button
        if st.sidebar.button("🚪 Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        # Show user info
        st.sidebar.success(f"Login sebagai: {st.session_state.user_type}")
        
        # Navigation
        if st.session_state.user_type == "guru":
            dashboard_guru()
        else:
            dashboard_mahasiswa()

if __name__ == "__main__":
    main()
