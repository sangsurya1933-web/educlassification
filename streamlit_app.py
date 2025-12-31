import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Penggunaan AI Mahasiswa",
    page_icon="🤖",
    layout="wide"
)

# Fungsi untuk inisialisasi session state
def init_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'encoders' not in st.session_state:
        st.session_state.encoders = {}
    if 'results' not in st.session_state:
        st.session_state.results = None

init_session_state()

# Knowledgebase rules untuk rekomendasi
KNOWLEDGEBASE_RULES = {
    'rendah': {
        'conditions': ['Usage_Intensity_Score <= 4', 'Trust_Level <= 2'],
        'recommendations': [
            'Perlu peningkatan pelatihan penggunaan AI tools',
            'Disarankan mengikuti workshop AI dasar',
            'Tingkatkan kepercayaan dengan mencoba tools yang user-friendly',
            'Mulai dengan penggunaan untuk tugas sederhana'
        ]
    },
    'sedang': {
        'conditions': ['4 < Usage_Intensity_Score <= 7', '2 < Trust_Level <= 4'],
        'recommendations': [
            'Tingkatkan variasi penggunaan AI tools',
            'Eksplorasi fitur-fitur advanced dari tools yang sudah digunakan',
            'Coba integrasikan AI dalam proyek akademik',
            'Bergabung dengan komunitas pengguna AI'
        ]
    },
    'tinggi': {
        'conditions': ['Usage_Intensity_Score > 7', 'Trust_Level > 4'],
        'recommendations': [
            'Optimalisasi penggunaan untuk penelitian',
            'Coba kembangkan project berbasis AI',
            'Berbagi pengalaman dengan mahasiswa lain',
            'Eksplorasi AI untuk pengembangan karir'
        ]
    }
}

def login_page():
    st.title("🔐 Login Sistem Analisis AI Mahasiswa")
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1,2,1])
    
    with col2:
        role = st.selectbox("Pilih Peran", ["Guru/Dosen", "Mahasiswa"])
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login", use_container_width=True):
            if role == "Guru/Dosen":
                if username == "guru" and password == "guru123":
                    st.session_state.logged_in = True
                    st.session_state.user_role = "guru"
                    st.success("Login berhasil sebagai Guru!")
                    st.rerun()
                else:
                    st.error("Username atau password salah!")
            else:
                if username and password:
                    st.session_state.logged_in = True
                    st.session_state.user_role = "siswa"
                    st.session_state.student_name = username
                    st.success(f"Login berhasil sebagai Mahasiswa: {username}")
                    st.rerun()
                else:
                    st.error("Username harus diisi!")

def load_data(uploaded_file=None):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, sep=';')
            return df
        except:
            st.error("Format file tidak valid. Gunakan CSV dengan pemisah titik koma (;)")
            return None
    else:
        # Load default dataset
        df = pd.read_csv("Dataset_Klasifikasi_Pengguna_AI_Mahasiswa_UMMgl_PAKE.csv", sep=';')
        return df

def clean_data(df):
    st.subheader("📊 Data Cleaning")
    
    st.write("**Data Asli:**")
    st.dataframe(df.head(), use_container_width=True)
    
    # 1. Handle missing values
    st.write("### 1. Penanganan Missing Values")
    missing_before = df.isnull().sum().sum()
    df_cleaned = df.dropna()
    missing_after = df_cleaned.isnull().sum().sum()
    
    st.info(f"Missing values sebelum: {missing_before}")
    st.success(f"Missing values sesudah: {missing_after}")
    
    # 2. Handle duplicate entries
    st.write("### 2. Penanganan Data Duplikat")
    duplicates_before = df_cleaned.duplicated().sum()
    df_cleaned = df_cleaned.drop_duplicates()
    duplicates_after = df_cleaned.duplicated().sum()
    
    st.info(f"Data duplikat sebelum: {duplicates_before}")
    st.success(f"Data duplikat sesudah: {duplicates_after}")
    
    # 3. Handle '10+' in Usage_Intensity_Score
    st.write("### 3. Normalisasi Usage_Intensity_Score")
    if 'Usage_Intensity_Score' in df_cleaned.columns:
        df_cleaned['Usage_Intensity_Score'] = df_cleaned['Usage_Intensity_Score'].replace('10+', '10')
        df_cleaned['Usage_Intensity_Score'] = pd.to_numeric(df_cleaned['Usage_Intensity_Score'], errors='coerce')
        df_cleaned = df_cleaned.dropna(subset=['Usage_Intensity_Score'])
    
    st.write("**Data Setelah Cleaning:**")
    st.dataframe(df_cleaned.head(), use_container_width=True)
    
    return df_cleaned

def encode_categorical_data(df):
    st.subheader("🔡 Encoding Data Kategorikal")
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != 'Nama']
    
    encoded_df = df.copy()
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
        encoders[col] = le
        
        # Show mapping
        st.write(f"**Encoding untuk {col}:**")
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        mapping_df = pd.DataFrame(list(mapping.items()), columns=['Nilai Asli', 'Kode'])
        st.dataframe(mapping_df, use_container_width=True)
    
    st.session_state.encoders = encoders
    
    st.write("**Data Setelah Encoding:**")
    st.dataframe(encoded_df.head(), use_container_width=True)
    
    return encoded_df, encoders

def prepare_features_target(df):
    st.subheader("🎯 Persiapan Fitur dan Target")
    
    # Buat target klasifikasi berdasarkan Usage_Intensity_Score
    df['AI_Usage_Class'] = pd.cut(df['Usage_Intensity_Score'],
                                   bins=[0, 4, 7, 11],
                                   labels=['rendah', 'sedang', 'tinggi'])
    
    # Pilih fitur
    feature_cols = ['Studi_Jurusan', 'Semester', 'AI_Tools', 'Trust_Level']
    
    # Pastikan semua fitur ada
    available_features = [col for col in feature_cols if col in df.columns]
    
    X = df[available_features]
    y = df['AI_Usage_Class']
    
    st.write("**Fitur yang digunakan:**", available_features)
    st.write("**Distribusi Kelas Target:**")
    st.bar_chart(y.value_counts())
    
    return X, y, df

def split_data(X, y):
    st.subheader("✂️ Split Data")
    
    test_size = st.slider("Persentase Data Testing", 10, 40, 20)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=42, stratify=y
    )
    
    st.write(f"**Data Training:** {len(X_train)} sampel ({100-test_size}%)")
    st.write(f"**Data Testing:** {len(X_test)} sampel ({test_size}%)")
    
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, X_test, y_train, y_test):
    st.subheader("🌲 Training Model Random Forest")
    
    # Hyperparameter tuning
    col1, col2, col3 = st.columns(3)
    with col1:
        n_estimators = st.slider("Jumlah Trees", 50, 300, 100)
    with col2:
        max_depth = st.slider("Kedalaman Maksimum", 5, 50, 20)
    with col3:
        random_state = st.slider("Random State", 0, 100, 42)
    
    if st.button("🚀 Train Model", use_container_width=True):
        with st.spinner("Training model Random Forest..."):
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Evaluation
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Store results
            st.session_state.model = model
            st.session_state.results = {
                'accuracy': accuracy,
                'report': report,
                'y_test': y_test,
                'y_pred': y_pred,
                'feature_names': X_train.columns.tolist()
            }
            
            # Display results
            st.success(f"✅ Model berhasil dilatih!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Akurasi Model", f"{accuracy:.2%}")
            with col2:
                st.metric("Jumlah Trees", n_estimators)
            with col3:
                st.metric("Kedalaman", max_depth)
            
            # Feature Importance
            st.write("### 🔝 Feature Importance")
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            st.bar_chart(feature_importance.set_index('feature'))
            
            # Classification Report
            st.write("### 📋 Classification Report")
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
            
            # Confusion Matrix
            st.write("### 🎯 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(cm, 
                                index=['rendah', 'sedang', 'tinggi'],
                                columns=['rendah', 'sedang', 'tinggi'])
            st.dataframe(cm_df, use_container_width=True)
    
    return st.session_state.model

def get_recommendation(usage_class, trust_level, usage_score):
    """Generate recommendation based on knowledgebase rules"""
    recommendations = KNOWLEDGEBASE_RULES.get(usage_class, {}).get('recommendations', [])
    
    # Tambahkan rekomendasi spesifik berdasarkan trust level
    if trust_level <= 2:
        recommendations.append("Fokus pada peningkatan kepercayaan terhadap teknologi AI")
    elif trust_level >= 4:
        recommendations.append("Manfaatkan kepercayaan tinggi untuk eksplorasi tools lebih lanjut")
    
    # Tambahkan rekomendasi berdasarkan usage score
    if usage_score <= 4:
        recommendations.append("Set target peningkatan penggunaan mingguan")
    elif usage_score >= 8:
        recommendations.append("Pertimbangkan untuk menjadi mentor AI bagi teman sekelas")
    
    return list(set(recommendations))[:4]  # Ambil 4 rekomendasi unik

def evaluation_recommendation_page():
    st.title("📈 Evaluasi & Rekomendasi")
    st.markdown("---")
    
    if st.session_state.results is None:
        st.warning("Silakan train model terlebih dahulu di menu 'Uji dengan Random Forest'")
        return
    
    results = st.session_state.results
    
    # Tampilkan evaluasi
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Akurasi Model", f"{results['accuracy']:.2%}")
    with col2:
        st.metric("Jumlah Sampel Testing", len(results['y_test']))
    with col3:
        st.metric("Model Siap", "✅" if st.session_state.model else "❌")
    
    # Prediksi untuk semua data
    if st.session_state.data is not None and st.session_state.model is not None:
        st.subheader("🎯 Prediksi & Rekomendasi per Mahasiswa")
        
        # Encode data untuk prediksi
        df_encoded, _ = encode_categorical_data(st.session_state.data.copy())
        
        # Siapkan fitur
        feature_cols = [col for col in ['Studi_Jurusan', 'Semester', 'AI_Tools', 'Trust_Level'] 
                       if col in df_encoded.columns]
        
        if len(feature_cols) > 0:
            X_all = df_encoded[feature_cols]
            
            # Predict
            predictions = st.session_state.model.predict(X_all)
            
            # Buat dataframe hasil
            results_df = st.session_state.data.copy()
            results_df['Klasifikasi_Penggunaan_AI'] = predictions
            
            # Tambahkan rekomendasi
            recommendations = []
            for idx, row in results_df.iterrows():
                usage_class = predictions[idx]
                trust_level = row['Trust_Level'] if 'Trust_Level' in row else 3
                usage_score = row['Usage_Intensity_Score'] if 'Usage_Intensity_Score' in row else 5
                recs = get_recommendation(usage_class, trust_level, usage_score)
                recommendations.append(", ".join(recs))
            
            results_df['Rekomendasi'] = recommendations
            
            # Tampilkan hasil
            st.write("**Hasil Klasifikasi & Rekomendasi:**")
            
            # Filter columns untuk tampilan
            display_cols = ['Nama', 'Studi_Jurusan', 'Semester', 'AI_Tools', 
                          'Trust_Level', 'Usage_Intensity_Score', 
                          'Klasifikasi_Penggunaan_AI', 'Rekomendasi']
            display_cols = [col for col in display_cols if col in results_df.columns]
            
            st.dataframe(results_df[display_cols], use_container_width=True)
            
            # Summary statistics
            st.subheader("📊 Statistik Klasifikasi")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                rendah = (results_df['Klasifikasi_Penggunaan_AI'] == 'rendah').sum()
                st.metric("Rendah", rendah)
            with col2:
                sedang = (results_df['Klasifikasi_Penggunaan_AI'] == 'sedang').sum()
                st.metric("Sedang", sedang)
            with col3:
                tinggi = (results_df['Klasifikasi_Penggunaan_AI'] == 'tinggi').sum()
                st.metric("Tinggi", tinggi)
            
            # Chart distribusi
            st.bar_chart(results_df['Klasifikasi_Penggunaan_AI'].value_counts())
            
            # Tombol download
            csv = results_df.to_csv(index=False, sep=';').encode('utf-8')
            st.download_button(
                label="📥 Download Hasil Analisis (CSV)",
                data=csv,
                file_name=f"hasil_analisis_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.error("Fitur yang diperlukan tidak ditemukan dalam dataset")

def student_dashboard():
    st.title("👨‍🎓 Dashboard Mahasiswa")
    st.markdown("---")
    
    student_name = st.session_state.get('student_name', 'Mahasiswa')
    st.success(f"Halo {student_name}! Berikut adalah hasil analisis penggunaan AI Anda:")
    
    if st.session_state.data is not None and st.session_state.model is not None:
        # Cari data mahasiswa berdasarkan nama
        student_data = st.session_state.data[
            st.session_state.data['Nama'].str.contains(student_name, case=False, na=False)
        ]
        
        if not student_data.empty:
            # Encode data untuk prediksi
            df_encoded, _ = encode_categorical_data(student_data.copy())
            
            # Siapkan fitur
            feature_cols = [col for col in ['Studi_Jurusan', 'Semester', 'AI_Tools', 'Trust_Level'] 
                          if col in df_encoded.columns]
            
            if len(feature_cols) > 0:
                X_student = df_encoded[feature_cols]
                
                # Predict
                predictions = st.session_state.model.predict(X_student)
                
                # Ambil prediksi terbaru
                latest_prediction = predictions[-1]
                latest_row = student_data.iloc[-1]
                
                # Tampilkan informasi
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info("**📋 Informasi Mahasiswa:**")
                    st.write(f"**Nama:** {latest_row['Nama']}")
                    st.write(f"**Jurusan:** {latest_row['Studi_Jurusan']}")
                    st.write(f"**Semester:** {latest_row['Semester']}")
                    st.write(f"**AI Tools:** {latest_row['AI_Tools']}")
                    
                    if 'Trust_Level' in latest_row:
                        st.write(f"**Tingkat Kepercayaan:** {latest_row['Trust_Level']}/5")
                    if 'Usage_Intensity_Score' in latest_row:
                        st.write(f"**Intensitas Penggunaan:** {latest_row['Usage_Intensity_Score']}/10")
                
                with col2:
                    # Tampilkan klasifikasi dengan warna
                    st.info("**🎯 Klasifikasi Penggunaan AI:**")
                    
                    if latest_prediction == 'rendah':
                        st.error(f"**RENDAH** ⚠️")
                        st.write("*Intensitas penggunaan AI masih perlu ditingkatkan*")
                    elif latest_prediction == 'sedang':
                        st.warning(f"**SEDANG** 📊")
                        st.write("*Penggunaan AI sudah baik, bisa ditingkatkan lagi*")
                    else:
                        st.success(f"**TINGGI** 🚀")
                        st.write("*Penggunaan AI sudah optimal*")
                
                # Tampilkan rekomendasi
                st.info("**💡 Rekomendasi Analisis:**")
                trust_level = latest_row['Trust_Level'] if 'Trust_Level' in latest_row else 3
                usage_score = latest_row['Usage_Intensity_Score'] if 'Usage_Intensity_Score' in latest_row else 5
                
                recommendations = get_recommendation(latest_prediction, trust_level, usage_score)
                
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
                
                # Riwayat penggunaan
                if len(student_data) > 1:
                    st.info("**📈 Riwayat Penggunaan:**")
                    history_cols = ['Semester', 'AI_Tools', 'Trust_Level', 'Usage_Intensity_Score']
                    history_cols = [col for col in history_cols if col in student_data.columns]
                    st.dataframe(student_data[history_cols], use_container_width=True)
            else:
                st.warning("Data tidak lengkap untuk analisis")
        else:
            st.warning(f"Data tidak ditemukan untuk mahasiswa: {student_name}")
            st.write("Data yang tersedia:")
            st.dataframe(st.session_state.data[['Nama', 'Studi_Jurusan']].head(10))
    else:
        st.warning("Data atau model belum tersedia. Silakan guru/dosen melakukan analisis terlebih dahulu.")

def teacher_dashboard():
    st.title("👨‍🏫 Dashboard Guru/Dosen - Analisis Penggunaan AI")
    st.markdown("---")
    
    # Menu sidebar
    menu = st.sidebar.selectbox(
        "Menu Analisis",
        ["Upload & Cleaning Data", "Encoding Data", "Split Data", 
         "Uji dengan Random Forest", "Evaluasi & Rekomendasi"]
    )
    
    if menu == "Upload & Cleaning Data":
        st.header("📤 Upload & Data Cleaning")
        
        uploaded_file = st.file_uploader(
            "Upload file CSV dengan pemisah titik koma (;)",
            type=['csv'],
            help="Format: Nama;Studi_Jurusan;Semester;AI_Tools;Trust_Level;Usage_Intensity_Score"
        )
        
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state.data = clean_data(df)
                st.success(f"✅ Data berhasil di-load: {len(df)} baris, {len(df.columns)} kolom")
        else:
            # Gunakan dataset default
            if st.button("Gunakan Dataset Contoh"):
                df = load_data()
                if df is not None:
                    st.session_state.data = clean_data(df)
                    st.success(f"✅ Dataset contoh berhasil di-load: {len(df)} baris")
    
    elif menu == "Encoding Data" and st.session_state.data is not None:
        st.header("🔡 Encoding Data Kategorikal")
        df_encoded, encoders = encode_categorical_data(st.session_state.data)
        st.session_state.data = df_encoded
        st.session_state.encoders = encoders
        
    elif menu == "Split Data" and st.session_state.data is not None:
        st.header("✂️ Split Data Training & Testing")
        X, y, df_with_target = prepare_features_target(st.session_state.data)
        st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = split_data(X, y)
        st.session_state.data = df_with_target
        
    elif menu == "Uji dengan Random Forest" and st.session_state.data is not None:
        st.header("🌲 Uji Model dengan Random Forest")
        if hasattr(st.session_state, 'X_train'):
            model = train_random_forest(
                st.session_state.X_train,
                st.session_state.X_test,
                st.session_state.y_train,
                st.session_state.y_test
            )
        else:
            st.warning("Silakan lakukan split data terlebih dahulu")
            
    elif menu == "Evaluasi & Rekomendasi":
        evaluation_recommendation_page()

def main():
    # Sidebar dengan informasi
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1995/1995515.png", width=100)
        st.title("Analisis AI Mahasiswa")
        st.markdown("---")
        
        if st.session_state.logged_in:
            st.success(f"Logged in as: {st.session_state.user_role}")
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.user_role = None
                st.rerun()
        
        st.markdown("---")
        st.caption("""
        **Analisis Tingkat Penggunaan AI terhadap Performa Akademik Mahasiswa**
        
        Sistem ini menggunakan:
        - Data cleaning & preprocessing
        - Random Forest Algorithm
        - Knowledge-based recommendations
        """)
    
    # Main content
    if not st.session_state.logged_in:
        login_page()
    else:
        if st.session_state.user_role == 'guru':
            teacher_dashboard()
        else:
            student_dashboard()

if __name__ == "__main__":
    # Simpan dataset contoh jika tidak ada
    if not os.path.exists("Dataset_Klasifikasi_Pengguna_AI_Mahasiswa_UMMgl_PAKE.csv"):
        # Create sample data from the content provided
        sample_data = """Nama;Studi_Jurusan;Semester;AI_Tools;Trust_Level;Usage_Intensity_Score
Althaf Rayyan Putra;Teknologi Informasi;7;Gemini;4;8
Ayesha Kinanti;Teknologi Informasi;3;Gemini;4;9
Salsabila Nurfadila;Teknik Informatika;1;Gemini;5;3
Anindya Safira;Teknik Informatika;5;Gemini;4;6
Iqbal Ramadhan;Farmasi;1;Gemini;5;10
Muhammad Rizky Pratama;Teknologi Informasi;5;Gemini;4;4
Fikri Alfarizi;Teknologi Informasi;1;ChatGPT;4;7
Citra Maharani;Keperawatan;5;Multiple;4;2
Zidan Harits;Farmasi;7;Gemini;4;4"""
        
        with open("Dataset_Klasifikasi_Pengguna_AI_Mahasiswa_UMMgl_PAKE.csv", "w", encoding="utf-8") as f:
            f.write(sample_data)
    
    main()
