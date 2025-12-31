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
    if 'original_data' not in st.session_state:
        st.session_state.original_data = None
    if 'cleaned_data' not in st.session_state:
        st.session_state.cleaned_data = None
    if 'encoded_data' not in st.session_state:
        st.session_state.encoded_data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'encoders' not in st.session_state:
        st.session_state.encoders = {}
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
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
                if username:
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
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None
    else:
        # Try to load from session state or create sample
        if st.session_state.original_data is not None:
            return st.session_state.original_data.copy()
        else:
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
            
            df = pd.read_csv(pd.compat.StringIO(sample_data), sep=';')
            return df

def clean_data(df):
    st.subheader("📊 Data Cleaning")
    
    st.write("**Data Asli:**")
    st.dataframe(df.head(), use_container_width=True)
    
    df_cleaned = df.copy()
    
    # 1. Handle '10+' in Usage_Intensity_Score
    st.write("### 1. Normalisasi Usage_Intensity_Score")
    if 'Usage_Intensity_Score' in df_cleaned.columns:
        # Convert '10+' to 10
        df_cleaned['Usage_Intensity_Score'] = df_cleaned['Usage_Intensity_Score'].astype(str)
        df_cleaned['Usage_Intensity_Score'] = df_cleaned['Usage_Intensity_Score'].replace('10+', '10')
        # Convert to numeric
        df_cleaned['Usage_Intensity_Score'] = pd.to_numeric(df_cleaned['Usage_Intensity_Score'], errors='coerce')
    
    # 2. Convert Semester to numeric if it's not
    if 'Semester' in df_cleaned.columns:
        df_cleaned['Semester'] = pd.to_numeric(df_cleaned['Semester'], errors='coerce')
    
    # 3. Convert Trust_Level to numeric if it's not
    if 'Trust_Level' in df_cleaned.columns:
        df_cleaned['Trust_Level'] = pd.to_numeric(df_cleaned['Trust_Level'], errors='coerce')
    
    # 4. Drop rows with missing values in key columns
    key_cols = ['Usage_Intensity_Score', 'Trust_Level', 'Semester']
    key_cols = [col for col in key_cols if col in df_cleaned.columns]
    
    if key_cols:
        df_cleaned = df_cleaned.dropna(subset=key_cols)
    
    st.write("**Data Setelah Cleaning:**")
    st.dataframe(df_cleaned.head(), use_container_width=True)
    
    st.write(f"**Informasi Dataset:**")
    st.write(f"- Jumlah baris: {len(df_cleaned)}")
    st.write(f"- Jumlah kolom: {len(df_cleaned.columns)}")
    st.write(f"- Kolom: {', '.join(df_cleaned.columns.tolist())}")
    
    return df_cleaned

def encode_categorical_data(df):
    st.subheader("🔡 Encoding Data Kategorikal")
    
    # Identifikasi kolom kategorikal
    categorical_cols = []
    for col in df.columns:
        if col != 'Nama' and df[col].dtype == 'object':
            categorical_cols.append(col)
    
    if not categorical_cols:
        st.info("Tidak ada kolom kategorikal yang perlu di-encode.")
        return df, {}
    
    encoded_df = df.copy()
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        # Handle missing values before encoding
        encoded_df[col] = encoded_df[col].fillna('Unknown')
        encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
        encoders[col] = le
        
        # Show mapping
        st.write(f"**Encoding untuk {col}:**")
        try:
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            mapping_df = pd.DataFrame(list(mapping.items()), columns=['Nilai Asli', 'Kode'])
            st.dataframe(mapping_df.head(), use_container_width=True)
        except:
            st.write(f"Tidak dapat menampilkan mapping untuk {col}")
    
    st.session_state.encoders = encoders
    
    st.write("**Data Setelah Encoding (5 baris pertama):**")
    st.dataframe(encoded_df.head(), use_container_width=True)
    
    return encoded_df, encoders

def prepare_features_target(df):
    st.subheader("🎯 Persiapan Fitur dan Target")
    
    df_prepared = df.copy()
    
    # Buat target klasifikasi berdasarkan Usage_Intensity_Score
    if 'Usage_Intensity_Score' in df_prepared.columns:
        # Pastikan kolom ini numeric
        df_prepared['Usage_Intensity_Score'] = pd.to_numeric(df_prepared['Usage_Intensity_Score'], errors='coerce')
        
        # Buat kelas target
        df_prepared['AI_Usage_Class'] = pd.cut(
            df_prepared['Usage_Intensity_Score'],
            bins=[0, 4, 7, 11],
            labels=['rendah', 'sedang', 'tinggi']
        )
        
        # Drop rows with NaN in target
        df_prepared = df_prepared.dropna(subset=['AI_Usage_Class'])
    else:
        st.error("Kolom 'Usage_Intensity_Score' tidak ditemukan!")
        return None, None, df
    
    # Pilih fitur yang akan digunakan
    possible_features = ['Studi_Jurusan', 'Semester', 'AI_Tools', 'Trust_Level']
    available_features = [col for col in possible_features if col in df_prepared.columns]
    
    if not available_features:
        st.error("Tidak ada fitur yang tersedia untuk training!")
        return None, None, df
    
    X = df_prepared[available_features]
    y = df_prepared['AI_Usage_Class']
    
    st.write("**Fitur yang digunakan:**", available_features)
    st.write(f"**Jumlah sampel:** {len(X)}")
    
    # Tampilkan distribusi kelas
    st.write("**Distribusi Kelas Target:**")
    class_dist = y.value_counts()
    st.bar_chart(class_dist)
    
    # Tampilkan tabel distribusi
    dist_df = pd.DataFrame({
        'Kelas': class_dist.index,
        'Jumlah': class_dist.values,
        'Persentase': (class_dist.values / len(y) * 100).round(2)
    })
    st.dataframe(dist_df, use_container_width=True)
    
    return X, y, df_prepared

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
    
    if X_train is None or X_test is None or y_train is None or y_test is None:
        st.error("Data tidak tersedia untuk training!")
        return None
    
    # Hyperparameter tuning
    col1, col2, col3 = st.columns(3)
    with col1:
        n_estimators = st.slider("Jumlah Trees", 50, 300, 100, 10)
    with col2:
        max_depth = st.slider("Kedalaman Maksimum", 5, 50, 20, 5)
    with col3:
        random_state = st.number_input("Random State", 0, 100, 42)
    
    if st.button("🚀 Train Model", use_container_width=True):
        with st.spinner("Training model Random Forest..."):
            try:
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
                    'feature_names': X_train.columns.tolist(),
                    'model_params': {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'random_state': random_state
                    }
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
                
                # Display feature importance table
                st.dataframe(feature_importance, use_container_width=True)
                
                # Classification Report
                st.write("### 📋 Classification Report")
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)
                
                # Confusion Matrix
                st.write("### 🎯 Confusion Matrix")
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_test, y_pred, labels=['rendah', 'sedang', 'tinggi'])
                cm_df = pd.DataFrame(cm, 
                                    index=['Aktual: rendah', 'Aktual: sedang', 'Aktual: tinggi'],
                                    columns=['Prediksi: rendah', 'Prediksi: sedang', 'Prediksi: tinggi'])
                st.dataframe(cm_df, use_container_width=True)
                
                return model
                
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
                return None
    
    return None

def get_recommendation(usage_class, trust_level, usage_score):
    """Generate recommendation based on knowledgebase rules"""
    recommendations = []
    
    # Get base recommendations from knowledgebase
    if usage_class in KNOWLEDGEBASE_RULES:
        recommendations.extend(KNOWLEDGEBASE_RULES[usage_class]['recommendations'])
    
    # Tambahkan rekomendasi spesifik berdasarkan trust level
    if trust_level <= 2:
        recommendations.append("Fokus pada peningkatan kepercayaan terhadap teknologi AI")
    elif trust_level >= 4:
        recommendations.append("Manfaatkan kepercayaan tinggi untuk eksplorasi tools lebih lanjut")
    
    # Tambahkan rekomendasi berdasarkan usage score
    if isinstance(usage_score, (int, float)):
        if usage_score <= 4:
            recommendations.append("Set target peningkatan penggunaan mingguan")
        elif usage_score >= 8:
            recommendations.append("Pertimbangkan untuk menjadi mentor AI bagi teman sekelas")
    
    return list(set(recommendations))[:4]  # Ambil 4 rekomendasi unik

def evaluation_recommendation_page():
    st.title("📈 Evaluasi & Rekomendasi")
    st.markdown("---")
    
    if st.session_state.model is None:
        st.warning("Silakan train model terlebih dahulu di menu 'Uji dengan Random Forest'")
        return
    
    if st.session_state.cleaned_data is None:
        st.warning("Data tidak tersedia. Silakan lakukan preprocessing data terlebih dahulu.")
        return
    
    # Tampilkan evaluasi model
    st.subheader("📊 Evaluasi Model")
    
    if st.session_state.results:
        results = st.session_state.results
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Akurasi Model", f"{results['accuracy']:.2%}")
        with col2:
            st.metric("Parameter Trees", results['model_params']['n_estimators'])
        with col3:
            st.metric("Kedalaman", results['model_params']['max_depth'])
    
    # Prediksi untuk semua data
    st.subheader("🎯 Prediksi & Rekomendasi untuk Semua Mahasiswa")
    
    # Gunakan data asli untuk tampilan
    if st.session_state.original_data is not None:
        display_data = st.session_state.original_data.copy()
    else:
        display_data = st.session_state.cleaned_data.copy()
    
    # Gunakan data encoded untuk prediksi
    if st.session_state.encoded_data is not None:
        predict_data = st.session_state.encoded_data.copy()
    else:
        # Encode jika belum diencode
        predict_data, _ = encode_categorical_data(st.session_state.cleaned_data.copy())
    
    # Siapkan fitur untuk prediksi
    feature_cols = [col for col in ['Studi_Jurusan', 'Semester', 'AI_Tools', 'Trust_Level'] 
                   if col in predict_data.columns]
    
    if not feature_cols:
        st.error("Fitur yang diperlukan tidak ditemukan untuk prediksi!")
        return
    
    # Pastikan semua fitur ada di display_data untuk rekomendasi
    missing_in_display = [col for col in feature_cols if col not in display_data.columns]
    for col in missing_in_display:
        if col in predict_data.columns:
            display_data[col] = predict_data[col]
    
    # Lakukan prediksi
    try:
        X_all = predict_data[feature_cols]
        predictions = st.session_state.model.predict(X_all)
        
        # Tambahkan hasil ke display_data
        display_data['Klasifikasi_Penggunaan_AI'] = predictions
        
        # Tambahkan rekomendasi
        recommendations = []
        for idx, row in display_data.iterrows():
            usage_class = predictions[idx]
            
            # Get trust level
            trust_level = 3  # default
            if 'Trust_Level' in row:
                try:
                    trust_level = float(row['Trust_Level'])
                except:
                    trust_level = 3
            
            # Get usage score
            usage_score = 5  # default
            if 'Usage_Intensity_Score' in row:
                try:
                    score_str = str(row['Usage_Intensity_Score'])
                    if score_str == '10+':
                        usage_score = 10
                    else:
                        usage_score = float(score_str)
                except:
                    usage_score = 5
            
            # Get recommendations
            recs = get_recommendation(usage_class, trust_level, usage_score)
            recommendations.append(", ".join(recs))
        
        display_data['Rekomendasi'] = recommendations
        
        # Tampilkan hasil
        st.write(f"**Total Mahasiswa:** {len(display_data)}")
        
        # Tampilkan tabel dengan hasil
        display_cols = ['Nama']
        if 'Studi_Jurusan' in display_data.columns:
            display_cols.append('Studi_Jurusan')
        if 'Semester' in display_data.columns:
            display_cols.append('Semester')
        if 'AI_Tools' in display_data.columns:
            display_cols.append('AI_Tools')
        if 'Trust_Level' in display_data.columns:
            display_cols.append('Trust_Level')
        if 'Usage_Intensity_Score' in display_data.columns:
            display_cols.append('Usage_Intensity_Score')
        
        display_cols.extend(['Klasifikasi_Penggunaan_AI', 'Rekomendasi'])
        
        st.dataframe(display_data[display_cols], use_container_width=True, height=400)
        
        # Statistik klasifikasi
        st.subheader("📊 Statistik Klasifikasi")
        
        class_counts = display_data['Klasifikasi_Penggunaan_AI'].value_counts()
        
        col1, col2, col3 = st.columns(3)
        colors = {'rendah': 'red', 'sedang': 'orange', 'tinggi': 'green'}
        
        with col1:
            if 'rendah' in class_counts:
                count = class_counts['rendah']
                percentage = (count / len(display_data) * 100)
                st.metric("Rendah", f"{count} ({percentage:.1f}%)")
            else:
                st.metric("Rendah", "0")
        
        with col2:
            if 'sedang' in class_counts:
                count = class_counts['sedang']
                percentage = (count / len(display_data) * 100)
                st.metric("Sedang", f"{count} ({percentage:.1f}%)")
            else:
                st.metric("Sedang", "0")
        
        with col3:
            if 'tinggi' in class_counts:
                count = class_counts['tinggi']
                percentage = (count / len(display_data) * 100)
                st.metric("Tinggi", f"{count} ({percentage:.1f}%)")
            else:
                st.metric("Tinggi", "0")
        
        # Chart distribusi
        st.bar_chart(class_counts)
        
        # Tombol download
        csv = display_data.to_csv(index=False, sep=';').encode('utf-8')
        st.download_button(
            label="📥 Download Hasil Analisis (CSV)",
            data=csv,
            file_name=f"hasil_analisis_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    except Exception as e:
        st.error(f"Error dalam prediksi: {str(e)}")

def student_dashboard():
    st.title("👨‍🎓 Dashboard Mahasiswa")
    st.markdown("---")
    
    student_name = st.session_state.get('student_name', 'Mahasiswa')
    st.success(f"Halo {student_name}! Berikut adalah hasil analisis penggunaan AI:")
    
    if st.session_state.model is None:
        st.warning("Model belum dilatih. Silakan guru/dosen melakukan training model terlebih dahulu.")
        return
    
    # Cari data mahasiswa berdasarkan nama (partial match)
    if st.session_state.original_data is not None:
        search_data = st.session_state.original_data.copy()
    elif st.session_state.cleaned_data is not None:
        search_data = st.session_state.cleaned_data.copy()
    else:
        st.warning("Data tidak tersedia.")
        return
    
    # Cari mahasiswa dengan nama yang mirip
    student_matches = search_data[search_data['Nama'].str.contains(student_name, case=False, na=False)]
    
    if student_matches.empty:
        st.warning(f"Data tidak ditemukan untuk mahasiswa: {student_name}")
        st.write("**Mahasiswa yang tersedia dalam dataset:**")
        st.dataframe(search_data[['Nama', 'Studi_Jurusan']].head(10), use_container_width=True)
        return
    
    # Ambil data pertama yang cocok
    student_row = student_matches.iloc[0]
    
    # Prepare data for prediction
    if st.session_state.encoded_data is not None:
        predict_data = st.session_state.encoded_data.copy()
        # Find the corresponding encoded row
        encoded_idx = predict_data.index[predict_data.index.isin(student_matches.index)]
        if len(encoded_idx) > 0:
            encoded_row = predict_data.loc[encoded_idx[0]]
        else:
            # Encode the student data
            temp_df = pd.DataFrame([student_row])
            encoded_temp, _ = encode_categorical_data(temp_df)
            encoded_row = encoded_temp.iloc[0]
    else:
        # Encode the student data
        temp_df = pd.DataFrame([student_row])
        encoded_temp, _ = encode_categorical_data(temp_df)
        encoded_row = encoded_temp.iloc[0]
    
    # Prepare features for prediction
    feature_cols = [col for col in ['Studi_Jurusan', 'Semester', 'AI_Tools', 'Trust_Level'] 
                   if col in encoded_row.index]
    
    if not feature_cols:
        st.error("Tidak dapat melakukan prediksi: fitur tidak tersedia")
        return
    
    # Create feature vector
    X_student = pd.DataFrame([encoded_row[feature_cols]])
    
    # Predict
    try:
        prediction = st.session_state.model.predict(X_student)[0]
        
        # Display student info
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**📋 Informasi Mahasiswa:**")
            st.write(f"**Nama:** {student_row['Nama']}")
            if 'Studi_Jurusan' in student_row:
                st.write(f"**Jurusan:** {student_row['Studi_Jurusan']}")
            if 'Semester' in student_row:
                st.write(f"**Semester:** {student_row['Semester']}")
            if 'AI_Tools' in student_row:
                st.write(f"**AI Tools:** {student_row['AI_Tools']}")
            if 'Trust_Level' in student_row:
                st.write(f"**Tingkat Kepercayaan:** {student_row['Trust_Level']}/5")
            if 'Usage_Intensity_Score' in student_row:
                st.write(f"**Intensitas Penggunaan:** {student_row['Usage_Intensity_Score']}/10")
        
        with col2:
            st.info("**🎯 Klasifikasi Penggunaan AI:**")
            
            if prediction == 'rendah':
                st.error(f"**RENDAH** ⚠️")
                st.write("*Intensitas penggunaan AI masih perlu ditingkatkan*")
            elif prediction == 'sedang':
                st.warning(f"**SEDANG** 📊")
                st.write("*Penggunaan AI sudah baik, bisa ditingkatkan lagi*")
            else:
                st.success(f"**TINGGI** 🚀")
                st.write("*Penggunaan AI sudah optimal*")
        
        # Get recommendations
        st.info("**💡 Rekomendasi Analisis:**")
        
        trust_level = 3
        if 'Trust_Level' in student_row:
            try:
                trust_level = float(student_row['Trust_Level'])
            except:
                pass
        
        usage_score = 5
        if 'Usage_Intensity_Score' in student_row:
            try:
                score_str = str(student_row['Usage_Intensity_Score'])
                if score_str == '10+':
                    usage_score = 10
                else:
                    usage_score = float(score_str)
            except:
                pass
        
        recommendations = get_recommendation(prediction, trust_level, usage_score)
        
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
        
    except Exception as e:
        st.error(f"Error dalam prediksi: {str(e)}")

def teacher_dashboard():
    st.title("👨‍🏫 Dashboard Guru/Dosen")
    st.markdown("---")
    
    # Menu sidebar
    menu = st.sidebar.selectbox(
        "Menu Analisis",
        ["Upload Data", "Data Cleaning", "Encoding Data", 
         "Split Data", "Uji dengan Random Forest", "Evaluasi & Rekomendasi"]
    )
    
    if menu == "Upload Data":
        st.header("📤 Upload Data")
        
        uploaded_file = st.file_uploader(
            "Upload file CSV dengan pemisah titik koma (;)",
            type=['csv'],
            help="Format: Nama;Studi_Jurusan;Semester;AI_Tools;Trust_Level;Usage_Intensity_Score"
        )
        
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state.original_data = df
                st.session_state.cleaned_data = df.copy()
                st.success(f"✅ Data berhasil di-load: {len(df)} baris, {len(df.columns)} kolom")
                st.dataframe(df.head(), use_container_width=True)
        else:
            if st.button("Gunakan Dataset Contoh", use_container_width=True):
                df = load_data()
                if df is not None:
                    st.session_state.original_data = df
                    st.session_state.cleaned_data = df.copy()
                    st.success(f"✅ Dataset contoh berhasil di-load: {len(df)} baris")
                    st.dataframe(df.head(), use_container_width=True)
    
    elif menu == "Data Cleaning":
        st.header("🧹 Data Cleaning")
        
        if st.session_state.original_data is None:
            st.warning("Silakan upload data terlebih dahulu di menu 'Upload Data'")
        else:
            df_cleaned = clean_data(st.session_state.original_data)
            st.session_state.cleaned_data = df_cleaned
            st.success("✅ Data cleaning selesai!")
    
    elif menu == "Encoding Data":
        st.header("🔡 Encoding Data Kategorikal")
        
        if st.session_state.cleaned_data is None:
            st.warning("Silakan lakukan data cleaning terlebih dahulu")
        else:
            df_encoded, encoders = encode_categorical_data(st.session_state.cleaned_data)
            st.session_state.encoded_data = df_encoded
            st.session_state.encoders = encoders
            st.success("✅ Encoding selesai!")
    
    elif menu == "Split Data":
        st.header("✂️ Split Data")
        
        if st.session_state.encoded_data is None:
            st.warning("Silakan lakukan encoding data terlebih dahulu")
        else:
            X, y, df_with_target = prepare_features_target(st.session_state.encoded_data)
            
            if X is not None and y is not None:
                st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = split_data(X, y)
                st.session_state.encoded_data = df_with_target
                st.success("✅ Data split selesai!")
    
    elif menu == "Uji dengan Random Forest":
        st.header("🌲 Uji dengan Random Forest")
        
        if st.session_state.X_train is None:
            st.warning("Silakan lakukan split data terlebih dahulu")
        else:
            model = train_random_forest(
                st.session_state.X_train,
                st.session_state.X_test,
                st.session_state.y_train,
                st.session_state.y_test
            )
            
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
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                init_session_state()
                st.rerun()
        else:
            st.info("Silakan login terlebih dahulu")
        
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
    main()
