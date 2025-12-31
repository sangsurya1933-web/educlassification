import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set halaman
st.set_page_config(
    page_title="Analisis Penggunaan AI - Performa Akademik",
    page_icon="🎓",
    layout="wide"
)

# Fungsi untuk membuat dataset contoh
def create_sample_dataset():
    data = {
        'Nama': [f'Mahasiswa_{i}' for i in range(1, 101)],
        'Usia': np.random.randint(18, 25, 100),
        'Jenis_Kelamin': np.random.choice(['Laki-laki', 'Perempuan'], 100),
        'Fakultas': np.random.choice(['Teknik', 'Sains', 'Ekonomi', 'Kedokteran', 'Hukum'], 100),
        'Semester': np.random.randint(1, 8, 100),
        'IPK': np.round(np.random.uniform(2.0, 4.0, 100), 2),
        'Jam_Belajar_Mingguan': np.random.randint(10, 40, 100),
        'Frekuensi_Penggunaan_AI': np.random.choice(['Rendah', 'Sedang', 'Tinggi'], 100, p=[0.3, 0.5, 0.2]),
        'Tujuan_Penggunaan_AI': np.random.choice(['Tugas', 'Penelitian', 'Belajar', 'Proyek'], 100),
        'Tingkat_Ketergantungan_AI': np.random.choice(['Rendah', 'Sedang', 'Tinggi'], 100),
        'Perform_Akademik': np.random.choice(['Buruk', 'Cukup', 'Baik', 'Sangat Baik'], 100)
    }
    return pd.DataFrame(data)

# Fungsi untuk preprocessing data
def preprocess_data(df):
    df_clean = df.copy()
    
    # Data Cleaning
    st.subheader("Data Cleaning")
    
    # Cek missing values
    st.write("### Missing Values sebelum Cleaning:")
    missing_values = df_clean.isnull().sum()
    st.write(missing_values[missing_values > 0])
    
    # Isi missing values dengan mode untuk kategorikal dan median untuk numerik
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Tidak Diketahui", inplace=True)
        else:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    st.write("### Missing Values setelah Cleaning:")
    st.write(df_clean.isnull().sum().sum(), "missing values tersisa")
    
    # Encoding data kategorikal
    st.subheader("Encoding Data Kategorikal")
    
    label_encoders = {}
    df_encoded = df_clean.copy()
    
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'Nama':  # Tidak encode kolom nama
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le
    
    st.write("### Data setelah Encoding:")
    st.write(df_encoded.head())
    
    return df_clean, df_encoded, label_encoders

# Fungsi untuk split data
def split_data(df_encoded):
    st.subheader("Split Data")
    
    # Pilih target variable
    target_options = ['Frekuensi_Penggunaan_AI', 'Tingkat_Ketergantungan_AI', 'Perform_Akademik']
    target = st.selectbox("Pilih target variable untuk prediksi:", target_options)
    
    # Pilih fitur
    feature_options = [col for col in df_encoded.columns if col not in ['Nama', target]]
    selected_features = st.multiselect(
        "Pilih fitur untuk model:",
        feature_options,
        default=[col for col in ['Usia', 'IPK', 'Jam_Belajar_Mingguan'] if col in feature_options]
    )
    
    if not selected_features:
        st.warning("Pilih minimal satu fitur!")
        return None, None, None, None, target
    
    X = df_encoded[selected_features]
    y = df_encoded[target]
    
    # Split data
    test_size = st.slider("Ukuran data testing (%):", 10, 40, 20) / 100
    random_state = st.number_input("Random state:", min_value=0, value=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    st.write(f"### Dimensi Data:")
    st.write(f"- Data training: {X_train.shape[0]} sampel, {X_train.shape[1]} fitur")
    st.write(f"- Data testing: {X_test.shape[0]} sampel")
    
    return X_train, X_test, y_train, y_test, target, selected_features

# Fungsi untuk training model Random Forest
def train_random_forest(X_train, X_test, y_train, y_test, target):
    st.subheader("Training Model Random Forest")
    
    # Parameter model
    col1, col2, col3 = st.columns(3)
    with col1:
        n_estimators = st.slider("Jumlah estimator:", 10, 200, 100)
    with col2:
        max_depth = st.slider("Max depth:", 2, 20, 10)
    with col3:
        min_samples_split = st.slider("Min samples split:", 2, 10, 2)
    
    # Training model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Prediksi
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    # Evaluasi
    accuracy_test = accuracy_score(y_test, y_pred)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    
    st.write("### Hasil Evaluasi Model:")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Akurasi Training", f"{accuracy_train:.2%}")
    with col2:
        st.metric("Akurasi Testing", f"{accuracy_test:.2%}")
    
    # Classification report
    st.write("### Classification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)
    
    # Confusion matrix
    st.write("### Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)
    
    # Feature importance
    st.write("### Feature Importance:")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature', ax=ax2)
    ax2.set_title('Feature Importance')
    st.pyplot(fig2)
    
    return model, y_pred, accuracy_test

# Fungsi untuk menghasilkan rekomendasi
def generate_recommendations(prediction, target):
    recommendations = {
        'Rendah': {
            'Frekuensi_Penggunaan_AI': "**LEVEL AMAN** - Penggunaan AI masih dalam batas wajar. Pertahankan dan gunakan AI secara bijak untuk meningkatkan produktivitas.",
            'Tingkat_Ketergantungan_AI': "**LEVEL AMAN** - Ketergantungan pada AI masih rendah. Pertahankan keseimbangan antara penggunaan AI dan kemampuan pribadi.",
            'Perform_Akademik': "**PERHATIAN** - Performa akademik rendah. Disarankan untuk meningkatkan jam belajar dan memanfaatkan AI sebagai alat bantu belajar."
        },
        'Sedang': {
            'Frekuensi_Penggunaan_AI': "**PERLU TEGURAN** - Penggunaan AI sudah mulai sering. Evaluasi kembali kebutuhan penggunaan AI dan pastikan tidak mengurangi kemampuan analisis mandiri.",
            'Tingkat_Ketergantungan_AI': "**PERLU TEGURAN** - Ketergantungan pada AI sedang. Disarankan untuk mengurangi ketergantungan dan mengembangkan kemampuan problem-solving mandiri.",
            'Perform_Akademik': "**CUKUP BAIK** - Performa akademik sedang. Pertahankan dan coba optimalkan dengan strategi belajar yang lebih efektif."
        },
        'Tinggi': {
            'Frekuensi_Penggunaan_AI': "**BUTUH PENGAWASAN LEBIH** - Penggunaan AI sangat tinggi. Perlu evaluasi menyeluruh dan batasi penggunaan AI agar tidak mengganggu pengembangan kemampuan kritis.",
            'Tingkat_Ketergantungan_AI': "**BUTUH PENGAWASAN LEBIH** - Ketergantungan pada AI sangat tinggi. Sangat disarankan untuk konsultasi dengan dosen pembimbing dan mengurangi ketergantungan secara bertahap.",
            'Perform_Akademik': "**SANGAT BAIK** - Performa akademik tinggi. Pertahankan prestasi dan gunakan AI sebagai alat pendukung, bukan pengganti kemampuan kognitif."
        }
    }
    
    # Mapping untuk kategori numerik ke string
    if isinstance(prediction, (int, np.integer)):
        if prediction == 0:
            prediction_str = 'Rendah'
        elif prediction == 1:
            prediction_str = 'Sedang'
        elif prediction == 2:
            prediction_str = 'Tinggi'
        else:
            prediction_str = 'Sedang'  # default
    else:
        prediction_str = prediction
    
    return recommendations.get(prediction_str, {}).get(target, "Rekomendasi tidak tersedia.")

# Halaman Login
def login_page():
    st.title("🎓 Sistem Analisis Penggunaan AI - Performa Akademik")
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://cdn.pixabay.com/photo/2017/08/01/00/38/man-2562325_1280.jpg", use_column_width=True)
    
    st.markdown("### Login ke Sistem")
    
    login_type = st.radio("Login sebagai:", ["Guru/Admin", "Mahasiswa"])
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if login_type == "Guru/Admin":
            if username == "guru" and password == "guru123":
                st.session_state.logged_in = True
                st.session_state.user_type = "guru"
                st.success("Login berhasil! Mengarahkan ke dashboard...")
                st.rerun()
            else:
                st.error("Username atau password salah untuk akun guru!")
        else:  # Mahasiswa
            if username and password == "mahasiswa123":
                st.session_state.logged_in = True
                st.session_state.user_type = "mahasiswa"
                st.session_state.student_name = username
                st.success(f"Login berhasil! Selamat datang {username}")
                st.rerun()
            else:
                st.error("Password harus 'mahasiswa123' untuk akun mahasiswa!")
    
    # Info login
    st.info("""
    **Credential Login:**
    - Guru/Admin: Username: `guru`, Password: `guru123`
    - Mahasiswa: Username: `Nama Anda`, Password: `mahasiswa123`
    """)

# Dashboard Guru
def guru_dashboard():
    st.sidebar.title("🎓 Dashboard Guru")
    st.sidebar.markdown("---")
    
    menu = st.sidebar.selectbox(
        "Menu",
        ["Data Awal", "Preprocessing & Cleaning", "Model Random Forest", "Evaluasi & Rekomendasi", "Logout"]
    )
    
    if menu == "Logout":
        st.session_state.logged_in = False
        st.session_state.user_type = None
        st.rerun()
    
    st.title("📊 Dashboard Analisis - Guru/Admin")
    st.markdown("---")
    
    # Inisialisasi session state untuk data
    if 'df' not in st.session_state:
        st.session_state.df = create_sample_dataset()
    
    if 'df_clean' not in st.session_state:
        st.session_state.df_clean = None
    
    if 'df_encoded' not in st.session_state:
        st.session_state.df_encoded = None
    
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    
    if 'target' not in st.session_state:
        st.session_state.target = None
    
    # Menu Data Awal
    if menu == "Data Awal":
        st.header("Data Awal Dataset")
        
        # Upload atau gunakan dataset contoh
        st.subheader("Upload Dataset atau Gunakan Data Contoh")
        
        uploaded_file = st.file_uploader("Upload file CSV dataset", type=['csv'])
        
        if uploaded_file is not None:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.success("Dataset berhasil diupload!")
        else:
            st.info("Menggunakan dataset contoh. Silakan upload dataset CSV jika ingin menggunakan data sendiri.")
        
        # Tampilkan data
        st.subheader("Preview Dataset")
        st.write(f"Shape dataset: {st.session_state.df.shape}")
        st.dataframe(st.session_state.df.head(10))
        
        # Statistik deskriptif
        st.subheader("Statistik Deskriptif")
        st.write(st.session_state.df.describe())
        
        # Distribusi target
        st.subheader("Distribusi Variabel Target")
        
        target_options = ['Frekuensi_Penggunaan_AI', 'Tingkat_Ketergantungan_AI', 'Perform_Akademik']
        target_to_analyze = st.selectbox("Pilih variabel untuk dianalisis:", target_options)
        
        if target_to_analyze in st.session_state.df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            st.session_state.df[target_to_analyze].value_counts().plot(kind='bar', ax=ax)
            ax.set_title(f'Distribusi {target_to_analyze}')
            ax.set_xlabel(target_to_analyze)
            ax.set_ylabel('Jumlah')
            st.pyplot(fig)
    
    # Menu Preprocessing & Cleaning
    elif menu == "Preprocessing & Cleaning":
        st.header("Preprocessing & Data Cleaning")
        
        if st.session_state.df is not None:
            # Preprocessing data
            df_clean, df_encoded, label_encoders = preprocess_data(st.session_state.df)
            
            # Simpan ke session state
            st.session_state.df_clean = df_clean
            st.session_state.df_encoded = df_encoded
            st.session_state.label_encoders = label_encoders
            
            # Tampilkan perbandingan sebelum dan sesudah
            st.subheader("Perbandingan Sebelum dan Sesudah Preprocessing")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Data Asli (5 baris pertama):**")
                st.write(st.session_state.df.head())
            with col2:
                st.write("**Data Setelah Cleaning (5 baris pertama):**")
                st.write(df_clean.head())
            
            # Informasi encoding
            if label_encoders:
                st.subheader("Mapping Encoding")
                for col, le in label_encoders.items():
                    st.write(f"**{col}:**")
                    classes = le.classes_
                    for i, cls in enumerate(classes):
                        st.write(f"  {cls} → {i}")
        else:
            st.warning("Silakan upload atau gunakan dataset terlebih dahulu di menu 'Data Awal'")
    
    # Menu Model Random Forest
    elif menu == "Model Random Forest":
        st.header("Uji Model Random Forest")
        
        if st.session_state.df_encoded is not None:
            # Split data
            X_train, X_test, y_train, y_test, target, selected_features = split_data(st.session_state.df_encoded)
            
            if X_train is not None:
                st.session_state.target = target
                st.session_state.selected_features = selected_features
                
                # Training model
                if st.button("Train Model Random Forest"):
                    with st.spinner("Training model..."):
                        model, predictions, accuracy = train_random_forest(X_train, X_test, y_train, y_test, target)
                        
                        # Simpan model dan hasil prediksi
                        st.session_state.model = model
                        st.session_state.X_test = X_test
                        st.session_state.y_test = y_test
                        st.session_state.predictions = predictions
                        st.session_state.accuracy = accuracy
                        
                        # Simpan prediksi untuk setiap mahasiswa
                        df_results = st.session_state.df_clean.copy()
                        # Ambil indeks untuk data testing
                        test_indices = X_test.index
                        
                        # Buat kolom prediksi
                        df_results['Prediksi'] = ''
                        df_results['Rekomendasi'] = ''
                        
                        for i, idx in enumerate(test_indices):
                            if i < len(predictions):
                                df_results.loc[idx, 'Prediksi'] = predictions[i]
                                df_results.loc[idx, 'Rekomendasi'] = generate_recommendations(predictions[i], target)
                        
                        st.session_state.df_results = df_results
                        
                        st.success("Model berhasil ditraining!")
        else:
            st.warning("Silakan lakukan preprocessing data terlebih dahulu di menu 'Preprocessing & Cleaning'")
    
    # Menu Evaluasi & Rekomendasi
    elif menu == "Evaluasi & Rekomendasi":
        st.header("Evaluasi & Rekomendasi")
        
        if st.session_state.model is not None and st.session_state.df_results is not None:
            st.subheader("Hasil Prediksi dan Rekomendasi")
            
            # Tampilkan akurasi
            st.metric("Akurasi Model", f"{st.session_state.accuracy:.2%}")
            
            # Filter berdasarkan prediksi
            st.subheader("Filter Berdasarkan Tingkat Prediksi")
            
            # Mapping prediksi numerik ke label
            if st.session_state.target in st.session_state.label_encoders:
                le = st.session_state.label_encoders[st.session_state.target]
                pred_mapping = {i: label for i, label in enumerate(le.classes_)}
            else:
                pred_mapping = {0: 'Rendah', 1: 'Sedang', 2: 'Tinggi'}
            
            pred_filter = st.selectbox(
                "Pilih tingkat prediksi:",
                ["Semua"] + list(pred_mapping.values())
            )
            
            # Filter data
            if pred_filter == "Semua":
                filtered_df = st.session_state.df_results[st.session_state.df_results['Prediksi'] != '']
            else:
                # Cari key yang sesuai dengan label
                pred_key = [k for k, v in pred_mapping.items() if v == pred_filter]
                if pred_key:
                    filtered_df = st.session_state.df_results[st.session_state.df_results['Prediksi'] == pred_key[0]]
                else:
                    filtered_df = st.session_state.df_results[st.session_state.df_results['Prediksi'] != '']
            
            # Tampilkan hasil
            st.write(f"**Jumlah Mahasiswa: {len(filtered_df)}**")
            
            # Tampilkan dalam bentuk tabel
            display_cols = ['Nama', 'IPK', 'Frekuensi_Penggunaan_AI', 
                          'Tingkat_Ketergantungan_AI', 'Perform_Akademik', 'Prediksi', 'Rekomendasi']
            
            # Filter kolom yang ada
            display_cols = [col for col in display_cols if col in filtered_df.columns]
            
            # Konversi prediksi numerik ke label
            display_df = filtered_df[display_cols].copy()
            if 'Prediksi' in display_df.columns:
                display_df['Prediksi'] = display_df['Prediksi'].apply(
                    lambda x: pred_mapping.get(x, x) if pd.notnull(x) else x
                )
            
            st.dataframe(display_df)
            
            # Statistik rekomendasi
            st.subheader("Statistik Rekomendasi")
            
            if 'Prediksi' in filtered_df.columns:
                pred_counts = display_df['Prediksi'].value_counts()
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Pie chart
                ax1.pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%', startangle=90)
                ax1.set_title('Distribusi Tingkat Prediksi')
                
                # Bar chart
                colors = {'Rendah': 'green', 'Sedang': 'orange', 'Tinggi': 'red'}
                bar_colors = [colors.get(label, 'gray') for label in pred_counts.index]
                ax2.bar(pred_counts.index, pred_counts.values, color=bar_colors)
                ax2.set_title('Jumlah per Tingkat Prediksi')
                ax2.set_ylabel('Jumlah Mahasiswa')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Download hasil
            st.subheader("Download Hasil Analisis")
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download hasil sebagai CSV",
                data=csv,
                file_name=f"hasil_analisis_ai_{pred_filter.lower()}.csv",
                mime="text/csv",
            )
        else:
            st.warning("Silakan train model terlebih dahulu di menu 'Model Random Forest'")

# Dashboard Mahasiswa
def mahasiswa_dashboard():
    st.sidebar.title("👨‍🎓 Dashboard Mahasiswa")
    st.sidebar.markdown("---")
    
    # Info mahasiswa
    if 'student_name' in st.session_state:
        st.sidebar.write(f"Nama: **{st.session_state.student_name}**")
    
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_type = None
        st.session_state.student_name = None
        st.rerun()
    
    st.title("📋 Hasil Analisis Penggunaan AI")
    st.markdown("---")
    
    # Cek apakah ada hasil prediksi
    if 'df_results' in st.session_state and st.session_state.df_results is not None:
        # Cari data mahasiswa berdasarkan nama
        student_name = st.session_state.student_name
        
        # Cari mahasiswa dengan nama yang mirip
        student_data = None
        for name in st.session_state.df_results['Nama']:
            if student_name.lower() in name.lower():
                student_data = st.session_state.df_results[st.session_state.df_results['Nama'] == name]
                break
        
        if student_data is not None and not student_data.empty:
            student_data = student_data.iloc[0]
            
            st.subheader(f"Hasil Analisis untuk {student_data['Nama']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("IPK", f"{student_data['IPK']:.2f}")
                if 'Frekuensi_Penggunaan_AI' in student_data:
                    st.metric("Frekuensi Penggunaan AI", student_data['Frekuensi_Penggunaan_AI'])
            
            with col2:
                if 'Perform_Akademik' in student_data:
                    st.metric("Performa Akademik", student_data['Perform_Akademik'])
                if 'Tingkat_Ketergantungan_AI' in student_data:
                    st.metric("Tingkat Ketergantungan AI", student_data['Tingkat_Ketergantungan_AI'])
            
            # Tampilkan prediksi dan rekomendasi
            if 'Prediksi' in student_data and 'Rekomendasi' in student_data:
                st.markdown("---")
                
                # Mapping prediksi numerik ke label
                if st.session_state.target in st.session_state.label_encoders:
                    le = st.session_state.label_encoders[st.session_state.target]
                    pred_mapping = {i: label for i, label in enumerate(le.classes_)}
                    pred_label = pred_mapping.get(student_data['Prediksi'], student_data['Prediksi'])
                else:
                    pred_label = {0: 'Rendah', 1: 'Sedang', 2: 'Tinggi'}.get(student_data['Prediksi'], student_data['Prediksi'])
                
                # Tampilkan dengan warna sesuai tingkat
                if pred_label == 'Rendah':
                    st.success(f"**TINGKAT PENGGUNAAN AI: {pred_label.upper()}**")
                elif pred_label == 'Sedang':
                    st.warning(f"**TINGKAT PENGGUNAAN AI: {pred_label.upper()}**")
                elif pred_label == 'Tinggi':
                    st.error(f"**TINGKAT PENGGUNAAN AI: {pred_label.upper()}**")
                else:
                    st.info(f"**TINGKAT PENGGUNAAN AI: {pred_label}**")
                
                # Tampilkan rekomendasi
                st.markdown("### 📝 Rekomendasi")
                st.markdown(student_data['Rekomendasi'])
            
            # Visualisasi perbandingan
            st.markdown("---")
            st.subheader("Perbandingan dengan Rata-rata Kelas")
            
            # Hitung rata-rata
            avg_ipk = st.session_state.df_results['IPK'].mean()
            if 'Frekuensi_Penggunaan_AI' in st.session_state.df_results.columns:
                frekuensi_counts = st.session_state.df_results['Frekuensi_Penggunaan_AI'].value_counts()
                most_common_freq = frekuensi_counts.index[0] if not frekuensi_counts.empty else "Tidak ada data"
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Chart IPK
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.bar(['Anda', 'Rata-rata Kelas'], 
                             [student_data['IPK'], avg_ipk], 
                             color=['blue', 'lightblue'])
                ax.set_ylabel('IPK')
                ax.set_title('Perbandingan IPK')
                ax.set_ylim(0, 4.0)
                
                # Tambahkan nilai di atas bar
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                           f'{height:.2f}', ha='center', va='bottom')
                
                st.pyplot(fig)
            
            with col2:
                # Info tambahan
                st.info("**Statistik Kelas:**")
                st.write(f"- Jumlah mahasiswa: {len(st.session_state.df_results)}")
                st.write(f"- IPK tertinggi: {st.session_state.df_results['IPK'].max():.2f}")
                st.write(f"- IPK terendah: {st.session_state.df_results['IPK'].min():.2f}")
                
                if 'Frekuensi_Penggunaan_AI' in st.session_state.df_results.columns:
                    st.write(f"- Frekuensi AI paling umum: {most_common_freq}")
        
        else:
            st.warning(f"Data untuk mahasiswa '{student_name}' tidak ditemukan.")
            st.info("Data yang tersedia untuk mahasiswa berikut:")
            available_students = st.session_state.df_results['Nama'].head(10).tolist()
            for student in available_students:
                st.write(f"- {student}")
    else:
        st.warning("Belum ada hasil analisis yang tersedia.")
        st.info("Silakan minta guru/admin untuk melakukan analisis data terlebih dahulu.")

# Aplikasi utama
def main():
    # Inisialisasi session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user_type = None
    
    # Tampilkan halaman sesuai status login
    if not st.session_state.logged_in:
        login_page()
    else:
        if st.session_state.user_type == "guru":
            guru_dashboard()
        else:
            mahasiswa_dashboard()

if __name__ == "__main__":
    main()
