import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Analisis AI Mahasiswa UMMgl",
    page_icon="🤖",
    layout="wide"
)

# Session state untuk login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'preprocessing_done' not in st.session_state:
    st.session_state.preprocessing_done = False

def load_sample_data():
    """Load dataset sampel yang disederhanakan"""
    # Data sampel (kita buat dataset kecil untuk contoh)
    data = {
        'Nama': [
            'Althaf Rayyan Putra', 'Ayesha Kinanti', 'Salsabila Nurfadila',
            'Anindya Safira', 'Iqbal Ramadhan', 'Muhammad Rizky Pratama',
            'Fikri Alfarizi', 'Citra Maharani', 'Zidan Harits'
        ],
        'Studi_Jurusan': [
            'Teknologi Informasi', 'Teknologi Informasi', 'Teknik Informatika',
            'Teknik Informatika', 'Farmasi', 'Teknologi Informasi',
            'Teknologi Informasi', 'Keperawatan', 'Farmasi'
        ],
        'Semester': [7, 3, 1, 5, 1, 5, 1, 5, 7],
        'AI_Tools': ['Gemini', 'Gemini', 'Gemini', 'Gemini', 'Gemini', 
                    'Gemini', 'ChatGPT', 'Multiple', 'Gemini'],
        'Trust_Level': [4, 4, 5, 4, 5, 4, 4, 4, 4],
        'Usage_Intensity_Score': [8, 9, 3, 6, 10, 4, 7, 2, 4]
    }
    
    df = pd.DataFrame(data)
    # Handle '10+' values
    df['Usage_Intensity_Score'] = df['Usage_Intensity_Score'].apply(
        lambda x: 10 if str(x) == '10+' else int(x)
    )
    
    # Create target variable (kategori intensitas penggunaan)
    def categorize_intensity(score):
        if score <= 3:
            return 'Rendah'
        elif score <= 7:
            return 'Sedang'
        else:
            return 'Tinggi'
    
    df['Kategori_Intensitas'] = df['Usage_Intensity_Score'].apply(categorize_intensity)
    return df

def load_full_data():
    """Load dataset lengkap dari konten yang diberikan"""
    # Data dari konten yang diberikan
    data = {
        'Nama': [],
        'Studi_Jurusan': [],
        'Semester': [],
        'AI_Tools': [],
        'Trust_Level': [],
        'Usage_Intensity_Score': []
    }
    
    # Parse data dari konten (contoh sebagian data)
    # Dalam implementasi nyata, Anda akan load dari Excel
    import io
    import csv
    
    # Contoh data parsing dari string
    csv_data = """Nama,Studi_Jurusan,Semester,AI_Tools,Trust_Level,Usage_Intensity_Score
Althaf Rayyan Putra,Teknologi Informasi,7,Gemini,4,8
Ayesha Kinanti,Teknologi Informasi,3,Gemini,4,9
Salsabila Nurfadila,Teknik Informatika,1,Gemini,5,3
Anindya Safira,Teknik Informatika,5,Gemini,4,6
Iqbal Ramadhan,Farmasi,1,Gemini,5,10
Muhammad Rizky Pratama,Teknologi Informasi,5,Gemini,4,4
Fikri Alfarizi,Teknologi Informasi,1,ChatGPT,4,7
Citra Maharani,Keperawatan,5,Multiple,4,2
Zidan Harits,Farmasi,7,Gemini,4,4"""
    
    df = pd.read_csv(io.StringIO(csv_data))
    
    # Create target variable
    def categorize_intensity(score):
        if score <= 3:
            return 'Rendah'
        elif score <= 7:
            return 'Sedang'
        else:
            return 'Tinggi'
    
    df['Kategori_Intensitas'] = df['Usage_Intensity_Score'].apply(categorize_intensity)
    return df

def preprocessing_data(df):
    """Melakukan preprocessing data"""
    # Copy dataframe
    df_processed = df.copy()
    
    # 1. Handle missing values
    df_processed = df_processed.dropna()
    
    # 2. Encoding variabel kategorikal
    label_encoders = {}
    
    # Encoding untuk fitur kategorikal
    categorical_cols = ['Studi_Jurusan', 'AI_Tools']
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col + '_encoded'] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
    
    # Encoding target
    le_target = LabelEncoder()
    df_processed['Target_encoded'] = le_target.fit_transform(df_processed['Kategori_Intensitas'])
    label_encoders['Target'] = le_target
    
    # 3. Split features and target
    features = ['Semester', 'Trust_Level', 'Studi_Jurusan_encoded', 'AI_Tools_encoded']
    X = df_processed[features]
    y = df_processed['Target_encoded']
    
    # 4. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 5. Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'df_processed': df_processed,
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'label_encoders': label_encoders,
        'scaler': scaler,
        'features': features
    }

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10):
    """Melatih model Random Forest"""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, label_encoders):
    """Evaluasi model dan tampilkan metrics"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Decode labels
    y_test_decoded = label_encoders['Target'].inverse_transform(y_test)
    y_pred_decoded = label_encoders['Target'].inverse_transform(y_pred)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'y_test_decoded': y_test_decoded,
        'y_pred_decoded': y_pred_decoded,
        'y_pred_proba': y_pred_proba
    }

def plot_confusion_matrix(cm, classes, ax):
    """Plot confusion matrix menggunakan matplotlib"""
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    # Tampilkan tick marks
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    
    # Tampilkan nilai dalam setiap cell
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")

def login_page():
    """Halaman login"""
    st.title("🔐 Login Sistem Analisis AI Mahasiswa")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Login Guru")
        if st.button("Login sebagai Guru", use_container_width=True):
            st.session_state.logged_in = True
            st.session_state.user_role = 'guru'
            st.rerun()
    
    with col2:
        st.subheader("Login Siswa")
        if st.button("Login sebagai Siswa", use_container_width=True):
            st.session_state.logged_in = True
            st.session_state.user_role = 'siswa'
            st.rerun()

def guru_dashboard():
    """Dashboard untuk guru"""
    st.sidebar.title(f"👨‍🏫 Dashboard Guru")
    
    menu = st.sidebar.radio(
        "Menu",
        ["📊 Data Overview", "⚙️ Preprocessing", "🤖 Analisis Model", "📈 Evaluasi", "📤 Export Hasil"]
    )
    
    st.title("Analisis Tingkat Penggunaan AI terhadap Performa Akademik")
    st.markdown("**Algoritma: Random Forest**")
    
    if st.session_state.df is None:
        st.session_state.df = load_sample_data()
    
    df = st.session_state.df
    
    if menu == "📊 Data Overview":
        st.header("📊 Overview Dataset")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.subheader("Statistik Deskriptif")
            st.dataframe(df.describe(), use_container_width=True)
        
        with col2:
            st.subheader("Informasi Dataset")
            buffer = BytesIO()
            df.info(buf=buffer)
            info_str = buffer.getvalue().decode('utf-8')
            st.text(info_str)
            
            st.subheader("Distribusi Kategori")
            category_counts = df['Kategori_Intensitas'].value_counts()
            st.dataframe(category_counts, use_container_width=True)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            category_counts.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax.set_title('Distribusi Kategori Intensitas Penggunaan AI')
            ax.set_xlabel('Kategori')
            ax.set_ylabel('Jumlah')
            st.pyplot(fig)
    
    elif menu == "⚙️ Preprocessing":
        st.header("⚙️ Preprocessing Data")
        
        if st.button("🚀 Jalankan Preprocessing", type="primary"):
            with st.spinner("Melakukan preprocessing data..."):
                preprocessing_results = preprocessing_data(df)
                st.session_state.preprocessing_results = preprocessing_results
                st.session_state.preprocessing_done = True
                
                st.success("✅ Preprocessing selesai!")
                
                # Tampilkan hasil preprocessing
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Data Setelah Preprocessing")
                    st.dataframe(preprocessing_results['df_processed'].head(), use_container_width=True)
                    
                    st.subheader("Distribusi Fitur yang Digunakan")
                    st.write(f"Jumlah fitur: {len(preprocessing_results['features'])}")
                    st.write(f"Fitur: {', '.join(preprocessing_results['features'])}")
                
                with col2:
                    st.subheader("Split Data")
                    st.write(f"Training samples: {len(preprocessing_results['X_train'])}")
                    st.write(f"Testing samples: {len(preprocessing_results['X_test'])}")
                    
                    # Visualisasi distribusi target
                    fig, ax = plt.subplots(figsize=(8, 4))
                    pd.Series(preprocessing_results['y_train']).value_counts().sort_index().plot(
                        kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4', '#45B7D1']
                    )
                    ax.set_title('Distribusi Target (Training Set)')
                    ax.set_xlabel('Kategori (encoded)')
                    ax.set_ylabel('Jumlah')
                    st.pyplot(fig)
        else:
            st.info("Klik tombol di atas untuk memulai preprocessing data.")
    
    elif menu == "🤖 Analisis Model":
        st.header("🤖 Analisis dengan Random Forest")
        
        if not st.session_state.preprocessing_done:
            st.warning("⚠️ Silakan jalankan preprocessing terlebih dahulu!")
            return
        
        preprocessing_results = st.session_state.preprocessing_results
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Parameter Model")
            n_estimators = st.slider("Jumlah Trees", 10, 200, 100, 10)
            max_depth = st.slider("Max Depth", 3, 20, 10, 1)
            
            if st.button("🎯 Latih Model", type="primary"):
                with st.spinner("Melatih model Random Forest..."):
                    # Update model parameters
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=42,
                        n_jobs=-1
                    )
                    model.fit(preprocessing_results['X_train'], preprocessing_results['y_train'])
                    st.session_state.model = model
                    
                    st.success("✅ Model berhasil dilatih!")
                    
                    # Tampilkan feature importance
                    feature_importance = model.feature_importances_
                    features = preprocessing_results['features']
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    indices = np.argsort(feature_importance)[::-1]
                    ax.barh(range(len(features)), feature_importance[indices], color='#4ECDC4')
                    ax.set_yticks(range(len(features)))
                    ax.set_yticklabels([features[i] for i in indices])
                    ax.set_xlabel('Feature Importance')
                    ax.set_title('Feature Importance Random Forest')
                    st.pyplot(fig)
        
        with col2:
            if st.session_state.model is not None:
                st.subheader("Informasi Model")
                model = st.session_state.model
                
                st.write(f"Jumlah Trees: {model.n_estimators}")
                st.write(f"Max Depth: {model.max_depth}")
                st.write(f"Classes: {model.classes_}")
                
                # Prediksi sampel
                st.subheader("Prediksi Sampel")
                sample_idx = st.number_input("Indeks sampel test", 0, len(preprocessing_results['X_test'])-1, 0)
                
                if st.button("🔮 Prediksi Sampel"):
                    sample = preprocessing_results['X_test'][sample_idx].reshape(1, -1)
                    prediction = model.predict(sample)
                    probability = model.predict_proba(sample)[0]
                    
                    label_decoder = preprocessing_results['label_encoders']['Target']
                    predicted_label = label_decoder.inverse_transform(prediction)[0]
                    true_label = label_decoder.inverse_transform(
                        [preprocessing_results['y_test'].iloc[sample_idx] if hasattr(preprocessing_results['y_test'], 'iloc') 
                         else preprocessing_results['y_test'][sample_idx]]
                    )[0]
                    
                    st.write(f"**True Label:** {true_label}")
                    st.write(f"**Predicted Label:** {predicted_label}")
                    
                    # Tampilkan probabilities
                    prob_df = pd.DataFrame({
                        'Kategori': label_decoder.classes_,
                        'Probability': probability
                    })
                    st.dataframe(prob_df, use_container_width=True)
    
    elif menu == "📈 Evaluasi":
        st.header("📈 Evaluasi Model")
        
        if st.session_state.model is None:
            st.warning("⚠️ Silakan latih model terlebih dahulu!")
            return
        
        model = st.session_state.model
        preprocessing_results = st.session_state.preprocessing_results
        
        # Evaluasi model
        evaluation_results = evaluate_model(
            model, 
            preprocessing_results['X_test'], 
            preprocessing_results['y_test'],
            preprocessing_results['label_encoders']
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Confusion Matrix")
            cm = evaluation_results['confusion_matrix']
            classes = preprocessing_results['label_encoders']['Target'].classes_
            
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_confusion_matrix(cm, classes, ax)
            st.pyplot(fig)
        
        with col2:
            st.subheader("📈 Metrics")
            st.metric("Accuracy", f"{evaluation_results['accuracy']:.3f}")
            
            report_df = pd.DataFrame(evaluation_results['classification_report']).transpose()
            st.dataframe(report_df, use_container_width=True)
        
        # Detail evaluasi
        st.subheader("🔍 Detail Evaluasi per Kelas")
        
        classes = preprocessing_results['label_encoders']['Target'].classes_
        for i, class_name in enumerate(classes):
            st.write(f"**Kelas: {class_name}**")
            precision = evaluation_results['classification_report'][str(i)]['precision']
            recall = evaluation_results['classification_report'][str(i)]['recall']
            f1 = evaluation_results['classification_report'][str(i)]['f1-score']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Precision", f"{precision:.3f}")
            with col2:
                st.metric("Recall", f"{recall:.3f}")
            with col3:
                st.metric("F1-Score", f"{f1:.3f}")
    
    elif menu == "📤 Export Hasil":
        st.header("📤 Export Hasil Analisis")
        
        if st.session_state.model is None:
            st.warning("⚠️ Tidak ada model yang tersedia untuk diexport!")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Export Model"):
                model_bytes = pickle.dumps(st.session_state.model)
                st.download_button(
                    label="Download Model",
                    data=model_bytes,
                    file_name="random_forest_model.pkl",
                    mime="application/octet-stream"
                )
        
        with col2:
            if st.button("📊 Export Data Preprocessing"):
                preprocessing_results = st.session_state.preprocessing_results
                df_bytes = preprocessing_results['df_processed'].to_csv(index=False).encode()
                st.download_button(
                    label="Download Data Preprocessing",
                    data=df_bytes,
                    file_name="data_preprocessed.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("📈 Export Evaluation Report"):
                if hasattr(st.session_state, 'preprocessing_results'):
                    preprocessing_results = st.session_state.preprocessing_results
                    model = st.session_state.model
                    
                    evaluation_results = evaluate_model(
                        model, 
                        preprocessing_results['X_test'], 
                        preprocessing_results['y_test'],
                        preprocessing_results['label_encoders']
                    )
                    
                    # Buat report
                    report_text = f"""
                    Random Forest Model Evaluation Report
                    ====================================
                    
                    Model Parameters:
                    - Algorithm: Random Forest
                    - Number of Trees: {model.n_estimators}
                    - Max Depth: {model.max_depth}
                    
                    Performance Metrics:
                    - Accuracy: {evaluation_results['accuracy']:.4f}
                    
                    Classification Report:
                    {pd.DataFrame(evaluation_results['classification_report']).to_string()}
                    
                    Confusion Matrix:
                    {evaluation_results['confusion_matrix']}
                    """
                    
                    st.download_button(
                        label="Download Evaluation Report",
                        data=report_text,
                        file_name="evaluation_report.txt",
                        mime="text/plain"
                    )

def siswa_dashboard():
    """Dashboard untuk siswa"""
    st.sidebar.title(f"👨‍🎓 Dashboard Siswa")
    
    menu = st.sidebar.radio(
        "Menu",
        ["🔍 Analisis Saya", "🤖 Prediksi Intensitas"]
    )
    
    st.title("Analisis Penggunaan AI untuk Mahasiswa")
    
    if menu == "🔍 Analisis Saya":
        st.header("🔍 Analisis Data Penggunaan AI")
        
        if st.session_state.df is None:
            st.session_state.df = load_sample_data()
        
        df = st.session_state.df
        
        # Filter untuk siswa tertentu (simulasi)
        nama_siswa = st.selectbox("Pilih Nama", df['Nama'].unique())
        
        if nama_siswa:
            siswa_data = df[df['Nama'] == nama_siswa]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Data Pribadi")
                st.write(f"**Nama:** {nama_siswa}")
                st.write(f"**Jurusan:** {siswa_data['Studi_Jurusan'].iloc[0]}")
                st.write(f"**Semester:** {siswa_data['Semester'].iloc[0]}")
                
                # Rata-rata penggunaan
                avg_score = siswa_data['Usage_Intensity_Score'].mean()
                avg_trust = siswa_data['Trust_Level'].mean()
                
                st.write(f"**Rata-rata Intensitas:** {avg_score:.1f}")
                st.write(f"**Rata-rata Trust Level:** {avg_trust:.1f}")
            
            with col2:
                st.subheader("📊 Statistik Penggunaan")
                
                # Tool preference
                tool_counts = siswa_data['AI_Tools'].value_counts()
                st.write("**Preferensi AI Tools:**")
                for tool, count in tool_counts.items():
                    st.write(f"- {tool}: {count} kali")
                
                # Kategori intensitas
                kategori_counts = siswa_data['Kategori_Intensitas'].value_counts()
                fig, ax = plt.subplots(figsize=(6, 4))
                kategori_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=['#FF6B6B', '#4ECDC4', '#45B7D1'])
                ax.set_title('Distribusi Kategori Intensitas')
                ax.set_ylabel('')
                st.pyplot(fig)
            
            # History penggunaan
            st.subheader("📜 Riwayat Penggunaan")
            st.dataframe(siswa_data[['Semester', 'AI_Tools', 'Trust_Level', 'Usage_Intensity_Score', 'Kategori_Intensitas']], 
                        use_container_width=True)
    
    elif menu == "🤖 Prediksi Intensitas":
        st.header("🤖 Prediksi Kategori Intensitas Penggunaan")
        
        if st.session_state.model is None:
            st.warning("⚠️ Model belum tersedia. Silakan hubungi administrator.")
            return
        
        # Form input untuk prediksi
        st.subheader("Masukkan Data Anda:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            jurusan = st.selectbox("Jurusan", [
                "Teknologi Informasi", "Teknik Informatika", "Farmasi", 
                "Keperawatan", "Mesin Otomotif"
            ])
            semester = st.slider("Semester", 1, 8, 5)
        
        with col2:
            ai_tool = st.selectbox("AI Tool yang Digunakan", [
                "ChatGPT", "Gemini", "Copilot", "Multiple"
            ])
            trust_level = st.slider("Trust Level", 1, 5, 4)
        
        if st.button("🔮 Prediksi Kategori Intensitas", type="primary"):
            # Load preprocessing results
            if hasattr(st.session_state, 'preprocessing_results'):
                preprocessing_results = st.session_state.preprocessing_results
                model = st.session_state.model
                
                # Encode input
                try:
                    # Encode jurusan
                    jurusan_encoded = preprocessing_results['label_encoders']['Studi_Jurusan'].transform([jurusan])[0]
                    # Encode AI tool
                    ai_tool_encoded = preprocessing_results['label_encoders']['AI_Tools'].transform([ai_tool])[0]
                    
                    # Prepare input array
                    input_array = np.array([[semester, trust_level, jurusan_encoded, ai_tool_encoded]])
                    
                    # Scale input
                    input_scaled = preprocessing_results['scaler'].transform(input_array)
                    
                    # Predict
                    prediction = model.predict(input_scaled)
                    probability = model.predict_proba(input_scaled)[0]
                    
                    # Decode prediction
                    label_decoder = preprocessing_results['label_encoders']['Target']
                    predicted_label = label_decoder.inverse_transform(prediction)[0]
                    
                    # Display results
                    st.success(f"**Hasil Prediksi: {predicted_label}**")
                    
                    # Show probabilities
                    st.subheader("Probabilitas per Kategori:")
                    
                    prob_df = pd.DataFrame({
                        'Kategori': label_decoder.classes_,
                        'Probabilitas': [f"{p:.1%}" for p in probability]
                    })
                    
                    # Visualisasi probabilitas
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.barh(prob_df['Kategori'], [float(p.strip('%'))/100 for p in prob_df['Probabilitas']], 
                           color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
                    ax.set_xlabel('Probabilitas')
                    ax.set_title('Distribusi Probabilitas Prediksi')
                    st.pyplot(fig)
                    
                    # Rekomendasi
                    st.subheader("💡 Rekomendasi:")
                    if predicted_label == 'Rendah':
                        st.info("""
                        **Saran Pengembangan:**
                        1. Coba eksplorasi lebih banyak fitur AI tools
                        2. Ikuti workshop pengenalan AI
                        3. Gunakan AI untuk tugas-tugas kecil terlebih dahulu
                        """)
                    elif predicted_label == 'Sedang':
                        st.info("""
                        **Saran Pengembangan:**
                        1. Tingkatkan penggunaan untuk proyek yang lebih kompleks
                        2. Eksplorasi integrasi AI dengan tools lain
                        3. Bagikan pengalaman Anda dengan teman
                        """)
                    else:
                        st.info("""
                        **Saran Pengembangan:**
                        1. Pertimbangkan untuk menjadi mentor AI
                        2. Eksplorasi AI tools yang lebih spesifik
                        3. Dokumentasikan best practices penggunaan AI
                        """)
                        
                except Exception as e:
                    st.error(f"Error dalam prediksi: {str(e)}")

def main():
    """Main function"""
    
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
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 10px;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if not st.session_state.logged_in:
        login_page()
    else:
        # Logout button
        if st.sidebar.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.user_role = None
            st.session_state.df = None
            st.session_state.model = None
            st.session_state.preprocessing_done = False
            st.rerun()
        
        if st.session_state.user_role == 'guru':
            guru_dashboard()
        else:
            siswa_dashboard()

if __name__ == "__main__":
    main()
