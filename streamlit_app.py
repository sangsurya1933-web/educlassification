import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Analisis Penggunaan AI Mahasiswa",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'le_jurusan' not in st.session_state:
    st.session_state.le_jurusan = None
if 'le_tools' not in st.session_state:
    st.session_state.le_tools = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

def load_data():
    """Load and preprocess the dataset"""
    # Using the provided dataset
    data = {
        'Nama': ['Althaf Rayyan Putra', 'Ayesha Kinanti', 'Salsabila Nurfadila', 'Anindya Safira', 
                'Iqbal Ramadhan', 'Muhammad Rizky Pratama', 'Fikri Alfarizi', 'Citra Maharani',
                'Zidan Harits', 'Rizky Kurniawan Putra', 'Raka Bimantara', 'Zahra Alya Safitri',
                'Muhammad Naufal Haidar', 'Ammar Zaky Firmansyah', 'Ilham Nurhadi', 'Nayla Syakira',
                'Arfan Maulana', 'Nabila Khairunnisa', 'Safira Azzahra Putri', 'Farah Amalia',
                'Muhammad Reza Ananda', 'Aulia Rahma', 'Yusuf Al Hakim', 'Damar Alif Prakoso',
                'Khansa Humaira Zahira', 'Syifa Nabila', 'Rafi Aditya', 'Aisyah Putri Ramadhani',
                'Bagas Arya Nugraha', 'Muhammad Fadhil Akbar Santoso', 'Muhammad Ilham Ramadhan',
                'Farrel Alvaro Nugroho', 'Nour Azzahra Salma Putri', 'Hanifa Azzahra'],
        
        'Studi_Jurusan': ['Teknologi Informasi', 'Teknologi Informasi', 'Teknik Informatika', 
                         'Teknik Informatika', 'Farmasi', 'Teknologi Informasi', 'Teknologi Informasi',
                         'Keperawatan', 'Farmasi', 'Teknik Informatika', 'Farmasi', 'Teknik Informatika',
                         'Farmasi', 'Farmasi', 'Teknologi Informasi', 'Teknologi Informasi',
                         'Teknik Informatika', 'Teknologi Informasi', 'Teknologi Informasi', 'Teknologi Informasi',
                         'Teknologi Informasi', 'Teknik Informatika', 'Teknik Informatika', 'Teknologi Informasi',
                         'Teknik Informatika', 'Teknologi Informasi', 'Teknik Informatika', 'Teknik Informatika',
                         'Teknologi Informasi', 'Mesin Otomotif', 'Mesin Otomotif', 'Keperawatan',
                         'Keperawatan', 'Teknologi Informasi'],
        
        'Semester': [7, 3, 1, 5, 1, 5, 1, 5, 7, 5, 3, 3, 1, 3, 1, 7, 7, 1, 1, 1,
                     1, 3, 5, 3, 1, 1, 5, 3, 5, 1, 3, 3, 3, 1],
        
        'AI_Tools': ['Gemini', 'Gemini', 'Gemini', 'Gemini', 'Gemini', 'Gemini', 'ChatGPT',
                    'Multiple', 'Gemini', 'ChatGPT', 'ChatGPT', 'Gemini', 'Gemini', 'ChatGPT',
                    'Gemini', 'ChatGPT', 'ChatGPT', 'Gemini', 'Gemini', 'ChatGPT',
                    'ChatGPT', 'Gemini', 'ChatGPT', 'ChatGPT', 'ChatGPT', 'Copilot', 'ChatGPT',
                    'Gemini', 'ChatGPT', 'ChatGPT', 'ChatGPT', 'ChatGPT', 'ChatGPT', 'ChatGPT'],
        
        'Trust_Level': [4, 4, 5, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 4, 1, 5, 5, 5,
                        4, 1, 3, 4, 4, 5, 4, 1, 3, 1, 3, 4, 3, 4],
        
        'Usage_Intensity_Score': [8, 9, 3, 6, 10, 4, 7, 2, 4, 3, 4, 10, 3, 7, 9, 8, 2, 9, 7, 6,
                                  3, 2, 8, 3, 7, 7, 9, 4, 5, 4, 6, 7, 7, 10]
    }
    
    df = pd.DataFrame(data)
    return df

def data_cleaning(df):
    """Perform data cleaning operations"""
    df_clean = df.copy()
    
    # Convert Usage_Intensity_Score to numeric (handle '10+' values)
    df_clean['Usage_Intensity_Score'] = df_clean['Usage_Intensity_Score'].apply(
        lambda x: 10 if str(x) == '10+' else float(x)
    )
    
    # Remove duplicates based on Nama and Studi_Jurusan
    df_clean = df_clean.drop_duplicates(subset=['Nama', 'Studi_Jurusan'], keep='first')
    
    # Fill missing values if any
    df_clean = df_clean.fillna({
        'Semester': df_clean['Semester'].median(),
        'Trust_Level': df_clean['Trust_Level'].median(),
        'Usage_Intensity_Score': df_clean['Usage_Intensity_Score'].median()
    })
    
    return df_clean

def encode_categorical(df):
    """Encode categorical variables"""
    df_encoded = df.copy()
    
    # Create label encoders
    le_jurusan = LabelEncoder()
    le_tools = LabelEncoder()
    
    # Encode categorical columns
    df_encoded['Studi_Jurusan_Encoded'] = le_jurusan.fit_transform(df_encoded['Studi_Jurusan'])
    df_encoded['AI_Tools_Encoded'] = le_tools.fit_transform(df_encoded['AI_Tools'])
    
    # Create target variable (categorize usage intensity)
    df_encoded['Usage_Level'] = pd.cut(
        df_encoded['Usage_Intensity_Score'],
        bins=[0, 4, 7, 11],
        labels=['Rendah', 'Sedang', 'Tinggi']
    )
    
    # Create feature matrix
    features = ['Semester', 'Trust_Level', 'Studi_Jurusan_Encoded', 'AI_Tools_Encoded']
    X = df_encoded[features]
    y = df_encoded['Usage_Level']
    
    return X, y, le_jurusan, le_tools, df_encoded

def train_random_forest(X, y):
    """Train Random Forest model"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, X_test, y_test, y_pred, accuracy, X_train, X_test

def generate_recommendations(predictions, df_students):
    """Generate recommendations based on predictions"""
    recommendations = {
        'Rendah': {
            'title': 'LEVEL AMAN',
            'message': 'Penggunaan AI dalam kategori aman. Mahasiswa dapat terus menggunakan AI sebagai alat bantu belajar dengan bijak.',
            'actions': [
                'Lanjutkan penggunaan AI secara bertanggung jawab',
                'Gunakan AI untuk meningkatkan pemahaman konsep',
                'Jaga keseimbangan antara penggunaan AI dan belajar mandiri'
            ],
            'color': 'green'
        },
        'Sedang': {
            'title': 'PERLU TEGURAN',
            'message': 'Penggunaan AI mulai mengkhawatirkan. Perlu pengawasan dan teguran untuk memastikan penggunaan yang tepat.',
            'actions': [
                'Berikan pemahaman tentang etika penggunaan AI',
                'Batasi penggunaan AI untuk tugas-tugas tertentu saja',
                'Lakukan monitoring berkala',
                'Berikan tugas yang mendorong pemikiran kritis'
            ],
            'color': 'orange'
        },
        'Tinggi': {
            'title': 'BUTUH PENGAWASAN LEBIH',
            'message': 'Tingkat penggunaan AI sangat tinggi dan berpotensi mengganggu proses belajar. Diperlukan pengawasan ketat.',
            'actions': [
                'Lakukan pembatasan akses ke tools AI',
                'Wajibkan konsultasi dengan dosen untuk tugas yang menggunakan AI',
                'Implementasikan sistem deteksi plagiarisme AI',
                'Berikan sanksi akademik jika melanggar',
                'Lakukan evaluasi berkala setiap minggu'
            ],
            'color': 'red'
        }
    }
    
    results = []
    for idx, student in df_students.iterrows():
        level = predictions[idx]
        rec = recommendations[level]
        
        results.append({
            'Nama': student['Nama'],
            'Studi_Jurusan': student['Studi_Jurusan'],
            'Semester': student['Semester'],
            'AI_Tools': student['AI_Tools'],
            'Trust_Level': student['Trust_Level'],
            'Usage_Intensity_Score': student['Usage_Intensity_Score'],
            'Predicted_Level': level,
            'Recommendation_Title': rec['title'],
            'Recommendation_Message': rec['message'],
            'Actions': rec['actions'],
            'Color': rec['color']
        })
    
    return pd.DataFrame(results)

def teacher_dashboard():
    """Teacher/Dashboard Interface"""
    st.title("🧑‍🏫 Dashboard Guru - Analisis Penggunaan AI")
    st.markdown("### Analisis Tingkat Penggunaan AI terhadap Performa Akademik Mahasiswa")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🧹 Data Cleaning", "🤖 Model Training", "📈 Evaluation"])
    
    with tab1:
        st.header("Data Overview")
        if st.session_state.df is None:
            st.session_state.df = load_data()
        
        df = st.session_state.df
        st.dataframe(df, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Mahasiswa", len(df))
        with col2:
            st.metric("Jumlah Jurusan", df['Studi_Jurusan'].nunique())
        with col3:
            avg_usage = df['Usage_Intensity_Score'].apply(
                lambda x: 10 if str(x) == '10+' else float(x)
            ).mean()
            st.metric("Rata-rata Penggunaan", f"{avg_usage:.1f}")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Usage Intensity Distribution
        usage_scores = df['Usage_Intensity_Score'].apply(
            lambda x: 10 if str(x) == '10+' else float(x)
        )
        axes[0, 0].hist(usage_scores, bins=10, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Distribusi Intensitas Penggunaan AI')
        axes[0, 0].set_xlabel('Skor Intensitas')
        axes[0, 0].set_ylabel('Frekuensi')
        
        # Trust Level Distribution
        trust_counts = df['Trust_Level'].value_counts().sort_index()
        axes[0, 1].bar(trust_counts.index, trust_counts.values, alpha=0.7)
        axes[0, 1].set_title('Distribusi Tingkat Kepercayaan AI')
        axes[0, 1].set_xlabel('Tingkat Kepercayaan')
        axes[0, 1].set_ylabel('Frekuensi')
        
        # AI Tools Usage
        tools_counts = df['AI_Tools'].value_counts()
        axes[1, 0].pie(tools_counts.values, labels=tools_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Distribusi Penggunaan Tools AI')
        
        # Study Program Distribution
        jurusan_counts = df['Studi_Jurusan'].value_counts()
        axes[1, 1].barh(list(jurusan_counts.index), jurusan_counts.values, alpha=0.7)
        axes[1, 1].set_title('Distribusi Mahasiswa per Jurusan')
        axes[1, 1].set_xlabel('Jumlah Mahasiswa')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab2:
        st.header("Data Cleaning")
        st.info("""
        **Proses Data Cleaning:**
        1. Konversi nilai '10+' menjadi 10
        2. Hapus duplikat berdasarkan Nama dan Jurusan
        3. Isi missing values dengan median
        """)
        
        if st.button("🚀 Jalankan Data Cleaning", type="primary"):
            with st.spinner("Melakukan data cleaning..."):
                df_clean = data_cleaning(st.session_state.df)
                st.session_state.df = df_clean
                st.success("✅ Data cleaning selesai!")
                
                st.subheader("Data Setelah Cleaning")
                st.dataframe(df_clean, use_container_width=True)
                
                # Show cleaning stats
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Data", len(df_clean))
                with col2:
                    st.metric("Data Duplikat Dihapus", len(st.session_state.df) - len(df_clean))
    
    with tab3:
        st.header("Encoding & Model Training")
        st.info("""
        **Proses Encoding & Training:**
        1. Encoding variabel kategorikal (Jurusan, AI Tools)
        2. Membagi data menjadi training dan testing set
        3. Melatih model Random Forest
        """)
        
        if st.button("🎯 Train Random Forest Model", type="primary"):
            if st.session_state.df is None:
                st.warning("Harap lakukan data cleaning terlebih dahulu!")
            else:
                with st.spinner("Melakukan encoding dan training model..."):
                    X, y, le_jurusan, le_tools, df_encoded = encode_categorical(st.session_state.df)
                    st.session_state.le_jurusan = le_jurusan
                    st.session_state.le_tools = le_tools
                    
                    model, X_test, y_test, y_pred, accuracy, X_train, X_test = train_random_forest(X, y)
                    st.session_state.model = model
                    
                    # Make predictions for all students
                    predictions = model.predict(X)
                    st.session_state.predictions = predictions
                    
                    # Generate recommendations
                    recommendations_df = generate_recommendations(predictions, st.session_state.df)
                    st.session_state.results = recommendations_df
                    
                    st.success(f"✅ Model berhasil dilatih dengan akurasi: {accuracy:.2%}")
                    
                    # Show encoded data
                    st.subheader("Data Setelah Encoding")
                    st.dataframe(df_encoded[['Nama', 'Studi_Jurusan', 'AI_Tools', 
                                            'Studi_Jurusan_Encoded', 'AI_Tools_Encoded', 
                                            'Usage_Level']].head(), use_container_width=True)
                    
                    # Show feature importance
                    feature_importance = pd.DataFrame({
                        'Feature': ['Semester', 'Trust_Level', 'Studi_Jurusan', 'AI_Tools'],
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    st.subheader("Feature Importance")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.barh(feature_importance['Feature'], feature_importance['Importance'])
                    ax.set_xlabel('Importance')
                    ax.set_title('Feature Importance dalam Model')
                    st.pyplot(fig)
    
    with tab4:
        st.header("Evaluation & Recommendations")
        
        if st.session_state.results is not None:
            # Show classification results
            st.subheader("Hasil Klasifikasi Mahasiswa")
            
            # Filter options
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_level = st.selectbox(
                    "Filter Berdasarkan Level",
                    ["Semua", "Rendah", "Sedang", "Tinggi"]
                )
            with col2:
                filter_jurusan = st.selectbox(
                    "Filter Berdasarkan Jurusan",
                    ["Semua"] + list(st.session_state.results['Studi_Jurusan'].unique())
                )
            with col3:
                filter_tools = st.selectbox(
                    "Filter Berdasarkan AI Tools",
                    ["Semua"] + list(st.session_state.results['AI_Tools'].unique())
                )
            
            # Apply filters
            filtered_results = st.session_state.results.copy()
            if filter_level != "Semua":
                filtered_results = filtered_results[filtered_results['Predicted_Level'] == filter_level]
            if filter_jurusan != "Semua":
                filtered_results = filtered_results[filtered_results['Studi_Jurusan'] == filter_jurusan]
            if filter_tools != "Semua":
                filtered_results = filtered_results[filtered_results['AI_Tools'] == filter_tools]
            
            # Display results
            st.dataframe(filtered_results[[
                'Nama', 'Studi_Jurusan', 'Semester', 'AI_Tools',
                'Usage_Intensity_Score', 'Predicted_Level', 'Recommendation_Title'
            ]], use_container_width=True)
            
            # Show detailed recommendations
            st.subheader("Detail Rekomendasi")
            
            for level in ['Tinggi', 'Sedang', 'Rendah']:
                level_data = filtered_results[filtered_results['Predicted_Level'] == level]
                if not level_data.empty:
                    with st.expander(f"📊 Mahasiswa dengan Level {level} ({len(level_data)} orang)"):
                        for _, student in level_data.iterrows():
                            color_map = {'red': '🔴', 'orange': '🟡', 'green': '🟢'}
                            st.markdown(f"""
                            **{color_map[student['Color']]} {student['Nama']}** ({student['Studi_Jurusan']} - Semester {student['Semester']})
                            - **AI Tools:** {student['AI_Tools']}
                            - **Trust Level:** {student['Trust_Level']}
                            - **Usage Score:** {student['Usage_Intensity_Score']}
                            - **Klasifikasi:** **{student['Predicted_Level']}**
                            - **Rekomendasi:** {student['Recommendation_Title']}
                            
                            **Tindakan yang disarankan:**
                            {chr(10).join(['• ' + action for action in student['Actions']])}
                            """)
            
            # Summary statistics
            st.subheader("📈 Statistik Klasifikasi")
            col1, col2, col3 = st.columns(3)
            
            level_counts = filtered_results['Predicted_Level'].value_counts()
            
            with col1:
                if 'Tinggi' in level_counts:
                    st.metric("Level TINGGI", level_counts['Tinggi'], 
                             delta="Butuh Pengawasan Ketat", delta_color="inverse")
            
            with col2:
                if 'Sedang' in level_counts:
                    st.metric("Level SEDANG", level_counts['Sedang'],
                             delta="Perlu Teguran")
            
            with col3:
                if 'Rendah' in level_counts:
                    st.metric("Level RENDAH", level_counts['Rendah'],
                             delta="Aman", delta_color="off")
            
            # Export option
            if st.button("📥 Export Hasil ke CSV"):
                csv = filtered_results.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="hasil_klasifikasi_ai_mahasiswa.csv",
                    mime="text/csv"
                )
        else:
            st.warning("Harap train model terlebih dahulu di tab 'Model Training'")

def student_login():
    """Student Login Interface"""
    st.title("👨‍🎨 Login Mahasiswa")
    
    if st.session_state.df is None:
        st.session_state.df = load_data()
    
    # Student selection
    student_names = sorted(st.session_state.df['Nama'].unique())
    selected_student = st.selectbox("Pilih Nama Anda", student_names)
    
    if st.button("🔍 Lihat Hasil Analisis", type="primary"):
        if st.session_state.results is not None:
            # Find student in results
            student_result = st.session_state.results[
                st.session_state.results['Nama'] == selected_student
            ]
            
            if not student_result.empty:
                student = student_result.iloc[0]
                
                st.success(f"✅ Data ditemukan untuk {selected_student}")
                
                # Display student information
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📋 Profil Mahasiswa")
                    st.markdown(f"""
                    **Nama:** {student['Nama']}
                    **Jurusan:** {student['Studi_Jurusan']}
                    **Semester:** {student['Semester']}
                    **AI Tools yang Digunakan:** {student['AI_Tools']}
                    **Tingkat Kepercayaan AI:** {student['Trust_Level']}/5
                    **Skor Penggunaan AI:** {student['Usage_Intensity_Score']}/10
                    """)
                
                with col2:
                    st.subheader("🎯 Hasil Klasifikasi")
                    
                    # Color-coded level display
                    level_colors = {
                        'Rendah': '🟢',
                        'Sedang': '🟡',
                        'Tinggi': '🔴'
                    }
                    
                    level_icon = level_colors.get(student['Predicted_Level'], '⚪')
                    st.markdown(f"""
                    **Tingkat Penggunaan AI:** {level_icon} **{student['Predicted_Level']}**
                    
                    **Status:** **{student['Recommendation_Title']}**
                    """)
                
                # Display detailed recommendation
                st.subheader("📝 Rekomendasi & Saran")
                
                # Create a colored box for the recommendation
                color_map = {'red': '#ffcccc', 'orange': '#fff3cd', 'green': '#d4edda'}
                border_color = {'red': '#f5c6cb', 'orange': '#ffeaa7', 'green': '#c3e6cb'}
                text_color = {'red': '#721c24', 'orange': '#856404', 'green': '#155724'}
                
                st.markdown(f"""
                <div style="background-color: {color_map[student['Color']]}; 
                            border-left: 5px solid {border_color[student['Color']]};
                            padding: 20px; 
                            border-radius: 5px;
                            color: {text_color[student['Color']]};
                            margin: 10px 0;">
                    <h4 style="margin-top: 0;">{student['Recommendation_Title']}</h4>
                    <p>{student['Recommendation_Message']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.subheader("✅ Tindakan yang Disarankan")
                for i, action in enumerate(student['Actions'], 1):
                    st.markdown(f"{i}. {action}")
                
                # Additional advice based on level
                st.subheader("💡 Tips untuk Anda")
                if student['Predicted_Level'] == 'Tinggi':
                    st.warning("""
                    **Perhatian:** Penggunaan AI yang berlebihan dapat menghambat perkembangan:
                    - Kemampuan berpikir kritis
                    - Kreativitas pribadi
                    - Pemahaman konsep fundamental
                    
                    **Saran:** Batasi penggunaan AI hanya untuk brainstorming dan verifikasi.
                    """)
                elif student['Predicted_Level'] == 'Sedang':
                    st.info("""
                    **Saran Penggunaan AI yang Sehat:**
                    - Gunakan AI sebagai asisten, bukan pengganti
                    - Selalu verifikasi informasi dari AI dengan sumber terpercaya
                    - Jangan gunakan AI untuk menyelesaikan seluruh tugas
                    """)
                else:
                    st.success("""
                    **Bagus!** Anda menggunakan AI dengan bijak:
                    - Teruskan penggunaan yang seimbang ini
                    - AI adalah alat bantu, bukan tujuan
                    - Pertahankan kemampuan analisis mandiri Anda
                    """)
                
                # Generate a personal development plan
                st.subheader("📅 Rencana Pengembangan Pribadi")
                
                development_plan = {
                    'Rendah': [
                        "Pertahankan pola penggunaan AI yang sehat",
                        "Eksplorasi fitur AI yang dapat meningkatkan produktivitas",
                        "Bagikan tips penggunaan AI yang bijak dengan teman"
                    ],
                    'Sedang': [
                        "Buat jadwal penggunaan AI (maksimal 2 jam/hari)",
                        "Ikuti workshop tentang etika penggunaan AI",
                        "Diskusikan dengan dosen tentang penggunaan AI yang tepat"
                    ],
                    'Tinggi': [
                        "Konsultasi dengan pembimbing akademik",
                        "Ikut program mentoring penggunaan teknologi",
                        "Lakukan 'digital detox' dari AI selama akhir pekan"
                    ]
                }
                
                for plan in development_plan.get(student['Predicted_Level'], []):
                    st.markdown(f"✓ {plan}")
            
            else:
                st.error("Data tidak ditemukan. Silakan hubungi administrator.")
        else:
            st.warning("⚠️ Sistem belum siap. Harap tunggu guru menyelesaikan analisis.")

def main():
    """Main application"""
    # Sidebar for login selection
    st.sidebar.title("🔐 Sistem Login")
    st.sidebar.markdown("---")
    
    user_type = st.sidebar.radio(
        "Pilih Jenis Pengguna",
        ["👨‍🎓 Mahasiswa", "🧑‍🏫 Guru/Dosen"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Sistem Analisis Penggunaan AI**
    
    **Fitur Guru:**
    - Data cleaning & preprocessing
    - Encoding data kategorikal
    - Model training dengan Random Forest
    - Evaluasi hasil & rekomendasi
    
    **Fitur Mahasiswa:**
    - Lihat hasil klasifikasi pribadi
    - Dapatkan rekomendasi spesifik
    - Tips penggunaan AI yang sehat
    """)
    
    # Show appropriate interface based on user type
    if user_type == "🧑‍🏫 Guru/Dosen":
        # Password protection for teacher
        password = st.sidebar.text_input("Password Guru", type="password")
        if password == "guru123":  # Simple password for demo
            teacher_dashboard()
        elif password and password != "guru123":
            st.sidebar.error("Password salah!")
        else:
            st.sidebar.warning("Masukkan password untuk akses dashboard guru")
    else:
        student_login()

if __name__ == "__main__":
    main()
