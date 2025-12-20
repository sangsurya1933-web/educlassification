# app.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import os
import base64
from io import BytesIO

# ==================== KONFIGURASI KELAS ====================
USER_CLASSES = {
    0: 'Light User',
    1: 'Moderate User', 
    2: 'Heavy User'
}

# ==================== PREPROCESSING DATA ====================
def classify_ai_usage(row):
    """Klasifikasi penggunaan AI berdasarkan Daily_Usage"""
    usage = row['Daily_Usage']
    
    if usage <= 10:
        return 0  # Light User
    elif 11 <= usage <= 30:
        return 1  # Moderate User
    else:
        return 2  # Heavy User

def create_sample_dataset():
    """Membuat dataset contoh"""
    np.random.seed(42)
    n_samples = 300
    
    data = {
        'Student_ID': [f'STD{i:03d}' for i in range(1, n_samples+1)],
        'College': np.random.choice(['Engineering', 'Commerce', 'Science', 'Arts', 'Medical'], n_samples),
        'Stream': np.random.choice(['Tech', 'Business', 'Science', 'Humanities', 'Health'], n_samples),
        'Year': np.random.randint(1, 5, n_samples),
        'AI_Tools': np.random.choice(['ChatGPT', 'Gemini', 'Copilot', 'Multiple'], n_samples),
        'Daily_Usage': np.random.randint(1, 50, n_samples),
        'Use_Cases': np.random.choice(['Assignments', 'Exam Prep', 'Research', 'Projects'], n_samples),
        'Trust_Level': np.random.randint(1, 6, n_samples),
        'Impact_Score': np.random.randint(-3, 4, n_samples)
    }
    
    df = pd.DataFrame(data)
    df['AI_User_Class'] = df.apply(classify_ai_usage, axis=1)
    
    # Tambah fitur engineering
    df['Multiple_Tools'] = df['AI_Tools'].apply(lambda x: 1 if x == 'Multiple' else 0)
    df['Usage_Per_Year'] = df['Daily_Usage'] / df['Year']
    
    return df

def preprocess_data(df):
    """Fungsi untuk preprocessing data lengkap"""
    df_processed = df.copy()
    
    # Drop kolom yang tidak diperlukan untuk modeling
    cols_to_drop = ['Student_ID']
    for col in cols_to_drop:
        if col in df_processed.columns:
            df_processed = df_processed.drop(columns=[col])
    
    # Handle missing values
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            df_processed[col].fillna(df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'Unknown', inplace=True)
        else:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
    
    # Encoding variabel kategorikal
    label_encoders = {}
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
    
    # Pisahkan fitur dan target
    target_col = 'AI_User_Class'
    
    if target_col in df_processed.columns:
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]
    else:
        st.error("Kolom target 'AI_User_Class' tidak ditemukan!")
        return None, None, None
    
    return X, y, label_encoders

# ==================== ANALISIS RANDOM FOREST ====================
def perform_random_forest_analysis(X, y, test_size=0.3, random_state=42, n_estimators=100, max_depth=10):
    """Melakukan analisis dengan Random Forest"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Inisialisasi dan train model
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    rf_model.fit(X_train_scaled, y_train)
    
    # Prediksi
    y_pred = rf_model.predict(X_test_scaled)
    y_pred_train = rf_model.predict(X_train_scaled)
    
    # Evaluasi
    accuracy_test = accuracy_score(y_test, y_pred)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    
    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Hitung metrics per class
    precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    return {
        'model': rf_model,
        'scaler': scaler,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'accuracy_test': accuracy_test,
        'accuracy_train': accuracy_train,
        'classification_report': class_report,
        'confusion_matrix': cm,
        'feature_importance': feature_importance,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class
    }

# ==================== VISUALISASI ====================
def plot_user_class_distribution(df):
    """Plot distribusi kelas pengguna AI"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Distribusi kelas
    class_counts = df['AI_User_Class'].value_counts().sort_index()
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    
    axes[0].bar([USER_CLASSES[i] for i in class_counts.index], class_counts.values, color=colors)
    axes[0].set_title('Distribusi Klasifikasi Pengguna AI', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Kategori Pengguna')
    axes[0].set_ylabel('Jumlah Siswa')
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Pie chart
    axes[1].pie(class_counts.values, labels=[USER_CLASSES[i] for i in class_counts.index], 
                colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Persentase Pengguna AI', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_analysis_results(analysis_results):
    """Plot hasil analisis lengkap"""
    results = analysis_results
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Confusion Matrix
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[USER_CLASSES[i] for i in range(cm.shape[1])],
                yticklabels=[USER_CLASSES[i] for i in range(cm.shape[0])],
                ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # 2. Feature Importance (Top 10)
    top_features = results['feature_importance'].head(10)
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_features)))
    axes[0, 1].barh(range(len(top_features)), top_features['importance'], color=colors)
    axes[0, 1].set_yticks(range(len(top_features)))
    axes[0, 1].set_yticklabels(top_features['feature'])
    axes[0, 1].set_title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Importance Score')
    axes[0, 1].invert_yaxis()
    
    # 3. Accuracy Comparison
    accuracy_data = {
        'Training': results['accuracy_train'],
        'Testing': results['accuracy_test']
    }
    colors_acc = ['#4CAF50', '#2196F3']
    bars = axes[0, 2].bar(accuracy_data.keys(), accuracy_data.values(), color=colors_acc)
    axes[0, 2].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].grid(axis='y', alpha=0.3)
    
    for bar, v in zip(bars, accuracy_data.values()):
        height = bar.get_height()
        axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Precision, Recall, F1-Score per Class
    x = np.arange(len(USER_CLASSES))
    width = 0.25
    
    axes[1, 0].bar(x - width, results['precision_per_class'], width, label='Precision', color='#FF9800')
    axes[1, 0].bar(x, results['recall_per_class'], width, label='Recall', color='#E91E63')
    axes[1, 0].bar(x + width, results['f1_per_class'], width, label='F1-Score', color='#9C27B0')
    
    axes[1, 0].set_title('Metrics per Class', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('User Class')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([USER_CLASSES[i] for i in range(len(USER_CLASSES))])
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 5. Classification Report Heatmap
    class_report = results['classification_report']
    if 'accuracy' in class_report:
        del class_report['accuracy']
    
    metrics_df = pd.DataFrame(class_report).transpose()
    if 'support' in metrics_df.columns:
        metrics_to_plot = metrics_df[['precision', 'recall', 'f1-score']].iloc[:-1]
        sns.heatmap(metrics_to_plot, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1, 1])
        axes[1, 1].set_title('Classification Metrics Heatmap', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Metric')
        axes[1, 1].set_ylabel('Class')
    
    # 6. Error Analysis
    error_rate = 1 - results['accuracy_test']
    success_rate = results['accuracy_test']
    
    axes[1, 2].pie([success_rate, error_rate], 
                   labels=['Success', 'Error'], 
                   colors=['#4CAF50', '#F44336'],
                   autopct='%1.1f%%', startangle=90)
    axes[1, 2].set_title('Success vs Error Rate', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

# ==================== EKSPOR DATA ====================
def export_to_csv(df, results, export_folder='exported_results'):
    """Ekspor hasil analisis ke CSV"""
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)
    
    # 1. Ekspor dataset dengan klasifikasi
    df_export = df.copy()
    df_export['AI_User_Class_Label'] = df_export['AI_User_Class'].map(USER_CLASSES)
    df_export.to_csv(f'{export_folder}/classified_data.csv', index=False)
    
    # 2. Ekspor feature importance
    results['feature_importance'].to_csv(f'{export_folder}/feature_importance.csv', index=False)
    
    # 3. Ekspor classification report
    class_report_df = pd.DataFrame(results['classification_report']).transpose()
    class_report_df.to_csv(f'{export_folder}/classification_report.csv')
    
    # 4. Ekspor confusion matrix
    cm_df = pd.DataFrame(results['confusion_matrix'], 
                        columns=[USER_CLASSES[i] for i in range(results['confusion_matrix'].shape[1])],
                        index=[USER_CLASSES[i] for i in range(results['confusion_matrix'].shape[0])])
    cm_df.to_csv(f'{export_folder}/confusion_matrix.csv')
    
    return export_folder

# ==================== PREDIKSI INDIVIDU ====================
def create_prediction_form():
    """Membuat form untuk prediksi individu"""
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            daily_usage = st.slider("Daily Usage (jam/minggu)", 1, 50, 15)
            year = st.selectbox("Tahun Studi", [1, 2, 3, 4])
            trust_level = st.slider("Trust Level (1-5)", 1, 5, 3)
        
        with col2:
            ai_tools = st.selectbox("AI Tools", ["ChatGPT", "Gemini", "Copilot", "Multiple"])
            use_cases = st.selectbox("Use Cases", ["Assignments", "Exam Prep", "Research", "Projects"])
            college = st.selectbox("College", ["Engineering", "Commerce", "Science", "Arts", "Medical"])
        
        submitted = st.form_submit_button("🔮 Prediksi Klasifikasi")
        
        if submitted:
            return {
                'Daily_Usage': daily_usage,
                'Year': year,
                'Trust_Level': trust_level,
                'AI_Tools': ai_tools,
                'Use_Cases': use_cases,
                'College': college,
                'Multiple_Tools': 1 if ai_tools == "Multiple" else 0,
                'Usage_Per_Year': daily_usage / year
            }
    return None

def predict_user_class(input_data, model, scaler, X_columns):
    """Prediksi kelas pengguna"""
    try:
        # Buat DataFrame dari input
        input_df = pd.DataFrame([input_data])
        
        # Encoding untuk variabel kategorikal
        for col in ['AI_Tools', 'Use_Cases', 'College']:
            if col in input_df.columns:
                # Encoding sederhana (dalam aplikasi nyata gunakan encoder yang sama)
                input_df[col] = pd.factorize(input_df[col])[0]
        
        # Pastikan semua kolom ada
        for col in X_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[X_columns]
        
        # Scale dan prediksi
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        
        return prediction, probabilities
        
    except Exception as e:
        st.error(f"Error dalam prediksi: {str(e)}")
        return None, None

# ==================== MAIN APPLICATION ====================
def main():
    st.set_page_config(
        page_title="Klasifikasi Pengguna AI - Random Forest",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Analisis Klasifikasi Pengguna AI")
    st.markdown("**Klasifikasi Light vs Moderate vs Heavy User menggunakan Random Forest**")
    st.markdown("---")
    
    # Inisialisasi session state
    if 'df' not in st.session_state:
        st.session_state.df = create_sample_dataset()
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Konfigurasi")
        
        st.subheader("Dataset")
        if st.button("🔄 Generate Dataset Baru", type="secondary"):
            st.session_state.df = create_sample_dataset()
            st.session_state.analysis_results = None
            st.success("Dataset baru dibuat!")
        
        st.markdown("---")
        st.subheader("Parameter Model")
        
        n_estimators = st.slider("Jumlah Estimator", 50, 500, 100)
        max_depth = st.slider("Max Depth", 5, 30, 10)
        test_size = st.slider("Test Size (%)", 10, 40, 30) / 100
        
        st.markdown("---")
        st.subheader("Aksi")
        
        if st.button("🚀 Jalankan Analisis", type="primary", use_container_width=True):
            st.session_state.run_analysis = True
    
    # Tab utama
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset", "📈 Analisis", "🎯 Prediksi", "📤 Ekspor"])
    
    # TAB 1: Dataset
    with tab1:
        st.header("Dataset & Klasifikasi")
        
        df = st.session_state.df
        
        # Tampilkan informasi dataset
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Siswa", len(df))
        with col2:
            light_users = len(df[df['AI_User_Class'] == 0])
            st.metric("Light Users", light_users)
        with col3:
            moderate_users = len(df[df['AI_User_Class'] == 1])
            st.metric("Moderate Users", moderate_users)
        with col4:
            heavy_users = len(df[df['AI_User_Class'] == 2])
            st.metric("Heavy Users", heavy_users)
        
        # Visualisasi distribusi
        st.subheader("Distribusi Klasifikasi Pengguna AI")
        fig_dist = plot_user_class_distribution(df)
        st.pyplot(fig_dist)
        
        # Tampilkan kriteria klasifikasi
        with st.expander("📋 Kriteria Klasifikasi", expanded=True):
            st.markdown("""
            ### **Klasifikasi Berdasarkan Daily Usage (jam/minggu):**
            
            | Kategori | Daily Usage | Deskripsi |
            |----------|-------------|-----------|
            | **Light User** | ≤ 10 jam | Penggunaan terbatas |
            | **Moderate User** | 11-30 jam | Penggunaan reguler |
            | **Heavy User** | > 30 jam | Penggunaan intensif |
            
            ### **Karakteristik:**
            - **Light User**: Menggunakan AI untuk tugas spesifik
            - **Moderate User**: Menggunakan AI secara rutin untuk berbagai kebutuhan
            - **Heavy User**: Bergantung pada AI untuk hampir semua aktivitas akademik
            """)
        
        # Tampilkan data
        st.subheader("Preview Data")
        df_display = df.copy()
        df_display['AI_User_Class'] = df_display['AI_User_Class'].map(USER_CLASSES)
        
        # Filter
        filter_option = st.selectbox(
            "Filter berdasarkan kelas:",
            ["Semua", "Light User", "Moderate User", "Heavy User"]
        )
        
        if filter_option != "Semua":
            df_display = df_display[df_display['AI_User_Class'] == filter_option]
        
        st.dataframe(df_display, use_container_width=True, height=300)
    
    # TAB 2: Analisis
    with tab2:
        st.header("Analisis Random Forest")
        
        if hasattr(st.session_state, 'run_analysis'):
            with st.spinner("Melakukan analisis dengan Random Forest..."):
                # Preprocess data
                X, y, label_encoders = preprocess_data(st.session_state.df)
                
                if X is not None:
                    # Lakukan analisis
                    analysis_results = perform_random_forest_analysis(
                        X, y, 
                        test_size=test_size,
                        n_estimators=n_estimators,
                        max_depth=max_depth
                    )
                    
                    # Simpan hasil
                    st.session_state.analysis_results = analysis_results
                    st.session_state.X_columns = X.columns
                    st.session_state.label_encoders = label_encoders
                    
                    # Tampilkan hasil
                    st.success("✅ Analisis selesai!")
                    
                    # Metrics utama
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Accuracy", f"{analysis_results['accuracy_test']:.2%}")
                    with col2:
                        st.metric("Precision (Avg)", f"{np.mean(analysis_results['precision_per_class']):.2%}")
                    with col3:
                        st.metric("Recall (Avg)", f"{np.mean(analysis_results['recall_per_class']):.2%}")
                    with col4:
                        st.metric("F1-Score (Avg)", f"{np.mean(analysis_results['f1_per_class']):.2%}")
                    
                    # Visualisasi lengkap
                    st.subheader("Visualisasi Hasil Analisis")
                    fig_analysis = plot_analysis_results(analysis_results)
                    st.pyplot(fig_analysis)
                    
                    # Feature importance detail
                    with st.expander("📋 Detail Feature Importance"):
                        st.dataframe(analysis_results['feature_importance'], use_container_width=True)
                        
                        # Cumulative importance
                        cumulative_importance = analysis_results['feature_importance']['importance'].cumsum()
                        fig_cum, ax_cum = plt.subplots(figsize=(10, 4))
                        ax_cum.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, marker='o')
                        ax_cum.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='80% Importance')
                        ax_cum.set_xlabel('Number of Features')
                        ax_cum.set_ylabel('Cumulative Importance')
                        ax_cum.set_title('Cumulative Feature Importance')
                        ax_cum.legend()
                        ax_cum.grid(True, alpha=0.3)
                        st.pyplot(fig_cum)
        else:
            st.info("👈 Klik 'Jalankan Analisis' di sidebar untuk memulai")
    
    # TAB 3: Prediksi
    with tab3:
        st.header("Prediksi Individu")
        
        if st.session_state.analysis_results is not None:
            st.success("Model siap untuk prediksi!")
            
            # Form input
            input_data = create_prediction_form()
            
            if input_data:
                # Prediksi
                prediction, probabilities = predict_user_class(
                    input_data,
                    st.session_state.analysis_results['model'],
                    st.session_state.analysis_results['scaler'],
                    st.session_state.X_columns
                )
                
                if prediction is not None:
                    # Tampilkan hasil
                    st.subheader("Hasil Prediksi")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Klasifikasi", USER_CLASSES[prediction])
                    with col2:
                        prob_value = probabilities[prediction] * 100
                        st.metric("Confidence", f"{prob_value:.1f}%")
                    with col3:
                        st.metric("Daily Usage", f"{input_data['Daily_Usage']} jam/minggu")
                    
                    # Probabilitas detail
                    st.subheader("Probabilitas Klasifikasi")
                    
                    prob_df = pd.DataFrame({
                        'Kelas': [USER_CLASSES[i] for i in range(len(probabilities))],
                        'Probabilitas': [f"{p*100:.1f}%" for p in probabilities]
                    })
                    
                    # Visualisasi probabilitas
                    fig_prob, ax_prob = plt.subplots(figsize=(10, 4))
                    colors_prob = ['#FF9999', '#66B2FF', '#99FF99']
                    bars = ax_prob.bar(prob_df['Kelas'], 
                                      [float(p.strip('%')) for p in prob_df['Probabilitas']], 
                                      color=colors_prob)
                    ax_prob.set_ylabel('Probabilitas (%)')
                    ax_prob.set_title('Distribusi Probabilitas Klasifikasi')
                    ax_prob.grid(axis='y', alpha=0.3)
                    
                    for bar, prob in zip(bars, prob_df['Probabilitas']):
                        height = bar.get_height()
                        ax_prob.text(bar.get_x() + bar.get_width()/2., height + 1,
                                   prob, ha='center', va='bottom', fontweight='bold')
                    
                    st.pyplot(fig_prob)
                    
                    # Rekomendasi
                    st.subheader("Rekomendasi")
                    
                    if prediction == 0:  # Light User
                        st.info("""
                        **🎯 Anda termasuk Light User:**
                        - **Saran**: Tingkatkan penggunaan AI secara bertahap
                        - **Tips**: Mulai dengan 2-3 jam tambahan per minggu
                        - **Tools**: Coba ChatGPT untuk tugas menulis, Gemini untuk research
                        """)
                    elif prediction == 1:  # Moderate User
                        st.success("""
                        **🎯 Anda termasuk Moderate User:**
                        - **Status**: Penggunaan optimal, pertahankan!
                        - **Tips**: Diversifikasi tools untuk kebutuhan berbeda
                        - **Goal**: Optimalkan produktivitas dengan workflow yang efisien
                        """)
                    else:  # Heavy User
                        st.warning("""
                        **🎯 Anda termasuk Heavy User:**
                        - **Perhatian**: Evaluasi efektivitas penggunaan
                        - **Tips**: Pastikan AI meningkatkan kualitas belajar, bukan hanya kuantitas
                        - **Balance**: Kombinasikan dengan metode belajar tradisional
                        """)
        else:
            st.info("👈 Jalankan analisis terlebih dahulu untuk menggunakan fitur prediksi")
    
    # TAB 4: Ekspor
    with tab4:
        st.header("Ekspor Hasil")
        
        if st.session_state.analysis_results is not None:
            st.success("Hasil analisis siap diekspor!")
            
            # Tombol ekspor
            if st.button("📥 Ekspor Semua Hasil ke CSV", type="primary"):
                with st.spinner("Mengekspor hasil..."):
                    export_folder = export_to_csv(
                        st.session_state.df,
                        st.session_state.analysis_results
                    )
                    
                    st.success(f"✅ Hasil berhasil diekspor ke folder: `{export_folder}/`")
                    
                    # Tampilkan file yang diekspor
                    st.subheader("File yang Telah Diekspor")
                    
                    files = os.listdir(export_folder)
                    
                    for file in files:
                        if file.endswith('.csv'):
                            col1, col2, col3 = st.columns([3, 1, 1])
                            
                            with col1:
                                st.write(f"`{file}`")
                            with col2:
                                file_path = os.path.join(export_folder, file)
                                file_size = os.path.getsize(file_path) / 1024
                                st.write(f"{file_size:.1f} KB")
                            with col3:
                                with open(file_path, 'rb') as f:
                                    st.download_button(
                                        label="📥",
                                        data=f,
                                        file_name=file,
                                        mime="text/csv",
                                        key=f"download_{file}"
                                    )
        else:
            st.info("👈 Jalankan analisis terlebih dahulu untuk mengekspor hasil")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        <p><b>Analisis Klasifikasi Pengguna AI</b> | Random Forest Classification</p>
        <p>Light User • Moderate User • Heavy User</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
