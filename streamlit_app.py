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

def prepare_dataset_from_text(text_data):
    """Mempersiapkan dataset dari text yang diberikan"""
    try:
        # Parsing data text menjadi DataFrame
        lines = text_data.strip().split('\n')
        
        # Ambil header
        header = lines[0].split(':')
        
        # Parse data
        data_rows = []
        for line in lines[1:]:
            if line.strip():
                values = line.split()
                if len(values) >= 6:  # Minimal memiliki kolom yang diperlukan
                    data_rows.append(values)
        
        # Buat DataFrame
        df = pd.DataFrame(data_rows)
        
        # Beri nama kolom berdasarkan data
        if len(df.columns) >= 6:
            df.columns = ['Student_Name', 'College', 'Stream', 'Year', 'AI_Tools', 'Daily_Usage', 
                         'Use_Cases', 'Trust_Level', 'Impact_Score'][:len(df.columns)]
        
        # Konversi tipe data
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df['Daily_Usage'] = pd.to_numeric(df['Daily_Usage'], errors='coerce')
        df['Trust_Level'] = pd.to_numeric(df['Trust_Level'], errors='coerce')
        df['Impact_Score'] = pd.to_numeric(df['Impact_Score'], errors='coerce')
        
        # Buat kolom target (klasifikasi penggunaan AI)
        df['AI_User_Class'] = df.apply(classify_ai_usage, axis=1)
        
        # Tambah fitur-fitur tambahan
        df['Multiple_Tools'] = df['AI_Tools'].apply(lambda x: 1 if ',' in str(x) else 0)
        df['Usage_per_Year'] = df['Daily_Usage'] / df['Year'].clip(lower=1)
        
        return df
        
    except Exception as e:
        st.error(f"Error preparing dataset: {str(e)}")
        return None

def preprocess_data(df):
    """Fungsi untuk preprocessing data lengkap"""
    
    df_processed = df.copy()
    
    # Drop kolom yang tidak diperlukan
    cols_to_drop = ['Student_Name']
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
    precision_per_class = precision_score(y_test, y_pred, average=None)
    recall_per_class = recall_score(y_test, y_pred, average=None)
    f1_per_class = f1_score(y_test, y_pred, average=None)
    
    return {
        'model': rf_model,
        'scaler': scaler,
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
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
def export_results_to_csv(df, analysis_results, label_encoders, prefix='ai_analysis'):
    """Ekspor hasil analisis ke CSV"""
    
    # Buat folder ekspor jika belum ada
    export_folder = 'exported_results'
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)
    
    results = analysis_results
    
    # 1. Ekspor dataset dengan klasifikasi
    df_with_class = df.copy()
    df_with_class['AI_User_Class_Label'] = df_with_class['AI_User_Class'].map(USER_CLASSES)
    df_with_class.to_csv(f'{export_folder}/{prefix}_classified_data.csv', index=False)
    
    # 2. Ekspor hasil prediksi
    predictions_df = pd.DataFrame({
        'Actual_Class': results['y_test'],
        'Actual_Label': [USER_CLASSES[i] for i in results['y_test']],
        'Predicted_Class': results['y_pred'],
        'Predicted_Label': [USER_CLASSES[i] for i in results['y_pred']],
        'Is_Correct': results['y_test'] == results['y_pred']
    })
    predictions_df.to_csv(f'{export_folder}/{prefix}_predictions.csv', index=False)
    
    # 3. Ekspor feature importance
    results['feature_importance'].to_csv(f'{export_folder}/{prefix}_feature_importance.csv', index=False)
    
    # 4. Ekspor classification report
    class_report_df = pd.DataFrame(results['classification_report']).transpose()
    class_report_df.to_csv(f'{export_folder}/{prefix}_classification_report.csv')
    
    # 5. Ekspor confusion matrix
    cm_df = pd.DataFrame(results['confusion_matrix'], 
                        columns=[USER_CLASSES[i] for i in range(results['confusion_matrix'].shape[1])],
                        index=[USER_CLASSES[i] for i in range(results['confusion_matrix'].shape[0])])
    cm_df.to_csv(f'{export_folder}/{prefix}_confusion_matrix.csv')
    
    # 6. Ekspor model metrics
    metrics = pd.DataFrame({
        'Metric': ['Train Accuracy', 'Test Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1'],
        'Value': [
            results['accuracy_train'],
            results['accuracy_test'],
            np.mean(results['precision_per_class']),
            np.mean(results['recall_per_class']),
            np.mean(results['f1_per_class'])
        ]
    })
    metrics.to_csv(f'{export_folder}/{prefix}_model_metrics.csv', index=False)
    
    # 7. Ekspor statistik per kelas
    class_stats = pd.DataFrame({
        'Class': [USER_CLASSES[i] for i in range(len(USER_CLASSES))],
        'Precision': results['precision_per_class'],
        'Recall': results['recall_per_class'],
        'F1_Score': results['f1_per_class']
    })
    class_stats.to_csv(f'{export_folder}/{prefix}_class_statistics.csv', index=False)
    
    return export_folder

# ==================== PREDIKSI INDIVIDU ====================
def predict_individual_user(model, scaler, input_data, feature_names):
    """Prediksi klasifikasi untuk pengguna individu"""
    try:
        # Buat DataFrame dari input
        input_df = pd.DataFrame([input_data])
        
        # Pastikan urutan kolom sama dengan training
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[feature_names]
        
        # Scale data
        input_scaled = scaler.transform(input_df)
        
        # Prediksi
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        return {
            'class': prediction,
            'label': USER_CLASSES[prediction],
            'probabilities': {
                USER_CLASSES[i]: f"{prob*100:.2f}%" 
                for i, prob in enumerate(probability)
            }
        }
    except Exception as e:
        return {'error': str(e)}

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
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'label_encoders' not in st.session_state:
        st.session_state.label_encoders = None
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Konfigurasi")
        
        # Input data
        st.subheader("1. Input Data")
        data_option = st.radio(
            "Pilih sumber data:",
            ["Upload CSV", "Tempel Data Text", "Gunakan Contoh Data"]
        )
        
        if data_option == "Upload CSV":
            uploaded_file = st.file_uploader("📁 Upload dataset CSV", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    # Cek apakah ada kolom Daily_Usage
                    if 'Daily_Usage' not in df.columns:
                        st.error("Dataset harus memiliki kolom 'Daily_Usage'!")
                    else:
                        # Tambah kolom klasifikasi
                        df['AI_User_Class'] = df.apply(classify_ai_usage, axis=1)
                        st.session_state.df = df
                        st.success(f"✅ Dataset dimuat: {len(df)} baris")
                        
                except Exception as e:
                    st.error(f"Error membaca file: {str(e)}")
        
        elif data_option == "Tempel Data Text":
            text_data = st.text_area("Tempel data text (format tab atau spasi):", height=200)
            
            if st.button("Proses Data Text", type="primary"):
                if text_data:
                    with st.spinner("Memproses data text..."):
                        df = prepare_dataset_from_text(text_data)
                        if df is not None:
                            st.session_state.df = df
                            st.success(f"✅ Dataset diproses: {len(df)} baris")
        
        else:  # Contoh data
            if st.button("Generate Contoh Data", type="primary"):
                # Buat data contoh
                np.random.seed(42)
                n_samples = 300
                
                data = {
                    'Student_Name': [f'Student_{i}' for i in range(n_samples)],
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
                st.session_state.df = df
                st.success(f"✅ Contoh data dibuat: {len(df)} baris")
        
        st.markdown("---")
        st.subheader("2. Parameter Model")
        
        n_estimators = st.slider("Jumlah Estimator", 50, 500, 100)
        max_depth = st.slider("Max Depth", 5, 30, 10)
        test_size = st.slider("Test Size (%)", 10, 40, 30) / 100
        
        st.markdown("---")
        st.subheader("3. Aksi")
        
        if st.button("🚀 Jalankan Analisis", type="primary", use_container_width=True):
            if st.session_state.df is not None:
                st.session_state.run_analysis = True
            else:
                st.error("⚠️ Mohon muat dataset terlebih dahulu!")
    
    # Tab utama
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset", "📈 Analisis", "🎯 Prediksi", "📤 Ekspor"])
    
    # TAB 1: Dataset
    with tab1:
        st.header("Dataset & Klasifikasi")
        
        if st.session_state.df is not None:
            df = st.session_state.df
            
            # Tampilkan informasi dataset
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Siswa", len(df))
            with col2:
                light_users = len(df[df['AI_User_Class'] == 0])
                st.metric("Light Users", light_users)
            with col3:
                heavy_users = len(df[df['AI_User_Class'] == 2])
                st.metric("Heavy Users", heavy_users)
            
            # Visualisasi distribusi
            st.subheader("Distribusi Klasifikasi Pengguna AI")
            fig_dist = plot_user_class_distribution(df)
            st.pyplot(fig_dist)
            
            # Tampilkan kriteria klasifikasi
            with st.expander("📋 Kriteria Klasifikasi", expanded=True):
                st.markdown("""
                | Kategori | Daily Usage | Deskripsi |
                |----------|-------------|-----------|
                | **Light User** | ≤ 10 jam/minggu | Penggunaan terbatas, hanya untuk tugas tertentu |
                | **Moderate User** | 11-30 jam/minggu | Penggunaan reguler untuk berbagai aktivitas akademik |
                | **Heavy User** | > 30 jam/minggu | Penggunaan intensif, hampir setiap hari untuk banyak tujuan |
                """)
            
            # Tampilkan data
            st.subheader("Preview Data")
            
            # Pilih kolom untuk ditampilkan
            display_cols = ['Student_Name', 'College', 'Year', 'AI_Tools', 'Daily_Usage', 
                          'AI_User_Class']
            
            df_display = df[display_cols].copy()
            df_display['AI_User_Class'] = df_display['AI_User_Class'].map(USER_CLASSES)
            
            # Filter berdasarkan kelas
            filter_class = st.selectbox("Filter berdasarkan kelas:", 
                                       ["Semua", "Light User", "Moderate User", "Heavy User"])
            
            if filter_class != "Semua":
                class_map = {v: k for k, v in USER_CLASSES.items()}
                df_display = df_display[df_display['AI_User_Class'] == filter_class]
            
            st.dataframe(df_display, use_container_width=True, height=300)
            
            # Statistik deskriptif
            with st.expander("📊 Statistik Deskriptif"):
                st.dataframe(df.describe(), use_container_width=True)
                
        else:
            st.info("👈 Silakan pilih sumber data di sidebar")
    
    # TAB 2: Analisis
    with tab2:
        st.header("Analisis Random Forest")
        
        if st.session_state.df is not None and hasattr(st.session_state, 'run_analysis'):
            with st.spinner("Melakukan analisis dengan Random Forest..."):
                # Preprocess data
                X, y, label_encoders = preprocess_data(st.session_state.df)
                
                if X is not None:
                    # Simpan ke session state
                    st.session_state.label_encoders = label_encoders
                    
                    # Lakukan analisis
                    analysis_results = perform_random_forest_analysis(
                        X, y, 
                        test_size=test_size,
                        n_estimators=n_estimators,
                        max_depth=max_depth
                    )
                    
                    # Simpan hasil
                    st.session_state.analysis_results = analysis_results
                    st.session_state.X = X
                    
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
                    
                    # Interpretasi hasil
                    st.subheader("📝 Interpretasi Hasil")
                    
                    col_int1, col_int2 = st.columns(2)
                    
                    with col_int1:
                        st.markdown("""
                        **🎯 Kinerja Model:**
                        - Model mencapai akurasi **{:.2%}** pada data testing
                        - **Light User**: Precision {:.2%}, Recall {:.2%}
                        - **Moderate User**: Precision {:.2%}, Recall {:.2%}
                        - **Heavy User**: Precision {:.2%}, Recall {:.2%}
                        
                        **📊 Insight:**
                        - Model paling baik dalam mengklasifikasikan **{}**
                        - Fitur paling penting: **{}**
                        """.format(
                            analysis_results['accuracy_test'],
                            analysis_results['precision_per_class'][0],
                            analysis_results['recall_per_class'][0],
                            analysis_results['precision_per_class'][1],
                            analysis_results['recall_per_class'][1],
                            analysis_results['precision_per_class'][2],
                            analysis_results['recall_per_class'][2],
                            USER_CLASSES[np.argmax(analysis_results['f1_per_class'])],
                            analysis_results['feature_importance'].iloc[0]['feature']
                        ))
                    
                    with col_int2:
                        st.markdown("""
                        **💡 Rekomendasi:**
                        
                        1. **Untuk Light Users:**
                           - Tingkatkan penggunaan AI secara bertahap
                           - Fokus pada tools yang mudah digunakan
                        
                        2. **Untuk Moderate Users:**
                           - Diversifikasi penggunaan tools
                           - Optimalkan untuk produktivitas
                        
                        3. **Untuk Heavy Users:**
                           - Evaluasi efektivitas penggunaan
                           - Cegah ketergantungan berlebihan
                        
                        **🔧 Improvement:**
                        - {} fitur menyumbang 80% importance
                        - Pertimbangkan feature engineering untuk fitur lemah
                        """.format(
                            len(analysis_results['feature_importance'][
                                analysis_results['feature_importance']['importance'].cumsum() <= 0.8
                            ])
                        ))
                    
                    # Feature importance detail
                    with st.expander("📋 Detail Feature Importance", expanded=False):
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
            
            # Form input untuk prediksi
            col1, col2 = st.columns(2)
            
            with col1:
                daily_usage = st.slider("Daily Usage (jam/minggu)", 1, 50, 15)
                year = st.selectbox("Tahun Studi", [1, 2, 3, 4])
                trust_level = st.slider("Trust Level (1-5)", 1, 5, 3)
            
            with col2:
                ai_tools = st.selectbox("AI Tools", ["ChatGPT", "Gemini", "Copilot", "Multiple"])
                use_cases = st.selectbox("Use Cases", ["Assignments", "Exam Prep", "Research", "Projects"])
                college = st.selectbox("College", ["Engineering", "Commerce", "Science", "Arts", "Medical"])
            
            # Fitur tambahan
            multiple_tools = 1 if ai_tools == "Multiple" else 0
            usage_per_year = daily_usage / year if year > 0 else daily_usage
            
            if st.button("🔮 Prediksi Klasifikasi", type="primary"):
                # Siapkan input data
                input_data = {
                    'Daily_Usage': daily_usage,
                    'Year': year,
                    'Trust_Level': trust_level,
                    'AI_Tools': ai_tools,
                    'Use_Cases': use_cases,
                    'College': college,
                    'Multiple_Tools': multiple_tools,
                    'Usage_per_Year': usage_per_year
                }
                
                # Prediksi
                model = st.session_state.analysis_results['model']
                scaler = st.session_state.analysis_results['scaler']
                
                result = predict_individual_user(
                    model, scaler, input_data, st.session_state.X.columns
                )
                
                if 'error' not in result:
                    # Tampilkan hasil
                    st.subheader("Hasil Prediksi")
                    
                    col_res1, col_res2, col_res3 = st.columns(3)
                    
                    with col_res1:
                        st.metric("Klasifikasi", result['label'])
                    
                    with col_res2:
                        st.metric("Confidence", result['probabilities'][result['label']])
                    
                    with col_res3:
                        st.metric("Daily Usage", f"{daily_usage} jam/minggu")
                    
                    # Probabilitas detail
                    st.subheader("Probabilitas Klasifikasi")
                    
                    prob_df = pd.DataFrame({
                        'Kelas': list(result['probabilities'].keys()),
                        'Probabilitas': list(result['probabilities'].values())
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
                    
                    # Rekomendasi berdasarkan klasifikasi
                    st.subheader("Rekomendasi")
                    
                    if result['class'] == 0:  # Light User
                        st.info("""
                        **🎯 Anda termasuk Light User:**
                        - **Saran**: Coba tingkatkan penggunaan AI secara bertahap
                        - **Tools**: Mulai dengan ChatGPT atau Gemini untuk tugas sederhana
                        - **Waktu**: Tambah 2-3 jam/minggu untuk eksplorasi
                        """)
                    elif result['class'] == 1:  # Moderate User
                        st.success("""
                        **🎯 Anda termasuk Moderate User:**
                        - **Status**: Penggunaan optimal, pertahankan!
                        - **Tips**: Diversifikasi tools untuk kebutuhan berbeda
                        - **Goal**: Optimalkan untuk produktivitas maksimal
                        """)
                    else:  # Heavy User
                        st.warning("""
                        **🎯 Anda termasuk Heavy User:**
                        - **Perhatian**: Pastikan penggunaan tetap produktif
                        - **Evaluasi**: Tinjau efektivitas penggunaan secara berkala
                        - **Balance**: Jaga keseimbangan dengan metode belajar tradisional
                        """)
                else:
                    st.error(f"Error dalam prediksi: {result['error']}")
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
                    export_folder = export_results_to_csv(
                        st.session_state.df,
                        st.session_state.analysis_results,
                        st.session_state.label_encoders,
                        prefix="ai_user_classification"
                    )
                    
                    st.success(f"✅ Hasil berhasil diekspor ke folder: `{export_folder}/`")
                    
                    # Tampilkan file yang diekspor
                    st.subheader("File yang Telah Diekspor")
                    
                    files = os.listdir(export_folder)
                    ai_files = [f for f in files if f.startswith("ai_user_classification")]
                    
                    for file in ai_files:
                        file_path = os.path.join(export_folder, file)
                        file_size = os.path.getsize(file_path
