# app.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import csv
import os
import io

# ==================== PREPROCESSING DATA ====================
def preprocess_data(df):
    """Fungsi untuk preprocessing data"""
    
    # Simpan data asli untuk referensi
    df_original = df.copy()
    
    # Identifikasi kolom numerik dan kategorikal
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Encoding variabel kategorikal
    label_encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    return df, df_original, label_encoders

def load_and_prepare_data(file_path):
    """Load dan prepare dataset"""
    try:
        # Baca dataset
        df = pd.read_csv(file_path)
        
        # Jika ada kolom tanpa nama, beri nama
        df.columns = [f'col_{i}' if col.startswith('Unnamed') else col for i, col in enumerate(df.columns)]
        
        # Analisis kolom target (asumsi kolom terakhir adalah target)
        target_col = df.columns[-1]
        
        # Pisahkan fitur dan target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Preprocessing
        X_processed, X_original, label_encoders = preprocess_data(X)
        
        return X_processed, y, X_original, label_encoders, target_col, df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None, None, None

# ==================== ANALISIS RANDOM FOREST ====================
def perform_random_forest_analysis(X, y, test_size=0.3, random_state=42, n_estimators=100, max_depth=10):
    """Melakukan analisis dengan Random Forest"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Inisialisasi dan train model
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Prediksi
    y_pred = rf_model.predict(X_test)
    y_pred_train = rf_model.predict(X_train)
    
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
    
    return {
        'model': rf_model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'accuracy_test': accuracy_test,
        'accuracy_train': accuracy_train,
        'classification_report': class_report,
        'confusion_matrix': cm,
        'feature_importance': feature_importance
    }

# ==================== VISUALISASI ====================
def plot_results(analysis_results):
    """Plot hasil analisis"""
    results = analysis_results
    
    # Buat figure dengan 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Confusion Matrix
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # 2. Feature Importance (Top 10)
    top_features = results['feature_importance'].head(10)
    axes[0, 1].barh(range(len(top_features)), top_features['importance'])
    axes[0, 1].set_yticks(range(len(top_features)))
    axes[0, 1].set_yticklabels(top_features['feature'])
    axes[0, 1].set_title('Top 10 Feature Importance')
    axes[0, 1].set_xlabel('Importance Score')
    
    # 3. Accuracy Comparison
    accuracy_data = {
        'Train': results['accuracy_train'],
        'Test': results['accuracy_test']
    }
    axes[1, 0].bar(accuracy_data.keys(), accuracy_data.values(), color=['blue', 'green'])
    axes[1, 0].set_title('Model Accuracy Comparison')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_ylim(0, 1)
    for i, v in enumerate(accuracy_data.values()):
        axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    # 4. Classification Metrics
    class_report = results['classification_report']
    if 'accuracy' in class_report:
        del class_report['accuracy']
    
    metrics_df = pd.DataFrame(class_report).transpose()
    if 'support' in metrics_df.columns:
        metrics_to_plot = metrics_df[['precision', 'recall', 'f1-score']]
        metrics_to_plot.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Classification Metrics per Class')
        axes[1, 1].set_xlabel('Class')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend(loc='lower right')
        axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    return fig

# ==================== EKSPOR DATA ====================
def export_results_to_csv(analysis_results, label_encoders=None, prefix='analysis'):
    """Ekspor hasil analisis ke CSV"""
    
    # Buat folder ekspor jika belum ada
    export_folder = 'exported_results'
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)
    
    results = analysis_results
    
    # 1. Ekspor dataset yang sudah diproses
    processed_data = pd.concat([
        pd.DataFrame(results['X_train'], columns=results['X_train'].columns),
        pd.DataFrame(results['y_train'].values, columns=['target'])
    ], axis=1)
    processed_data.to_csv(f'{export_folder}/{prefix}_processed_data.csv', index=False)
    
    # 2. Ekspor hasil prediksi
    predictions = pd.DataFrame({
        'y_test': results['y_test'].values,
        'y_pred': results['y_pred']
    })
    predictions.to_csv(f'{export_folder}/{prefix}_predictions.csv', index=False)
    
    # 3. Ekspor feature importance
    results['feature_importance'].to_csv(f'{export_folder}/{prefix}_feature_importance.csv', index=False)
    
    # 4. Ekspor classification report
    class_report_df = pd.DataFrame(results['classification_report']).transpose()
    class_report_df.to_csv(f'{export_folder}/{prefix}_classification_report.csv')
    
    # 5. Ekspor confusion matrix
    cm_df = pd.DataFrame(results['confusion_matrix'])
    cm_df.to_csv(f'{export_folder}/{prefix}_confusion_matrix.csv', index=False)
    
    # 6. Ekspor model metrics
    metrics = pd.DataFrame({
        'Metric': ['Train Accuracy', 'Test Accuracy'],
        'Value': [results['accuracy_train'], results['accuracy_test']]
    })
    metrics.to_csv(f'{export_folder}/{prefix}_model_metrics.csv', index=False)
    
    # 7. Ekspor label encoders mapping (jika ada)
    if label_encoders:
        encoder_mappings = {}
        for col, le in label_encoders.items():
            encoder_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
        
        # Simpan sebagai JSON-like CSV
        encoder_df = pd.DataFrame([encoder_mappings])
        encoder_df.to_csv(f'{export_folder}/{prefix}_encoder_mappings.csv', index=False)
    
    return export_folder

# ==================== CREATE SAMPLE DATASET ====================
def create_sample_dataset():
    """Membuat dataset contoh untuk demo"""
    np.random.seed(42)
    
    n_samples = 200
    
    # Buat fitur-fitur
    data = {
        'Student_ID': [f'STD{i:03d}' for i in range(1, n_samples+1)],
        'College': np.random.choice(['Engineering', 'Commerce', 'Science', 'Arts', 'Medical'], n_samples),
        'Year_of_Study': np.random.randint(1, 5, n_samples),
        'AI_Tools_Used': np.random.choice(['ChatGPT', 'Gemini', 'Copilot', 'Bard', 'Multiple'], n_samples),
        'Daily_Usage_Hours': np.random.randint(1, 10, n_samples),
        'Use_Cases': np.random.choice(['Assignments', 'Exam Prep', 'Content Writing', 'MCQ Practice', 'Learning'], n_samples),
        'Trust_in_AI': np.random.randint(1, 6, n_samples),
        'Previous_GPA': np.random.uniform(2.0, 4.0, n_samples),
        'Study_Hours_Daily': np.random.randint(2, 12, n_samples),
        'Impact_on_Performance': np.random.choice(['High Improvement', 'Moderate Improvement', 
                                                  'No Change', 'Negative Impact'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Simpan ke CSV
    df.to_csv('sample_student_data.csv', index=False)
    return df

# ==================== MAIN APPLICATION ====================
def main():
    st.set_page_config(
        page_title="AI Usage Analysis - Random Forest",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🎓 Analisis Tingkat Penggunaan AI terhadap Performa Akademik")
    st.markdown("**Menggunakan Algoritma Random Forest**")
    st.markdown("---")
    
    # Inisialisasi session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Sidebar untuk kontrol
    with st.sidebar:
        st.header("⚙️ Kontrol Analisis")
        
        st.subheader("1. Sumber Data")
        data_source = st.radio(
            "Pilih sumber data:",
            ["Upload Dataset", "Gunakan Dataset Contoh"]
        )
        
        if data_source == "Upload Dataset":
            uploaded_file = st.file_uploader("📁 Upload dataset CSV", type=['csv'])
            
            if uploaded_file is not None:
                # Simpan file sementara
                with open("temp_dataset.csv", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                if st.button("🔍 Proses Dataset", type="primary"):
                    with st.spinner("Memproses dataset..."):
                        X, y, X_original, label_encoders, target_col, df = load_and_prepare_data("temp_dataset.csv")
                        
                        if X is not None:
                            st.session_state.data_loaded = True
                            st.session_state.X = X
                            st.session_state.y = y
                            st.session_state.X_original = X_original
                            st.session_state.label_encoders = label_encoders
                            st.session_state.target_col = target_col
                            st.session_state.df = df
                            st.success("✅ Dataset berhasil diproses!")
        
        else:  # Gunakan dataset contoh
            if st.button("📊 Generate Dataset Contoh", type="primary"):
                with st.spinner("Membuat dataset contoh..."):
                    df = create_sample_dataset()
                    
                    # Load dan prepare data
                    X, y, X_original, label_encoders, target_col, df_processed = load_and_prepare_data('sample_student_data.csv')
                    
                    if X is not None:
                        st.session_state.data_loaded = True
                        st.session_state.X = X
                        st.session_state.y = y
                        st.session_state.X_original = X_original
                        st.session_state.label_encoders = label_encoders
                        st.session_state.target_col = target_col
                        st.session_state.df = df
                        st.success("✅ Dataset contoh berhasil dibuat!")
        
        st.markdown("---")
        st.subheader("2. Parameter Model")
        
        n_estimators = st.slider("Jumlah Estimator", 10, 500, 100)
        max_depth = st.slider("Max Depth", 2, 50, 10)
        test_size = st.slider("Test Size (%)", 10, 50, 30) / 100
        random_state = st.number_input("Random State", 0, 100, 42)
        
        st.markdown("---")
        st.subheader("3. Aksi")
        
        col1, col2 = st.columns(2)
        with col1:
            run_analysis = st.button("🚀 Jalankan Analisis", type="primary", use_container_width=True)
        with col2:
            export_data = st.button("📥 Ekspor Hasil", use_container_width=True)
    
    # ==================== TAB UTAMA ====================
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Dataset", "🔧 Preprocessing", "🤖 Analisis", "📈 Evaluasi"])
    
    # TAB 1: Dataset
    with tab1:
        st.header("Dataset Information")
        
        if st.session_state.data_loaded:
            st.success(f"✅ Dataset berhasil dimuat: {len(st.session_state.df)} baris × {len(st.session_state.df.columns)} kolom")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Preview Data")
                st.dataframe(st.session_state.df.head(), use_container_width=True)
                
                st.subheader("Informasi Dataset")
                st.write(f"- **Jumlah Sampel:** {st.session_state.df.shape[0]}")
                st.write(f"- **Jumlah Fitur:** {st.session_state.df.shape[1] - 1}")
                st.write(f"- **Variabel Target:** {st.session_state.target_col}")
                st.write(f"- **Tipe Dataset:** {data_source}")
            
            with col2:
                st.subheader("Statistik Deskriptif")
                st.dataframe(st.session_state.df.describe(), use_container_width=True)
                
                st.subheader("Info Kolom")
                info_df = pd.DataFrame({
                    'Kolom': st.session_state.df.columns,
                    'Tipe Data': st.session_state.df.dtypes.values,
                    'Non-Null Count': st.session_state.df.notnull().sum().values
                })
                st.dataframe(info_df, use_container_width=True)
        else:
            st.info("👈 Silakan pilih sumber data dan proses dataset terlebih dahulu di sidebar")
    
    # TAB 2: Preprocessing
    with tab2:
        st.header("Data Preprocessing")
        
        if st.session_state.data_loaded:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Data Asli")
                st.write("**Data sebelum preprocessing:**")
                st.dataframe(st.session_state.X_original.head(), use_container_width=True)
                
                st.subheader("Missing Values Check")
                missing_data = st.session_state.df.isnull().sum()
                missing_df = pd.DataFrame({
                    'Kolom': missing_data.index,
                    'Missing Values': missing_data.values,
                    'Persentase': (missing_data.values / len(st.session_state.df) * 100).round(2)
                })
                st.dataframe(missing_df[missing_df['Missing Values'] > 0], use_container_width=True)
                
                if len(missing_df[missing_df['Missing Values'] > 0]) == 0:
                    st.success("✅ Tidak ada missing values")
            
            with col2:
                st.subheader("Data Setelah Preprocessing")
                st.write("**Data setelah encoding dan cleaning:**")
                st.dataframe(st.session_state.X.head(), use_container_width=True)
                
                st.subheader("Encoding Information")
                if st.session_state.label_encoders:
                    st.write(f"**Jumlah variabel kategorikal:** {len(st.session_state.label_encoders)}")
                    
                    # Tampilkan contoh encoding
                    encoder_expander = st.expander("Lihat Detail Encoding")
                    with encoder_expander:
                        for col, le in list(st.session_state.label_encoders.items())[:3]:  # Tampilkan 3 pertama
                            st.write(f"**{col}:**")
                            encoding_dict = dict(zip(le.classes_, le.transform(le.classes_)))
                            for key, value in list(encoding_dict.items())[:5]:  # Tampilkan 5 pertama
                                st.write(f"  {key} → {value}")
                else:
                    st.info("Tidak ada variabel kategorikal yang perlu di-encode")
                
                st.subheader("Distribusi Target")
                target_counts = st.session_state.y.value_counts()
                fig_target, ax = plt.subplots(figsize=(8, 4))
                target_counts.plot(kind='bar', ax=ax, color='skyblue')
                ax.set_title('Distribusi Variabel Target')
                ax.set_xlabel('Kategori')
                ax.set_ylabel('Jumlah')
                ax.grid(axis='y', alpha=0.3)
                st.pyplot(fig_target)
        else:
            st.info("👈 Silakan muat dataset terlebih dahulu")
    
    # TAB 3: Analisis
    with tab3:
        st.header("Analisis Random Forest")
        
        if st.session_state.data_loaded:
            if run_analysis:
                with st.spinner("Melakukan analisis dengan Random Forest..."):
                    # Lakukan analisis
                    analysis_results = perform_random_forest_analysis(
                        st.session_state.X, 
                        st.session_state.y, 
                        test_size=test_size,
                        random_state=random_state,
                        n_estimators=n_estimators,
                        max_depth=max_depth
                    )
                    
                    # Simpan ke session state
                    st.session_state.analysis_results = analysis_results
                    
                    st.success("✅ Analisis selesai!")
            
            if st.session_state.analysis_results is not None:
                results = st.session_state.analysis_results
                
                # Tampilkan hasil utama
                st.subheader("Hasil Utama")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy Training", f"{results['accuracy_train']:.3%}")
                with col2:
                    st.metric("Accuracy Testing", f"{results['accuracy_test']:.3%}")
                with col3:
                    st.metric("Jumlah Fitur", st.session_state.X.shape[1])
                with col4:
                    st.metric("Estimator RF", n_estimators)
                
                # Feature Importance
                st.subheader("Feature Importance")
                st.write("**Top 15 Fitur Paling Penting:**")
                
                # Buat visualisasi feature importance
                fig_imp, ax_imp = plt.subplots(figsize=(12, 6))
                top_15 = results['feature_importance'].head(15)
                ax_imp.barh(range(len(top_15)), top_15['importance'][::-1])
                ax_imp.set_yticks(range(len(top_15)))
                ax_imp.set_yticklabels(top_15['feature'][::-1])
                ax_imp.set_xlabel('Importance Score')
                ax_imp.set_title('Top 15 Feature Importance')
                ax_imp.grid(axis='x', alpha=0.3)
                st.pyplot(fig_imp)
                
                # Tabel feature importance
                st.dataframe(results['feature_importance'].head(20), use_container_width=True)
                
                # Classification Report
                st.subheader("Classification Report")
                class_report_df = pd.DataFrame(results['classification_report']).transpose()
                st.dataframe(class_report_df, use_container_width=True)
            else:
                st.info("👈 Klik 'Jalankan Analisis' di sidebar untuk memulai")
        else:
            st.info("👈 Silakan muat dataset terlebih dahulu")
    
    # TAB 4: Evaluasi
    with tab4:
        st.header("Evaluasi Model")
        
        if st.session_state.data_loaded and st.session_state.analysis_results is not None:
            results = st.session_state.analysis_results
            
            # Plot hasil lengkap
            st.subheader("Visualisasi Hasil")
            fig = plot_results(results)
            st.pyplot(fig)
            
            # Detail confusion matrix
            st.subheader("Detail Confusion Matrix")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                cm_df = pd.DataFrame(
                    results['confusion_matrix'],
                    columns=[f'Pred_{i}' for i in range(results['confusion_matrix'].shape[1])],
                    index=[f'Actual_{i}' for i in range(results['confusion_matrix'].shape[0])]
                )
                st.dataframe(cm_df.style.background_gradient(cmap='Blues'), use_container_width=True)
            
            with col2:
                # Hitung metrics dari confusion matrix
                total = np.sum(results['confusion_matrix'])
                accuracy = np.trace(results['confusion_matrix']) / total
                
                st.metric("Overall Accuracy", f"{accuracy:.3%}")
                st.metric("Total Samples", total)
                st.metric("Classes", len(results['confusion_matrix']))
            
            # Interpretasi hasil
            st.subheader("Interpretasi Hasil")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **📊 Performa Model:**
                - Akurasi pada data training: **{:.3%}**
                - Akurasi pada data testing: **{:.3%}**
                - Selisih (gap): **{:.3%}**
                
                **🎯 Kesimpulan:**
                - Model menunjukkan akurasi **{}** pada data testing.
                - Gap yang kecil ({:.3%}) menunjukkan model tidak overfitting.
                - Fitur paling penting: **{}**
                """.format(
                    results['accuracy_train'],
                    results['accuracy_test'],
                    abs(results['accuracy_train'] - results['accuracy_test']),
                    "baik" if results['accuracy_test'] > 0.7 else "cukup",
                    abs(results['accuracy_train'] - results['accuracy_test']),
                    results['feature_importance'].iloc[0]['feature']
                ))
            
            with col2:
                st.markdown("""
                **💡 Rekomendasi:**
                1. Jika accuracy < 70%: Pertimbangkan untuk:
                   - Feature engineering tambahan
                   - Tuning hyperparameter lebih lanjut
                   - Mencoba algoritma lain
                
                2. Jika accuracy > 85%: Model sudah baik untuk:
                   - Prediksi performa akademik
                   - Identifikasi faktor penting
                   - Rekomendasi intervensi
                
                3. Aksi berdasarkan fitur penting:
                   - Fokus pada variabel yang paling berpengaruh
                   - Evaluasi kebijakan terkait variabel tersebut
                """)
            
            # Ekspor hasil
            st.subheader("Ekspor Hasil")
            
            if export_data:
                with st.spinner("Mengekspor hasil analisis..."):
                    export_folder = export_results_to_csv(
                        results,
                        st.session_state.label_encoders,
                        prefix="ai_analysis"
                    )
                    
                    st.success(f"✅ Hasil berhasil diekspor ke folder: `{export_folder}/`")
                    
                    # Tampilkan file yang diekspor
                    files = os.listdir(export_folder)
                    st.write("**File yang telah diekspor:**")
                    for file in files:
                        if file.startswith("ai_analysis"):
                            file_size = os.path.getsize(f'{export_folder}/{file}') / 1024
                            st.write(f"- `{file}` ({file_size:.1f} KB)")
                    
                    # Download link untuk file utama
                    st.markdown("---")
                    st.subheader("📥 Download Hasil Analisis")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    # Buat fungsi download untuk file
                    def create_download_link(file_path, label):
                        with open(file_path, 'rb') as f:
                            data = f.read()
                        b64 = base64.b64encode(data).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="{os.path.basename(file_path)}">{label}</a>'
                        return href
                    
                    try:
                        import base64
                        
                        with col1:
                            with open(f'{export_folder}/ai_analysis_predictions.csv', 'rb') as f:
                                st.download_button(
                                    label="📊 Download Predictions",
                                    data=f,
                                    file_name="predictions.csv",
                                    mime="text/csv"
                                )
                        
                        with col2:
                            with open(f'{export_folder}/ai_analysis_feature_importance.csv', 'rb') as f:
                                st.download_button(
                                    label="📈 Download Feature Importance",
                                    data=f,
                                    file_name="feature_importance.csv",
                                    mime="text/csv"
                                )
                        
                        with col3:
                            with open(f'{export_folder}/ai_analysis_classification_report.csv', 'rb') as f:
                                st.download_button(
                                    label="📋 Download Classification Report",
                                    data=f,
                                    file_name="classification_report.csv",
                                    mime="text/csv"
                                )
                    except:
                        st.info("Import base64 untuk download otomatis")
            
            else:
                st.info("Klik 'Ekspor Hasil' di sidebar untuk mengekspor semua hasil analisis")
        
        elif st.session_state.data_loaded:
            st.info("👈 Jalankan analisis terlebih dahulu di tab Analisis")
        else:
            st.info("👈 Silakan muat dataset terlebih dahulu")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>📚 <b>Analisis Tingkat Penggunaan AI terhadap Performa Akademik</b></p>
        <p>Menggunakan Algoritma Random Forest | Streamlit Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
