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
import hashlib
import csv
import os

# ==================== FUNGSI UTILITAS ====================
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return True
    return False

    }
    return users

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
def perform_random_forest_analysis(X, y, test_size=0.3, random_state=42):
    """Melakukan analisis dengan Random Forest"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Inisialisasi dan train model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
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
def export_results_to_csv(analysis_results, label_encoders, prefix='analysis'):
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
    
    # 7. Ekspor label encoders mapping
    if label_encoders:
        encoder_mappings = {}
        for col, le in label_encoders.items():
            encoder_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
        
        # Simpan sebagai JSON-like CSV
        encoder_df = pd.DataFrame([encoder_mappings])
        encoder_df.to_csv(f'{export_folder}/{prefix}_encoder_mappings.csv', index=False)
    
    return export_folder

# ==================== DASHBOARD STREAMLIT ====================
def main():
    st.set_page_config(
        page_title="AI Usage Analysis - Random Forest",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🎓 Analisis Tingkat Penggunaan AI terhadap Performa Akademik")
    st.markdown("---")
    
    # Inisialisasi session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_type' not in st.session_state:
        st.session_state.user_type = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Sidebar untuk login
    with st.sidebar:
        st.header("🔐 Login")
        
        if not st.session_state.logged_in:
            user_type = st.selectbox("Pilih jenis pengguna", ["", "Guru", "Siswa"])
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Login"):
                users_db = create_user_db()
                
                if user_type == "Guru" and username == "guru" and check_hashes(password, users_db['guru']):
                    st.session_state.logged_in = True
                    st.session_state.user_type = "Guru"
                    st.success("Login berhasil sebagai Guru!")
                    st.rerun()
                elif user_type == "Siswa" and username == "siswa" and check_hashes(password, users_db['siswa']):
                    st.session_state.logged_in = True
                    st.session_state.user_type = "Siswa"
                    st.success("Login berhasil sebagai Siswa!")
                    st.rerun()
                else:
                    st.error("Username atau password salah!")
        
        if st.session_state.logged_in:
            st.success(f"Login sebagai: {st.session_state.user_type}")
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.user_type = None
                st.session_state.analysis_results = None
                st.session_state.data_loaded = False
                st.rerun()
    
    # Jika belum login, tampilkan instruksi
    if not st.session_state.logged_in:
        st.info("Silakan login terlebih dahulu menggunakan sidebar")
        st.markdown("""
        ### Credentials Login:
        - **Guru**: username=`guru`, password=`guru123`
        - **Siswa**: username=`siswa`, password=`siswa123`
        
        ### Dataset Information:
        Analisis ini menggunakan algoritma Random Forest untuk mengklasifikasikan
        dampak penggunaan AI terhadap performa akademik siswa.
        """)
        return
    
    # ==================== MENU UTAMA ====================
    st.header(f"Dashboard {st.session_state.user_type}")
    
    # Upload dataset (hanya untuk Guru)
    if st.session_state.user_type == "Guru":
        uploaded_file = st.file_uploader("📁 Upload dataset CSV", type=['csv'])
        
        if uploaded_file is not None:
            # Simpan file sementara
            with open("temp_dataset.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load dan process data
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
                    
                    # Tampilkan preview data
                    with st.expander("👁️ Preview Dataset"):
                        st.write("**Data Asli:**")
                        st.dataframe(df.head())
                        st.write(f"**Shape:** {df.shape}")
                        st.write(f"**Kolom Target:** {target_col}")
                        
                        st.write("**Data Setelah Preprocessing:**")
                        st.dataframe(X.head())
                        
                        st.write("**Distribusi Target:**")
                        st.bar_chart(y.value_counts())
    
    # Jika data sudah dimuat atau untuk siswa (gunakan dataset default)
    if not st.session_state.data_loaded and st.session_state.user_type == "Siswa":
        # Load dataset default untuk siswa
        st.info("Menggunakan dataset default untuk analisis")
        # Di sini Anda bisa load dataset default
        
    # Tab menu berdasarkan user type
    if st.session_state.user_type == "Guru":
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Preprocessing", "🤖 Analisis", "📈 Evaluasi", "💾 Ekspor"])
    else:
        tab1, tab2 = st.tabs(["🤖 Analisis", "📈 Visualisasi"])
    
    # ==================== TAB PREPROCESSING (GURU ONLY) ====================
    if st.session_state.user_type == "Guru" and st.session_state.data_loaded:
        with tab1:
            st.subheader("Data Preprocessing")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Informasi Dataset:**")
                st.write(f"- Jumlah sampel: {st.session_state.df.shape[0]}")
                st.write(f"- Jumlah fitur: {st.session_state.df.shape[1] - 1}")
                st.write(f"- Kolom target: {st.session_state.target_col}")
                
                st.write("**Tipe Data:**")
                dtypes_df = pd.DataFrame(st.session_state.df.dtypes, columns=['Data Type'])
                st.dataframe(dtypes_df)
            
            with col2:
                st.write("**Statistik Deskriptif:**")
                st.dataframe(st.session_state.df.describe())
            
            st.write("**Distribusi Variabel Target:**")
            target_counts = st.session_state.y.value_counts()
            fig_target, ax = plt.subplots(figsize=(8, 4))
            target_counts.plot(kind='bar', ax=ax)
            ax.set_title('Distribusi Variabel Target')
            ax.set_xlabel('Kategori')
            ax.set_ylabel('Jumlah')
            st.pyplot(fig_target)
    
    # ==================== TAB ANALISIS ====================
    if (st.session_state.user_type == "Guru" and st.session_state.data_loaded) or st.session_state.user_type == "Siswa":
        analysis_tab = tab2 if st.session_state.user_type == "Siswa" else tab2
        
        with analysis_tab:
            st.subheader("Analisis dengan Random Forest")
            
            if st.session_state.user_type == "Guru" and not st.session_state.data_loaded:
                st.warning("Silakan upload dataset terlebih dahulu")
            else:
                # Parameter untuk model (hanya guru yang bisa mengubah)
                if st.session_state.user_type == "Guru":
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_estimators = st.slider("Jumlah Estimator", 10, 500, 100)
                    with col2:
                        max_depth = st.slider("Max Depth", 2, 50, 10)
                    with col3:
                        test_size = st.slider("Test Size (%)", 10, 50, 30) / 100
                else:
                    # Parameter default untuk siswa
                    n_estimators = 100
                    max_depth = 10
                    test_size = 0.3
                
                if st.button("🚀 Jalankan Analisis", type="primary"):
                    with st.spinner("Melakukan analisis dengan Random Forest..."):
                        # Gunakan data yang sudah diproses
                        if st.session_state.user_type == "Guru":
                            X = st.session_state.X
                            y = st.session_state.y
                        else:
                            # Untuk siswa, load dataset default
                            # Di sini Anda bisa menambahkan dataset default
                            st.info("Menggunakan dataset contoh untuk analisis")
                            # Buat dataset contoh
                            from sklearn.datasets import make_classification
                            X, y = make_classification(
                                n_samples=1000, n_features=20, n_classes=3,
                                n_informative=10, random_state=42
                            )
                            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
                            y = pd.Series(y, name='target')
                        
                        # Lakukan analisis
                        analysis_results = perform_random_forest_analysis(
                            X, y, test_size=test_size
                        )
                        
                        # Simpan ke session state
                        st.session_state.analysis_results = analysis_results
                        
                        st.success("✅ Analisis selesai!")
                        
                        # Tampilkan hasil singkat
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Accuracy Training", f"{analysis_results['accuracy_train']:.3f}")
                        with col2:
                            st.metric("Accuracy Testing", f"{analysis_results['accuracy_test']:.3f}")
                        with col3:
                            st.metric("Jumlah Fitur", X.shape[1])
                
                # Tampilkan hasil jika sudah ada
                if st.session_state.analysis_results is not None:
                    results = st.session_state.analysis_results
                    
                    # Feature Importance
                    st.subheader("Feature Importance")
                    st.dataframe(results['feature_importance'].head(15))
                    
                    # Classification Report
                    st.subheader("Classification Report")
                    class_report_df = pd.DataFrame(results['classification_report']).transpose()
                    st.dataframe(class_report_df)
    
    # ==================== TAB EVALUASI (GURU ONLY) ====================
    if st.session_state.user_type == "Guru" and st.session_state.analysis_results is not None:
        with tab3:
            st.subheader("Evaluasi Model")
            
            results = st.session_state.analysis_results
            
            # Plot hasil
            fig = plot_results(results)
            st.pyplot(fig)
            
            # Detail confusion matrix
            st.subheader("Detail Confusion Matrix")
            cm_df = pd.DataFrame(
                results['confusion_matrix'],
                columns=[f'Pred_{i}' for i in range(results['confusion_matrix'].shape[1])],
                index=[f'Actual_{i}' for i in range(results['confusion_matrix'].shape[0])]
            )
            st.dataframe(cm_df)
            
            # Interpretasi hasil
            st.subheader("Interpretasi Hasil")
            st.write(f"""
            **Akurasi Model:**
            - Akurasi pada data training: **{results['accuracy_train']:.3f}**
            - Akurasi pada data testing: **{results['accuracy_test']:.3f}**
            
            **Kesimpulan:**
            - Model Random Forest berhasil mencapai akurasi sebesar **{results['accuracy_test']:.3f}** pada data testing.
            - Selisih antara akurasi training dan testing: **{abs(results['accuracy_train'] - results['accuracy_test']):.3f}**
            - Fitur paling penting dalam prediksi: **{results['feature_importance'].iloc[0]['feature']}**
            """)
    
    # ==================== TAB EKSPOR (GURU ONLY) ====================
    if st.session_state.user_type == "Guru":
        export_tab = tab4 if st.session_state.user_type == "Guru" else None
        
        if export_tab:
            with export_tab:
                st.subheader("Ekspor Hasil Analisis")
                
                if st.session_state.analysis_results is not None:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        export_prefix = st.text_input("Nama file prefix", "ai_analysis")
                    
                    with col2:
                        st.write("")
                        st.write("")
                        if st.button("📥 Ekspor Semua Hasil ke CSV", type="primary"):
                            with st.spinner("Mengekspor hasil..."):
                                export_folder = export_results_to_csv(
                                    st.session_state.analysis_results,
                                    st.session_state.label_encoders if 'label_encoders' in st.session_state else None,
                                    prefix=export_prefix
                                )
                                
                                st.success(f"✅ Hasil berhasil diekspor ke folder: {export_folder}/")
                                
                                # Tampilkan list file yang diekspor
                                files = os.listdir(export_folder)
                                st.write("**File yang telah diekspor:**")
                                for file in files:
                                    if file.startswith(export_prefix):
                                        st.write(f"- {file}")
                    
                    # Preview file yang akan diekspor
                    st.subheader("Preview Data untuk Ekspor")
                    
                    if st.session_state.analysis_results is not None:
                        preview_option = st.selectbox(
                            "Pilih data untuk preview",
                            ["Feature Importance", "Classification Report", "Predictions", "Confusion Matrix"]
                        )
                        
                        if preview_option == "Feature Importance":
                            st.dataframe(st.session_state.analysis_results['feature_importance'])
                        elif preview_option == "Classification Report":
                            class_report_df = pd.DataFrame(
                                st.session_state.analysis_results['classification_report']
                            ).transpose()
                            st.dataframe(class_report_df)
                        elif preview_option == "Predictions":
                            predictions_df = pd.DataFrame({
                                'Actual': st.session_state.analysis_results['y_test'].values,
                                'Predicted': st.session_state.analysis_state.analysis_results['y_pred']
                            })
                            st.dataframe(predictions_df.head(20))
                        elif preview_option == "Confusion Matrix":
                            cm_df = pd.DataFrame(st.session_state.analysis_results['confusion_matrix'])
                            st.dataframe(cm_df)
                else:
                    st.info("Lakukan analisis terlebih dahulu untuk mengekspor hasil")
    
    # ==================== TAB VISUALISASI (SISWA ONLY) ====================
    if st.session_state.user_type == "Siswa" and st.session_state.analysis_results is not None:
        with tab2:  # Tab Visualisasi untuk siswa
            st.subheader("Visualisasi Hasil Analisis")
            
            if st.session_state.analysis_results is not None:
                results = st.session_state.analysis_results
                
                # Plot sederhana untuk siswa
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Feature Importance (Top 10)**")
                    top_features = results['feature_importance'].head(10)
                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    ax1.barh(range(len(top_features)), top_features['importance'])
                    ax1.set_yticks(range(len(top_features)))
                    ax1.set_yticklabels(top_features['feature'])
                    ax1.set_xlabel('Importance')
                    ax1.set_title('Top 10 Most Important Features')
                    st.pyplot(fig1)
                
                with col2:
                    st.write("**Model Accuracy**")
                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                    labels = ['Training', 'Testing']
                    values = [results['accuracy_train'], results['accuracy_test']]
                    bars = ax2.bar(labels, values, color=['blue', 'green'])
                    ax2.set_ylabel('Accuracy')
                    ax2.set_title('Model Performance')
                    ax2.set_ylim(0, 1)
                    
                    # Tambahkan nilai di atas bar
                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                                f'{value:.3f}', ha='center', va='bottom')
                    
                    st.pyplot(fig2)
                
                # Confusion Matrix
                st.write("**Confusion Matrix**")
                fig3, ax3 = plt.subplots(figsize=(8, 6))
                sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax3)
                ax3.set_title('Confusion Matrix')
                ax3.set_xlabel('Predicted Label')
                ax3.set_ylabel('True Label')
                st.pyplot(fig3)

if __name__ == "__main__":
    main()
