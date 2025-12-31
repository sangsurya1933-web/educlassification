import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine learning libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, mean_squared_error,
    r2_score, mean_absolute_error
)
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set halaman
st.set_page_config(
    page_title="Analisis Penggunaan AI - Performa Akademik",
    page_icon="🎓",
    layout="wide"
)

# Fungsi untuk membuat dataset contoh dengan variasi data
def create_sample_dataset():
    np.random.seed(42)
    
    # Buat data dengan pola yang lebih realistis
    n_samples = 200
    
    data = {
        'NIM': [f'202300{i:03d}' for i in range(1, n_samples + 1)],
        'Nama': [f'Mahasiswa_{i}' for i in range(1, n_samples + 1)],
        'Usia': np.random.randint(18, 25, n_samples),
        'Jenis_Kelamin': np.random.choice(['Laki-laki', 'Perempuan'], n_samples, p=[0.55, 0.45]),
        'Fakultas': np.random.choice(['Teknik', 'Sains', 'Ekonomi', 'Kedokteran', 'Hukum', 'Psikologi', 'Seni'], n_samples),
        'Semester': np.random.randint(1, 9, n_samples),
        'IPK': np.round(np.clip(np.random.normal(3.2, 0.5, n_samples), 2.0, 4.0), 2),
        'Jam_Belajar_Mingguan': np.random.randint(5, 50, n_samples),
        'Frekuensi_Penggunaan_AI': np.random.choice(['Sangat Rendah', 'Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi'], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1]),
        'Tujuan_Penggunaan_AI': np.random.choice(['Mengerjakan Tugas', 'Penelitian', 'Belajar Mandiri', 'Proyek Akhir', 'Ujian'], n_samples),
        'Tingkat_Ketergantungan_AI': np.random.choice(['Sangat Rendah', 'Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi'], n_samples),
        'Durasi_Penggunaan_AI_Jam': np.round(np.random.exponential(10, n_samples), 1),
        'Nilai_UTS': np.random.randint(50, 100, n_samples),
        'Nilai_UAS': np.random.randint(50, 100, n_samples),
        'Jumlah_Tugas_Tepat_Waktu': np.random.randint(0, 15, n_samples),
        'Status': np.random.choice(['Aktif', 'Cuti', 'Lulus'], n_samples, p=[0.85, 0.1, 0.05])
    }
    
    df = pd.DataFrame(data)
    
    # Tambahkan beberapa missing values untuk simulasi
    cols_with_nan = ['IPK', 'Jam_Belajar_Mingguan', 'Durasi_Penggunaan_AI_Jam']
    for col in cols_with_nan:
        idx = np.random.choice(df.index, size=int(n_samples*0.05), replace=False)
        df.loc[idx, col] = np.nan
    
    # Tambahkan outliers
    outlier_idx = np.random.choice(df.index, size=int(n_samples*0.03), replace=False)
    df.loc[outlier_idx, 'Jam_Belajar_Mingguan'] = np.random.randint(60, 100, len(outlier_idx))
    
    # Tambahkan data duplikat untuk contoh
    duplicate_idx = np.random.choice(df.index, size=int(n_samples*0.02), replace=False)
    for idx in duplicate_idx[:len(duplicate_idx)//2]:
        df = pd.concat([df, df.loc[[idx]]], ignore_index=True)
    
    return df

# Fungsi untuk preprocessing data lengkap
def comprehensive_preprocessing(df):
    st.header("📊 Preprocessing Data Lengkap")
    
    # Tampilkan data awal
    st.subheader("1. Data Awal")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(f"**Shape Data:** {df.shape}")
        st.write(f"**Jumlah Data:** {len(df)} baris, {len(df.columns)} kolom")
    with col2:
        st.write(f"**Tipe Data:**")
        st.write(df.dtypes.value_counts())
    
    # Pilih kolom untuk preprocessing
    st.subheader("2. Seleksi Kolom")
    
    all_columns = list(df.columns)
    selected_columns = st.multiselect(
        "Pilih kolom yang akan digunakan untuk analisis:",
        all_columns,
        default=[col for col in all_columns if col not in ['NIM', 'Nama', 'Status']]
    )
    
    if not selected_columns:
        st.warning("Silakan pilih minimal satu kolom!")
        return None, None, None, None
    
    df_selected = df[selected_columns].copy()
    
    # Tampilkan data terpilih
    with st.expander("Lihat Data Terpilih"):
        st.dataframe(df_selected.head())
    
    # Data Cleaning
    st.subheader("3. Data Cleaning")
    
    # Identifikasi masalah data
    col1, col2, col3 = st.columns(3)
    
    with col1:
        missing_values = df_selected.isnull().sum()
        missing_percentage = (missing_values / len(df_selected)) * 100
        st.metric("Missing Values", f"{missing_values.sum()} ({missing_percentage.sum():.1f}%)")
    
    with col2:
        duplicate_count = df_selected.duplicated().sum()
        st.metric("Data Duplikat", f"{duplicate_count}")
    
    with col3:
        numeric_cols = df_selected.select_dtypes(include=[np.number]).columns
        outlier_count = 0
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                Q1 = df_selected[col].quantile(0.25)
                Q3 = df_selected[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df_selected[col] < (Q1 - 1.5 * IQR)) | (df_selected[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_count += outliers
        st.metric("Outliers", f"{outlier_count}")
    
    # Opsi cleaning
    cleaning_options = st.multiselect(
        "Pilih operasi cleaning yang akan dilakukan:",
        ["Handle Missing Values", "Remove Duplicates", "Handle Outliers", "Normalize Text"]
    )
    
    df_clean = df_selected.copy()
    
    if "Handle Missing Values" in cleaning_options:
        st.write("**Handle Missing Values:**")
        missing_cols = missing_values[missing_values > 0].index.tolist()
        
        if missing_cols:
            for col in missing_cols:
                col_type = df_clean[col].dtype
                impute_method = st.selectbox(
                    f"Metode imputasi untuk {col}:",
                    ["Mean", "Median", "Mode", "Drop", "Custom Value"],
                    key=f"impute_{col}"
                )
                
                if impute_method == "Mean":
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                elif impute_method == "Median":
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                elif impute_method == "Mode":
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                elif impute_method == "Drop":
                    df_clean = df_clean.dropna(subset=[col])
                elif impute_method == "Custom Value":
                    custom_val = st.text_input(f"Nilai custom untuk {col}:", value="0", key=f"custom_{col}")
                    try:
                        if col_type in [np.float64, np.int64]:
                            df_clean[col].fillna(float(custom_val), inplace=True)
                        else:
                            df_clean[col].fillna(custom_val, inplace=True)
                    except:
                        df_clean[col].fillna(custom_val, inplace=True)
        else:
            st.write("Tidak ada missing values.")
    
    if "Remove Duplicates" in cleaning_options:
        st.write("**Remove Duplicates:**")
        before_len = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        after_len = len(df_clean)
        st.write(f"Dihapus {before_len - after_len} data duplikat.")
    
    if "Handle Outliers" in cleaning_options and len(numeric_cols) > 0:
        st.write("**Handle Outliers:**")
        outlier_method = st.selectbox(
            "Metode handling outliers:",
            ["IQR Method (Capping)", "Z-Score Method", "Remove Outliers", "Transformasi Log"]
        )
        
        for col in numeric_cols:
            if outlier_method == "IQR Method (Capping)":
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
                
            elif outlier_method == "Z-Score Method":
                from scipy import stats
                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                threshold = 3
                df_clean.loc[z_scores > threshold, col] = df_clean[col].mean()
                
            elif outlier_method == "Remove Outliers":
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                df_clean = df_clean[~((df_clean[col] < (Q1 - 1.5 * IQR)) | (df_clean[col] > (Q3 + 1.5 * IQR)))]
                
            elif outlier_method == "Transformasi Log":
                if (df_clean[col] > 0).all():
                    df_clean[col] = np.log1p(df_clean[col])
    
    if "Normalize Text" in cleaning_options:
        st.write("**Normalize Text:**")
        text_cols = df_clean.select_dtypes(include=['object']).columns
        for col in text_cols:
            df_clean[col] = df_clean[col].str.strip().str.title()
    
    # Encoding Data Kategorikal
    st.subheader("4. Encoding Data Kategorikal")
    
    categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        encoding_method = st.selectbox(
            "Pilih metode encoding:",
            ["Label Encoding", "One-Hot Encoding", "Target Encoding", "Frequency Encoding"]
        )
        
        df_encoded = df_clean.copy()
        encoders = {}
        
        if encoding_method == "Label Encoding":
            for col in categorical_cols:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                encoders[col] = le
                
        elif encoding_method == "One-Hot Encoding":
            df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, prefix=categorical_cols)
            
        elif encoding_method == "Frequency Encoding":
            for col in categorical_cols:
                freq = df_encoded[col].value_counts() / len(df_encoded)
                df_encoded[col] = df_encoded[col].map(freq)
                encoders[col] = freq.to_dict()
                
        elif encoding_method == "Target Encoding":
            st.warning("Target Encoding memerlukan target variable. Akan dilakukan setelah split data.")
            df_encoded = df_clean.copy()
    
        st.write("**Data setelah Encoding:**")
        st.dataframe(df_encoded.head())
        
        # Tampilkan informasi encoding
        with st.expander("Lihat Detail Encoding"):
            if encoding_method == "Label Encoding" and encoders:
                for col, le in encoders.items():
                    st.write(f"**{col}:**")
                    for i, class_name in enumerate(le.classes_):
                        st.write(f"  {class_name} → {i}")
    else:
        st.write("Tidak ada kolom kategorikal untuk di-encode.")
        df_encoded = df_clean.copy()
        encoders = {}
    
    # Feature Scaling
    st.subheader("5. Feature Scaling")
    
    scaling_method = st.selectbox(
        "Pilih metode scaling:",
        ["Tidak ada Scaling", "StandardScaler", "MinMaxScaler", "RobustScaler"]
    )
    
    df_scaled = df_encoded.copy()
    
    if scaling_method != "Tidak ada Scaling":
        numeric_cols_for_scaling = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols_for_scaling:
            if scaling_method == "StandardScaler":
                scaler = StandardScaler()
                df_scaled[numeric_cols_for_scaling] = scaler.fit_transform(df_scaled[numeric_cols_for_scaling])
                
            elif scaling_method == "MinMaxScaler":
                scaler = MinMaxScaler()
                df_scaled[numeric_cols_for_scaling] = scaler.fit_transform(df_scaled[numeric_cols_for_scaling])
                
            elif scaling_method == "RobustScaler":
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
                df_scaled[numeric_cols_for_scaling] = scaler.fit_transform(df_scaled[numeric_cols_for_scaling])
            
            st.write("**Data setelah Scaling:**")
            st.dataframe(df_scaled.head())
        else:
            st.write("Tidak ada kolom numerik untuk di-scaling.")
    else:
        scaler = None
    
    # Split Data
    st.subheader("6. Split Data")
    
    # Pilih target variable
    available_columns = df_scaled.columns.tolist()
    
    # Coba cari kolom target yang umum
    target_candidates = ['Frekuensi_Penggunaan_AI', 'Tingkat_Ketergantungan_AI', 'IPK', 'Nilai_UAS']
    target_default = None
    for candidate in target_candidates:
        if candidate in available_columns:
            target_default = candidate
            break
    
    if target_default is None and available_columns:
        target_default = available_columns[-1]
    
    target = st.selectbox(
        "Pilih target variable (variabel yang akan diprediksi):",
        available_columns,
        index=available_columns.index(target_default) if target_default in available_columns else 0
    )
    
    # Pilih fitur
    feature_options = [col for col in available_columns if col != target]
    default_features = [col for col in ['Usia', 'IPK', 'Jam_Belajar_Mingguan'] if col in feature_options]
    
    selected_features = st.multiselect(
        "Pilih fitur untuk model:",
        feature_options,
        default=default_features[:3] if default_features else feature_options[:3]
    )
    
    if not selected_features:
        st.error("Pilih minimal satu fitur!")
        return None, None, None, None, None, None
    
    # Konfigurasi split
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Ukuran data testing (%):", 10, 50, 20) / 100
    
    with col2:
        random_state = st.number_input("Random state:", min_value=0, max_value=100, value=42)
        shuffle = st.checkbox("Shuffle data", value=True)
    
    # Split data
    X = df_scaled[selected_features]
    y = df_scaled[target]
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            shuffle=shuffle,
            stratify=y if len(y.unique()) < 10 else None
        )
        
        st.success("✅ Data berhasil di-split!")
        
        # Tampilkan hasil split
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Data", len(df_scaled))
        with col2:
            st.metric("Training Data", len(X_train))
        with col3:
            st.metric("Testing Data", len(X_test))
        with col4:
            st.metric("Test Size", f"{test_size*100:.0f}%")
        
        # Visualisasi distribusi target
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Distribusi Target (Training)', 'Distribusi Target (Testing)'))
        
        y_train_counts = pd.Series(y_train).value_counts()
        fig.add_trace(
            go.Bar(x=y_train_counts.index.astype(str), y=y_train_counts.values, name='Training'),
            row=1, col=1
        )
        
        y_test_counts = pd.Series(y_test).value_counts()
        fig.add_trace(
            go.Bar(x=y_test_counts.index.astype(str), y=y_test_counts.values, name='Testing', marker_color='orange'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        return df_clean, df_scaled, X_train, X_test, y_train, y_test, target, selected_features, encoders
        
    except Exception as e:
        st.error(f"Error saat split data: {e}")
        return None, None, None, None, None, None, None, None, None

# Fungsi untuk training model Random Forest
def train_random_forest_model(X_train, X_test, y_train, y_test, target):
    st.header("🌳 Model Random Forest")
    
    # Deteksi jenis masalah (klasifikasi atau regresi)
    problem_type = st.selectbox(
        "Tipe masalah:",
        ["Auto Detect", "Klasifikasi", "Regresi"]
    )
    
    if problem_type == "Auto Detect":
        # Coba deteksi otomatis
        if y_train.dtype == 'object' or len(y_train.unique()) < 10:
            problem_type = "Klasifikasi"
        else:
            problem_type = "Regresi"
    
    st.write(f"**Tipe masalah terdeteksi:** {problem_type}")
    
    # Parameter model
    st.subheader("Hyperparameter Tuning")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_estimators = st.slider("Jumlah estimator (n_estimators):", 10, 500, 100, 10)
        max_depth = st.selectbox("Max depth:", ["None"] + list(range(5, 51, 5)), index=5)
        max_depth = None if max_depth == "None" else int(max_depth)
    
    with col2:
        min_samples_split = st.slider("Min samples split:", 2, 20, 2, 1)
        min_samples_leaf = st.slider("Min samples leaf:", 1, 10, 1, 1)
    
    with col3:
        max_features = st.selectbox("Max features:", ["auto", "sqrt", "log2", "None"], index=0)
        max_features = None if max_features == "None" else max_features
        bootstrap = st.checkbox("Bootstrap", value=True)
    
    # Opsi cross-validation
    use_cv = st.checkbox("Gunakan Cross-Validation", value=True)
    
    if use_cv:
        cv_folds = st.slider("Jumlah fold CV:", 3, 10, 5)
    
    # Training model
    if st.button("🚀 Train Model", type="primary"):
        try:
            with st.spinner("Training model Random Forest..."):
                # Inisialisasi model
                if problem_type == "Klasifikasi":
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        bootstrap=bootstrap,
                        random_state=42,
                        n_jobs=-1
                    )
                else:  # Regresi
                    model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        bootstrap=bootstrap,
                        random_state=42,
                        n_jobs=-1
                    )
                
                # Training
                model.fit(X_train, y_train)
                
                # Prediksi
                y_pred = model.predict(X_test)
                y_pred_train = model.predict(X_train)
                
                # Evaluasi model
                st.subheader("📈 Hasil Evaluasi Model")
                
                # Metrik evaluasi
                if problem_type == "Klasifikasi":
                    # Hitung berbagai metrik
                    accuracy_test = accuracy_score(y_test, y_pred)
                    accuracy_train = accuracy_score(y_train, y_pred_train)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    # Tampilkan metrik
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy Training", f"{accuracy_train:.2%}")
                    with col2:
                        st.metric("Accuracy Testing", f"{accuracy_test:.2%}")
                    with col3:
                        st.metric("Precision", f"{precision:.2%}")
                    with col4:
                        st.metric("Recall", f"{recall:.2%}")
                    
                    # Classification report
                    st.subheader("Classification Report")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.format("{:.2f}"))
                    
                    # Confusion matrix
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    
                    fig = px.imshow(
                        cm,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=[str(i) for i in np.unique(y_test)],
                        y=[str(i) for i in np.unique(y_test)],
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(width=600, height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:  # Regresi
                    # Hitung metrik regresi
                    mse_test = mean_squared_error(y_test, y_pred)
                    mse_train = mean_squared_error(y_train, y_pred_train)
                    rmse_test = np.sqrt(mse_test)
                    rmse_train = np.sqrt(mse_train)
                    mae_test = mean_absolute_error(y_test, y_pred)
                    mae_train = mean_absolute_error(y_train, y_pred_train)
                    r2_test = r2_score(y_test, y_pred)
                    r2_train = r2_score(y_train, y_pred_train)
                    
                    # Tampilkan metrik
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("RMSE Testing", f"{rmse_test:.4f}")
                        st.metric("RMSE Training", f"{rmse_train:.4f}")
                    with col2:
                        st.metric("MAE Testing", f"{mae_test:.4f}")
                        st.metric("MAE Training", f"{mae_train:.4f}")
                    with col3:
                        st.metric("R² Testing", f"{r2_test:.4f}")
                        st.metric("R² Training", f"{r2_train:.4f}")
                    
                    # Scatter plot actual vs predicted
                    st.subheader("Actual vs Predicted")
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=y_test, y=y_pred,
                        mode='markers',
                        name='Testing Data',
                        marker=dict(color='blue', size=8, opacity=0.6)
                    ))
                    
                    # Tambahkan garis diagonal
                    min_val = min(y_test.min(), y_pred.min())
                    max_val = max(y_test.max(), y_pred.max())
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val], y=[min_val, max_val],
                        mode='lines',
                        name='Ideal Fit',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig.update_layout(
                        xaxis_title='Actual Values',
                        yaxis_title='Predicted Values',
                        width=700, height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Cross-validation jika dipilih
                if use_cv:
                    st.subheader("📊 Cross-Validation Results")
                    
                    if problem_type == "Klasifikasi":
                        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
                        cv_metric = "Accuracy"
                    else:
                        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
                        cv_metric = "R² Score"
                    
                    cv_df = pd.DataFrame({
                        'Fold': range(1, cv_folds + 1),
                        cv_metric: cv_scores
                    })
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig = px.line(cv_df, x='Fold', y=cv_metric, markers=True,
                                     title=f'Cross-Validation Scores (Mean: {cv_scores.mean():.4f})')
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.write("**CV Statistics:**")
                        st.write(f"Mean: {cv_scores.mean():.4f}")
                        st.write(f"Std: {cv_scores.std():.4f}")
                        st.write(f"Min: {cv_scores.min():.4f}")
                        st.write(f"Max: {cv_scores.max():.4f}")
                
                # Feature importance
                st.subheader("🔍 Feature Importance")
                
                feature_importance = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                                color='Importance', color_continuous_scale='viridis')
                    fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.write("**Top 5 Features:**")
                    for idx, row in feature_importance.head().iterrows():
                        st.write(f"{row['Feature']}: {row['Importance']:.4f}")
                
                # Simpan model
                model_bytes = joblib.dump(model, 'random_forest_model.pkl')
                st.session_state.model = model
                st.session_state.y_pred = y_pred
                st.session_state.y_test = y_test
                st.session_state.problem_type = problem_type
                st.session_state.feature_importance = feature_importance
                
                st.success("✅ Model berhasil ditraining dan disimpan!")
                
                return model, y_pred
                
        except Exception as e:
            st.error(f"Error dalam training model: {str(e)}")
            return None, None
    
    return None, None

# Fungsi untuk evaluasi dan rekomendasi
def evaluation_and_recommendations(df_clean, target, predictions=None):
    st.header("📋 Evaluasi dan Rekomendasi")
    
    if 'model' not in st.session_state:
        st.warning("Silakan train model terlebih dahulu di menu 'Model Random Forest'")
        return
    
    # Analisis hasil
    st.subheader("1. Analisis Hasil Prediksi")
    
    # Tampilkan distribusi prediksi
    if predictions is not None:
        pred_series = pd.Series(predictions)
        pred_counts = pred_series.value_counts().sort_index()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.pie(values=pred_counts.values, names=pred_counts.index.astype(str),
                        title='Distribusi Prediksi')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Statistik Prediksi:**")
            st.write(f"Total: {len(predictions)}")
            st.write(f"Unique: {len(pred_counts)}")
    
    # Sistem rekomendasi berbasis knowledge
    st.subheader("2. Sistem Rekomendasi Berbasis Knowledge")
    
    # Knowledge base untuk rekomendasi
    knowledge_base = {
        'Frekuensi_Penggunaan_AI': {
            'Sangat Rendah': {
                'level': 'Aman',
                'kategori': 'Rendah',
                'rekomendasi': [
                    'Penggunaan AI masih sangat rendah dan dalam batas wajar',
                    'Pertahankan penggunaan AI sebagai alat bantu, bukan ketergantungan',
                    'Eksplorasi potensi AI untuk meningkatkan produktivitas akademik'
                ],
                'tindakan': 'Monitoring berkala'
            },
            'Rendah': {
                'level': 'Aman',
                'kategori': 'Rendah',
                'rekomendasi': [
                    'Penggunaan AI masih rendah dan terkendali',
                    'Pertahankan keseimbangan antara penggunaan AI dan kemampuan mandiri',
                    'Manfaatkan AI untuk tugas-tugas kompleks'
                ],
                'tindakan': 'Edukasi penggunaan optimal'
            },
            'Sedang': {
                'level': 'Perhatian',
                'kategori': 'Sedang',
                'rekomendasi': [
                    'Penggunaan AI mulai sering, perlu evaluasi',
                    'Pastikan tidak terjadi ketergantungan berlebihan',
                    'Kembangkan kemampuan analisis tanpa bantuan AI'
                ],
                'tindakan': 'Konsultasi dengan pembimbing'
            },
            'Tinggi': {
                'level': 'Waspada',
                'kategori': 'Tinggi',
                'rekomendasi': [
                    'Penggunaan AI sudah tinggi, perlu pembatasan',
                    'Evaluasi dampak terhadap kemampuan kritis mahasiswa',
                    'Implementasi kuota penggunaan AI'
                ],
                'tindakan': 'Pembatasan dan monitoring ketat'
            },
            'Sangat Tinggi': {
                'level': 'Kritis',
                'kategori': 'Sangat Tinggi',
                'rekomendasi': [
                    'Penggunaan AI sangat tinggi, butuh intervensi segera',
                    'Konsultasi dengan ahli pendidikan dan teknologi',
                    'Program rehabilitasi pengurangan ketergantungan AI'
                ],
                'tindakan': 'Intervensi intensif'
            }
        },
        'Tingkat_Ketergantungan_AI': {
            'Sangat Rendah': {
                'level': 'Sangat Baik',
                'rekomendasi': 'Pertahankan kemandirian belajar'
            },
            'Rendah': {
                'level': 'Baik',
                'rekomendasi': 'Manfaatkan AI sebagai pendukung, bukan ketergantungan'
            },
            'Sedang': {
                'level': 'Cukup',
                'rekomendasi': 'Kurangi ketergantungan secara bertahap'
            },
            'Tinggi': {
                'level': 'Kurang',
                'rekomendasi': 'Butuh program pengurangan ketergantungan'
            },
            'Sangat Tinggi': {
                'level': 'Buruk',
                'rekomendasi': 'Intervensi segera oleh pembimbing akademik'
            }
        }
    }
    
    # Pilih variabel untuk rekomendasi
    recommendation_target = st.selectbox(
        "Pilih variabel untuk rekomendasi:",
        ['Frekuensi_Penggunaan_AI', 'Tingkat_Ketergantungan_AI', 'IPK']
    )
    
    if recommendation_target in df_clean.columns:
        # Analisis distribusi
        value_counts = df_clean[recommendation_target].value_counts()
        
        st.write(f"**Distribusi {recommendation_target}:**")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(x=value_counts.index.astype(str), y=value_counts.values,
                        title=f'Distribusi {recommendation_target}')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Statistik:**")
            for value, count in value_counts.items():
                percentage = (count / len(df_clean)) * 100
                st.write(f"{value}: {count} ({percentage:.1f}%)")
        
        # Generate rekomendasi
        if recommendation_target in knowledge_base:
            st.subheader("3. Rekomendasi Berdasarkan Tingkat")
            
            for level, data in knowledge_base[recommendation_target].items():
                if level in value_counts.index:
                    count = value_counts[level]
                    percentage = (count / len(df_clean)) * 100
                    
                    with st.expander(f"📊 {level} ({count} mahasiswa, {percentage:.1f}%)"):
                        st.write(f"**Level:** {data['level']}")
                        st.write(f"**Kategori:** {data.get('kategori', 'N/A')}")
                        
                        if 'rekomendasi' in data:
                            if isinstance(data['rekomendasi'], list):
                                st.write("**Rekomendasi:**")
                                for rec in data['rekomendasi']:
                                    st.write(f"• {rec}")
                            else:
                                st.write(f"**Rekomendasi:** {data['rekomendasi']}")
                        
                        if 'tindakan' in data:
                            st.write(f"**Tindakan:** {data['tindakan']}")
        
        # Rekomendasi umum
        st.subheader("4. Rekomendasi Umum untuk Institusi")
        
        recommendations = [
            "1. **Implementasi Kebijakan AI**: Buat panduan penggunaan AI yang jelas untuk mahasiswa",
            "2. **Monitoring Berkala**: Lakukan pemantauan penggunaan AI melalui survei regular",
            "3. **Edukasi dan Pelatihan**: Berikan pelatihan penggunaan AI yang bertanggung jawab",
            "4. **Sistem Pendukung**: Sediakan konseling untuk mahasiswa dengan ketergantungan AI tinggi",
            "5. **Integrasi Kurikulum**: Masukkan etika penggunaan AI dalam kurikulum",
            "6. **Kolaborasi dengan Ahli**: Bekerjasama dengan ahli AI dan pendidikan",
            "7. **Penelitian Lanjutan**: Lakukan penelitian dampak AI terhadap pendidikan secara berkala"
        ]
        
        for rec in recommendations:
            st.write(rec)
    
    # Download hasil analisis
    st.subheader("5. Export Hasil Analisis")
    
    if st.button("📥 Download Report Analisis"):
        # Buat report sederhana
        report_data = {
            'Total Mahasiswa': len(df_clean),
            'Variabel Target': target,
            'Model Used': 'Random Forest',
            'Timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        report_df = pd.DataFrame([report_data])
        
        # Konversi ke CSV
        csv = report_df.to_csv(index=False)
        
        st.download_button(
            label="Download Report (CSV)",
            data=csv,
            file_name="ai_analysis_report.csv",
            mime="text/csv"
        )

# Halaman Login
def login_page():
    st.title("🎓 Sistem Analisis Penggunaan AI - Performa Akademik")
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://cdn.pixabay.com/photo/2020/05/18/16/17/social-media-5187243_1280.png", use_column_width=True)
    
    st.markdown("### Login ke Sistem")
    
    login_type = st.radio("Login sebagai:", ["Guru/Admin", "Mahasiswa"])
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login", type="primary"):
        if login_type == "Guru/Admin":
            if username == "guru" and password == "guru123":
                st.session_state.logged_in = True
                st.session_state.user_type = "guru"
                st.success("✅ Login berhasil! Mengarahkan ke dashboard...")
                st.rerun()
            else:
                st.error("❌ Username atau password salah untuk akun guru!")
        else:  # Mahasiswa
            if username and password == "mahasiswa123":
                st.session_state.logged_in = True
                st.session_state.user_type = "mahasiswa"
                st.session_state.student_name = username
                st.success(f"✅ Login berhasil! Selamat datang {username}")
                st.rerun()
            else:
                st.error("❌ Password harus 'mahasiswa123' untuk akun mahasiswa!")
    
    # Info login
    st.info("""
    **Credential Login:**
    - 👨‍🏫 **Guru/Admin**: Username: `guru`, Password: `guru123`
    - 👨‍🎓 **Mahasiswa**: Username: `Nama Anda`, Password: `mahasiswa123`
    
    **Catatan:** Aplikasi ini menggunakan dataset contoh untuk demonstrasi.
    """)

# Dashboard Guru
def guru_dashboard():
    st.sidebar.title("👨‍🏫 Dashboard Guru/Admin")
    st.sidebar.markdown("---")
    
    menu = st.sidebar.radio(
        "Menu Navigasi",
        ["📊 Data Awal", "🔧 Preprocessing Data", "🌳 Model Random Forest", 
         "📋 Evaluasi & Rekomendasi", "📈 Dashboard Analisis", "🚪 Logout"]
    )
    
    if menu == "🚪 Logout":
        st.session_state.logged_in = False
        st.session_state.user_type = None
        st.rerun()
    
    st.title("📊 Dashboard Analisis - Guru/Admin")
    st.markdown("---")
    
    # Inisialisasi session state
    if 'df' not in st.session_state:
        st.session_state.df = create_sample_dataset()
    
    if 'df_clean' not in st.session_state:
        st.session_state.df_clean = None
    
    if 'df_processed' not in st.session_state:
        st.session_state.df_processed = None
    
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    # Menu Data Awal
    if menu == "📊 Data Awal":
        st.header("📁 Data Awal Dataset")
        
        # Upload atau gunakan dataset contoh
        st.subheader("Upload Dataset")
        
        uploaded_file = st.file_uploader("Upload file CSV dataset", type=['csv'])
        
        if uploaded_file is not None:
            try:
                st.session_state.df = pd.read_csv(uploaded_file)
                st.success("✅ Dataset berhasil diupload!")
            except Exception as e:
                st.error(f"❌ Error membaca file: {e}")
                st.info("Menggunakan dataset contoh sebagai fallback.")
        else:
            st.info("📋 Menggunakan dataset contoh. Silakan upload dataset CSV jika ingin menggunakan data sendiri.")
        
        # Tampilkan data
        st.subheader("Preview Dataset")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Jumlah Data", len(st.session_state.df))
        with col2:
            st.metric("Jumlah Fitur", len(st.session_state.df.columns))
        with col3:
            st.metric("Memory Usage", f"{st.session_state.df.memory_usage().sum() / 1024:.1f} KB")
        
        # Tampilkan dataframe dengan tab
        tab1, tab2, tab3 = st.tabs(["Data", "Info", "Statistik"])
        
        with tab1:
            st.dataframe(st.session_state.df.head(20))
        
        with tab2:
            # Info data
            buffer = pd.io.common.StringIO()
            st.session_state.df.info(buf=buffer)
            info_str = buffer.getvalue()
            st.text(info_str)
        
        with tab3:
            # Statistik deskriptif
            st.write(st.session_state.df.describe())
        
        # Visualisasi data awal
        st.subheader("Visualisasi Data Awal")
        
        if st.checkbox("Tampilkan visualisasi"):
            numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                selected_num_col = st.selectbox("Pilih kolom numerik:", numeric_cols)
                
                fig = px.histogram(st.session_state.df, x=selected_num_col, 
                                 title=f'Distribusi {selected_num_col}')
                st.plotly_chart(fig, use_container_width=True)
            
            # Korelasi heatmap
            if len(numeric_cols) > 1:
                st.subheader("Heatmap Korelasi")
                corr_matrix = st.session_state.df[numeric_cols].corr()
                
                fig = px.imshow(corr_matrix, 
                              labels=dict(color="Korelasi"),
                              x=corr_matrix.columns,
                              y=corr_matrix.columns,
                              color_continuous_scale='RdBu')
                fig.update_layout(width=800, height=600)
                st.plotly_chart(fig, use_container_width=True)
    
    # Menu Preprocessing Data
    elif menu == "🔧 Preprocessing Data":
        st.header("🔧 Preprocessing Data Lengkap")
        
        if st.session_state.df is not None:
            results = comprehensive_preprocessing(st.session_state.df)
            
            if results[0] is not None:
                df_clean, df_processed, X_train, X_test, y_train, y_test, target, features, encoders = results
                
                # Simpan ke session state
                st.session_state.df_clean = df_clean
                st.session_state.df_processed = df_processed
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.target = target
                st.session_state.features = features
                st.session_state.encoders = encoders
                
                st.success("✅ Preprocessing data selesai!")
                
                # Tampilkan ringkasan
                with st.expander("📊 Ringkasan Preprocessing"):
                    st.write("**Data Cleaning:**")
                    st.write(f"- Missing values ditangani: Ya")
                    st.write(f"- Duplikat dihapus: Ya")
                    
                    st.write("**Encoding:**")
                    st.write(f"- Kolom kategorikal di-encode: Ya")
                    
                    st.write("**Split Data:**")
                    st.write(f"- Training data: {len(X_train)} sampel")
                    st.write(f"- Testing data: {len(X_test)} sampel")
                    st.write(f"- Fitur: {len(features)} variabel")
                    st.write(f"- Target: {target}")
        else:
            st.warning("Silakan upload atau gunakan dataset terlebih dahulu di menu 'Data Awal'")
    
    # Menu Model Random Forest
    elif menu == "🌳 Model Random Forest":
        if hasattr(st.session_state, 'X_train') and st.session_state.X_train is not None:
            model, predictions = train_random_forest_model(
                st.session_state.X_train,
                st.session_state.X_test,
                st.session_state.y_train,
                st.session_state.y_test,
                st.session_state.target
            )
            
            if model is not None:
                st.session_state.model = model
                st.session_state.predictions = predictions
        else:
            st.warning("Silakan lakukan preprocessing data terlebih dahulu di menu 'Preprocessing Data'")
    
    # Menu Evaluasi & Rekomendasi
    elif menu == "📋 Evaluasi & Rekomendasi":
        if st.session_state.df_clean is not None:
            evaluation_and_recommendations(
                st.session_state.df_clean,
                st.session_state.target if hasattr(st.session_state, 'target') else None,
                st.session_state.predictions if hasattr(st.session_state, 'predictions') else None
            )
        else:
            st.warning("Silakan lakukan preprocessing data terlebih dahulu")
    
    # Dashboard Analisis
    elif menu == "📈 Dashboard Analisis":
        st.header("📈 Dashboard Analisis Komprehensif")
        
        if st.session_state.df is not None:
            # Buat dashboard dengan metrik
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_ipk = st.session_state.df['IPK'].mean() if 'IPK' in st.session_state.df.columns else 0
                st.metric("Rata-rata IPK", f"{avg_ipk:.2f}")
            
            with col2:
                ai_usage = st.session_state.df['Frekuensi_Penggunaan_AI'].value_counts().index[0] if 'Frekuensi_Penggunaan_AI' in st.session_state.df.columns else "N/A"
                st.metric("Frekuensi AI Terbanyak", ai_usage)
            
            with col3:
                study_hours = st.session_state.df['Jam_Belajar_Mingguan'].mean() if 'Jam_Belajar_Mingguan' in st.session_state.df.columns else 0
                st.metric("Rata-rata Jam Belajar", f"{study_hours:.1f} jam/minggu")
            
            with col4:
                total_students = len(st.session_state.df)
                st.metric("Total Mahasiswa", total_students)
            
            # Visualisasi interaktif
            st.subheader("Analisis Hubungan Penggunaan AI dan Performa")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Frekuensi_Penggunaan_AI' in st.session_state.df.columns and 'IPK' in st.session_state.df.columns:
                    fig = px.box(st.session_state.df, x='Frekuensi_Penggunaan_AI', y='IPK',
                               title='Distribusi IPK berdasarkan Frekuensi Penggunaan AI')
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'Tingkat_Ketergantungan_AI' in st.session_state.df.columns and 'Nilai_UAS' in st.session_state.df.columns:
                    fig = px.scatter(st.session_state.df, x='Tingkat_Ketergantungan_AI', y='Nilai_UAS',
                                   color='Semester' if 'Semester' in st.session_state.df.columns else None,
                                   title='Hubungan Ketergantungan AI dan Nilai UAS')
                    st.plotly_chart(fig, use_container_width=True)
            
            # Analisis fakultas
            if 'Fakultas' in st.session_state.df.columns:
                st.subheader("Analisis per Fakultas")
                
                faculty_stats = st.session_state.df.groupby('Fakultas').agg({
                    'IPK': 'mean',
                    'Jam_Belajar_Mingguan': 'mean',
                    'Frekuensi_Penggunaan_AI': lambda x: x.mode()[0] if not x.mode().empty else 'N/A'
                }).round(2)
                
                st.dataframe(faculty_stats)
        else:
            st.warning("Silakan upload dataset terlebih dahulu")

# Dashboard Mahasiswa
def mahasiswa_dashboard():
    st.sidebar.title("👨‍🎓 Dashboard Mahasiswa")
    st.sidebar.markdown("---")
    
    # Info mahasiswa
    if 'student_name' in st.session_state:
        st.sidebar.write(f"👤 **Nama:** {st.session_state.student_name}")
    
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.user_type = None
        st.session_state.student_name = None
        st.rerun()
    
    st.title("📋 Hasil Analisis Penggunaan AI")
    st.markdown("---")
    
    # Simulasi data mahasiswa
    if 'student_name' in st.session_state:
        student_name = st.session_state.student_name
        
        # Generate data acak untuk mahasiswa
        np.random.seed(hash(student_name) % 1000)
        
        student_data = {
            'Nama': student_name,
            'NIM': f'2023{np.random.randint(10000, 99999)}',
            'Fakultas': np.random.choice(['Teknik', 'Sains', 'Ekonomi', 'Kedokteran', 'Hukum']),
            'Semester': np.random.randint(1, 9),
            'IPK': np.round(np.random.uniform(2.5, 4.0), 2),
            'Frekuensi_Penggunaan_AI': np.random.choice(['Sangat Rendah', 'Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi'], p=[0.1, 0.2, 0.4, 0.2, 0.1]),
            'Tingkat_Ketergantungan_AI': np.random.choice(['Sangat Rendah', 'Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi']),
            'Durasi_Penggunaan_AI': f"{np.random.randint(1, 30)} jam/minggu",
            'Rata-rata_Nilai': np.random.randint(65, 95)
        }
        
        # Tampilkan data mahasiswa
        st.subheader(f"Profil Mahasiswa: {student_data['Nama']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("NIM", student_data['NIM'])
            st.metric("Fakultas", student_data['Fakultas'])
            st.metric("Semester", student_data['Semester'])
        
        with col2:
            st.metric("IPK", student_data['IPK'])
            st.metric("Rata-rata Nilai", student_data['Rata-rata_Nilai'])
            st.metric("Frekuensi Penggunaan AI", student_data['Frekuensi_Penggunaan_AI'])
        
        # Analisis dan klasifikasi
        st.subheader("🎯 Klasifikasi Tingkat Penggunaan AI")
        
        # Tentukan klasifikasi berdasarkan frekuensi penggunaan AI
        ai_frequency = student_data['Frekuensi_Penggunaan_AI']
        ai_dependency = student_data['Tingkat_Ketergantungan_AI']
        
        # Mapping klasifikasi
        classification_map = {
            'Sangat Rendah': {'level': 'Aman', 'color': 'green'},
            'Rendah': {'level': 'Aman', 'color': 'green'},
            'Sedang': {'level': 'Perhatian', 'color': 'orange'},
            'Tinggi': {'level': 'Waspada', 'color': 'red'},
            'Sangat Tinggi': {'level': 'Kritis', 'color': 'darkred'}
        }
        
        classification = classification_map.get(ai_frequency, {'level': 'Tidak Diketahui', 'color': 'gray'})
        
        # Tampilkan klasifikasi
        st.markdown(f"""
        <div style='background-color:{classification['color']}20; padding:20px; border-radius:10px; border-left:5px solid {classification['color']}'>
            <h3 style='color:{classification['color']}; margin-top:0;'>🏷️ Klasifikasi: {classification['level'].upper()}</h3>
            <p><strong>Frekuensi Penggunaan AI:</strong> {ai_frequency}</p>
            <p><strong>Tingkat Ketergantungan AI:</strong> {ai_dependency}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Rekomendasi personal
        st.subheader("💡 Rekomendasi Personal")
        
        recommendations = {
            'Sangat Rendah': [
                "✅ Penggunaan AI Anda masih sangat rendah",
                "✅ Pertahankan penggunaan yang sehat",
                "💡 Coba eksplorasi tools AI untuk meningkatkan produktivitas"
            ],
            'Rendah': [
                "✅ Penggunaan AI Anda dalam batas wajar",
                "✅ Pertahankan keseimbangan",
                "💡 Manfaatkan AI untuk tugas-tugas kompleks"
            ],
            'Sedang': [
                "⚠️ Penggunaan AI Anda mulai sering",
                "⚠️ Evaluasi kebutuhan penggunaan AI",
                "💡 Kembangkan kemampuan analisis mandiri"
            ],
            'Tinggi': [
                "🚨 Penggunaan AI Anda tinggi",
                "🚨 Kurangi ketergantungan pada AI",
                "💡 Konsultasi dengan dosen pembimbing"
            ],
            'Sangat Tinggi': [
                "🚨🚨 Penggunaan AI Anda sangat tinggi",
                "🚨🚨 Butuh intervensi segera",
                "💡 Program khusus pengurangan ketergantungan AI"
            ]
        }
        
        student_recommendations = recommendations.get(ai_frequency, [
            "Tidak ada rekomendasi spesifik",
            "Konsultasikan dengan dosen pembimbing"
        ])
        
        for rec in student_recommendations:
            st.write(f"• {rec}")
        
        # Visualisasi perbandingan
        st.subheader("📊 Perbandingan dengan Rata-rata Kelas")
        
        # Data contoh rata-rata kelas
        class_avg = {
            'IPK': 3.25,
            'Frekuensi_Penggunaan_AI': 'Sedang',
            'Jam_Belajar': 25,
            'Ketergantungan_AI': 'Sedang'
        }
        
        fig = go.Figure()
        
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=student_data['IPK'],
            delta={'reference': class_avg['IPK']},
            title={'text': "IPK"},
            domain={'row': 0, 'column': 0},
            gauge={'axis': {'range': [2.0, 4.0]}}
        ))
        
        fig.update_layout(
            grid={'rows': 1, 'columns': 1, 'pattern': "independent"},
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tips penggunaan AI yang sehat
        st.subheader("🌟 Tips Penggunaan AI yang Sehat")
        
        tips = [
            "1. **Gunakan AI sebagai alat bantu**, bukan pengganti pemikiran",
            "2. **Verifikasi hasil AI** dengan sumber terpercaya",
            "3. **Kembangkan kemampuan kritis** tanpa bergantung pada AI",
            "4. **Batasi waktu penggunaan** AI untuk tugas akademik",
            "5. **Diskusikan dengan dosen** tentang penggunaan AI yang tepat",
            "6. **Pelajari konsep dasar** sebelum menggunakan AI untuk solusi",
            "7. **Jaga keseimbangan** antara teknologi dan kemampuan manusiawi"
        ]
        
        for tip in tips:
            st.write(tip)
    else:
        st.warning("Data mahasiswa tidak ditemukan.")

# Aplikasi utama
def main():
    # Inisialisasi session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user_type = None
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton button {
        background-color: #2E86AB;
        color: white;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #1B657D;
    }
    .metric-card {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
    }
    </style>
    """, unsafe_allow_html=True)
    
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
