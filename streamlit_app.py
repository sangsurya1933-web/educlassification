import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, mean_squared_error, mean_absolute_error, 
                           confusion_matrix, classification_report)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import os

class PreprocessingModule:
    def __init__(self, data):
        self.data = data.copy()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def remove_missing(self):
        """Remove missing values"""
        before = len(self.data)
        self.data = self.data.dropna()
        after = len(self.data)
        return f"Removed {before - after} rows with missing values"
    
    def fill_missing(self):
        """Fill missing values with median/mode"""
        for col in self.data.columns:
            if self.data[col].dtype in ['int64', 'float64']:
                self.data[col].fillna(self.data[col].median(), inplace=True)
            else:
                self.data[col].fillna(self.data[col].mode()[0], inplace=True)
        return "Missing values filled with median/mode"
    
    def remove_duplicates(self):
        """Remove duplicate rows"""
        before = len(self.data)
        self.data = self.data.drop_duplicates()
        after = len(self.data)
        return f"Removed {before - after} duplicate rows"
    
    def detect_outliers_iqr(self):
        """Detect outliers using IQR method"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        outliers_info = {}
        
        for col in numeric_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]
            outliers_info[col] = len(outliers)
        
        return outliers_info
    
    def remove_outliers_iqr(self):
        """Remove outliers using IQR method"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        before = len(self.data)
        
        for col in numeric_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            self.data = self.data[(self.data[col] >= lower_bound) & (self.data[col] <= upper_bound)]
        
        after = len(self.data)
        return f"Removed {before - after} outliers"
    
    def label_encode_column(self, column):
        """Apply label encoding to a column"""
        if column in self.data.columns:
            le = LabelEncoder()
            self.data[column] = le.fit_transform(self.data[column].astype(str))
            self.label_encoders[column] = le
            return f"Label encoding applied to {column}"
        return f"Column {column} not found"
    
    def onehot_encode_column(self, column):
        """Apply one-hot encoding to a column"""
        if column in self.data.columns:
            dummies = pd.get_dummies(self.data[column], prefix=column)
            self.data = pd.concat([self.data, dummies], axis=1)
            self.data = self.data.drop(column, axis=1)
            return f"One-hot encoding applied to {column}"
        return f"Column {column} not found"
    
    def standard_scale_column(self, column):
        """Apply standard scaling to a column"""
        if column in self.data.columns and self.data[column].dtype in ['int64', 'float64']:
            scaler = StandardScaler()
            self.data[column] = scaler.fit_transform(self.data[[column]])
            return f"Standard scaling applied to {column}"
        return f"Column {column} is not numeric"
    
    def minmax_scale_column(self, column):
        """Apply min-max scaling to a column"""
        if column in self.data.columns and self.data[column].dtype in ['int64', 'float64']:
            scaler = MinMaxScaler()
            self.data[column] = scaler.fit_transform(self.data[[column]])
            return f"Min-Max scaling applied to {column}"
        return f"Column {column} is not numeric"
    
    def apply_pca(self, n_components):
        """Apply PCA for dimensionality reduction"""
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            return "Need at least 2 numeric columns for PCA"
        
        if n_components >= len(numeric_data.columns):
            return "Number of components must be less than number of features"
        
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(numeric_data)
        
        # Create new column names
        pca_cols = [f'PCA_{i+1}' for i in range(n_components)]
        
        # Create DataFrame with PCA results
        pca_df = pd.DataFrame(pca_result, columns=pca_cols)
        
        # Drop original numeric columns and add PCA columns
        self.data = self.data.drop(numeric_data.columns, axis=1)
        self.data = pd.concat([self.data, pca_df], axis=1)
        
        return f"PCA applied: {len(numeric_data.columns)} features → {n_components} components"
    
    def split_data(self, target_column, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        if target_column not in self.data.columns:
            return None, None, None, None, "Target column not found"
        
        X = self.data.drop(target_column, axis=1)
        y = self.data[target_column]
        
        # Encode target if categorical
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y if len(set(y)) < 10 else None
        )
        
        return X_train, X_test, y_train, y_test, "Data split completed"
    
    def get_data_info(self):
        """Get information about the dataset"""
        info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': str(self.data.dtypes.to_dict()),
            'missing_values': self.data.isnull().sum().to_dict(),
            'numeric_columns': list(self.data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.data.select_dtypes(include=['object']).columns)
        }
        return info

class RandomForestModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.problem_type = None
        self.metrics = {}
        
    def train(self, X_train, y_train, n_estimators=100, max_depth=None, random_state=42):
        """Train Random Forest model"""
        # Determine problem type
        unique_classes = len(np.unique(y_train))
        
        # Scale features
        X_train_scaled = X_train.copy()
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            X_train_scaled[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
        
        if unique_classes < 10:  # Classification
            self.problem_type = "classification"
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
        else:  # Regression
            self.problem_type = "regression"
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
        
        self.model.fit(X_train_scaled, y_train)
        return f"Model trained ({self.problem_type}) with {len(X_train)} samples"
    
    def predict(self, X_test):
        """Make predictions"""
        if self.model is None:
            return None, "Model not trained"
        
        X_test_scaled = X_test.copy()
        numeric_cols = X_test.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            X_test_scaled[numeric_cols] = self.scaler.transform(X_test[numeric_cols])
        
        predictions = self.model.predict(X_test_scaled)
        return predictions, "Predictions made"
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        if self.model is None:
            return "Model not trained"
        
        predictions, _ = self.predict(X_test)
        
        if self.problem_type == "classification":
            # Classification metrics
            self.metrics['accuracy'] = accuracy_score(y_test, predictions)
            self.metrics['precision'] = precision_score(y_test, predictions, average='weighted', zero_division=0)
            self.metrics['recall'] = recall_score(y_test, predictions, average='weighted', zero_division=0)
            self.metrics['f1_score'] = f1_score(y_test, predictions, average='weighted', zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, predictions)
            
            report = classification_report(y_test, predictions, output_dict=True)
            
            return {
                'metrics': self.metrics,
                'confusion_matrix': cm.tolist(),
                'classification_report': report,
                'predictions': predictions.tolist()
            }
        else:
            # Regression metrics
            self.metrics['mse'] = mean_squared_error(y_test, predictions)
            self.metrics['mae'] = mean_absolute_error(y_test, predictions)
            self.metrics['rmse'] = np.sqrt(self.metrics['mse'])
            
            residuals = y_test - predictions
            
            return {
                'metrics': self.metrics,
                'residuals': residuals.tolist(),
                'predictions': predictions.tolist()
            }
    
    def get_feature_importance(self, feature_names):
        """Get feature importance from model"""
        if self.model is None:
            return None
        
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def predict_single(self, features):
        """Predict for a single sample"""
        if self.model is None:
            return None, "Model not trained"
        
        # Convert features to DataFrame
        features_df = pd.DataFrame([features])
        
        # Scale features
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            features_df[numeric_cols] = self.scaler.transform(features_df[numeric_cols])
        
        prediction = self.model.predict(features_df)[0]
        
        if self.problem_type == "classification" and hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(features_df)[0]
            return prediction, proba
        
        return prediction, None

class StudentAIAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analisis Penggunaan AI dalam Pendidikan")
        self.root.geometry("1400x800")
        
        # Initialize components
        self.data = None
        self.preprocessor = None
        self.model = RandomForestModel()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.is_teacher = False
        
        # Setup GUI
        self.setup_login_screen()
    
    def setup_login_screen(self):
        """Setup login screen"""
        self.clear_window()
        
        tk.Label(self.root, text="SISTEM ANALISIS AI DALAM PENDIDIKAN", 
                font=("Arial", 20, "bold"), fg="blue").pack(pady=30)
        
        tk.Label(self.root, text="Pilih Peran Anda:", 
                font=("Arial", 14)).pack(pady=20)
        
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=20)
        
        tk.Button(btn_frame, text="GURU/ADMIN", 
                 command=self.login_as_teacher,
                 width=20, height=3, bg="navy", fg="white",
                 font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=20)
        
        tk.Button(btn_frame, text="SISWA", 
                 command=self.login_as_student,
                 width=20, height=3, bg="darkgreen", fg="white",
                 font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=20)
    
    def login_as_teacher(self):
        """Login as teacher"""
        password = tk.simpledialog.askstring("Login Guru", "Masukkan Password:", show='*')
        if password == "admin123":
            self.is_teacher = True
            self.setup_main_interface()
            messagebox.showinfo("Login Berhasil", "Selamat datang, Guru!")
        else:
            messagebox.showerror("Login Gagal", "Password salah!")
    
    def login_as_student(self):
        """Login as student"""
        self.is_teacher = False
        self.setup_main_interface()
        messagebox.showinfo("Login Berhasil", "Selamat datang, Siswa!")
    
    def clear_window(self):
        """Clear all widgets from window"""
        for widget in self.root.winfo_children():
            widget.destroy()
    
    def setup_main_interface(self):
        """Setup main interface based on role"""
        self.clear_window()
        
        # Title
        title = "PANEL GURU - Analisis Penggunaan AI" if self.is_teacher else "PANEL SISWA - Analisis Penggunaan AI"
        title_label = tk.Label(self.root, text=title, 
                              font=("Arial", 18, "bold"), 
                              bg="lightgray", fg="darkblue")
        title_label.pack(fill=tk.X, pady=5)
        
        # Menu Frame
        menu_frame = tk.Frame(self.root, bg="lightblue")
        menu_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Common buttons
        common_buttons = [
            ("📁 Load Data", self.load_data),
            ("📊 Analisis Data", self.show_analysis),
            ("📈 Prediksi", self.show_prediction),
            ("💾 Export Data", self.export_data),
            ("🚪 Logout", self.logout)
        ]
        
        # Teacher-specific buttons
        if self.is_teacher:
            teacher_buttons = [
                ("🧹 Preprocessing", self.show_preprocessing),
                ("🤖 Train Model", self.train_model),
                ("📋 Evaluasi Model", self.evaluate_model)
            ]
            all_buttons = teacher_buttons + common_buttons
        else:
            all_buttons = common_buttons
        
        # Create buttons
        for text, command in all_buttons:
            btn = tk.Button(menu_frame, text=text, command=command,
                           width=15, height=2, bg="white", fg="black",
                           font=("Arial", 10))
            btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Status: Siap")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                             bd=1, relief=tk.SUNKEN, anchor=tk.W,
                             bg="lightgray", fg="black")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Main Content Area
        self.content_frame = tk.Frame(self.root, bg="white")
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Welcome message
        welcome_msg = "Silakan load data untuk memulai analisis" if self.data is None else f"Data loaded: {len(self.data)} records"
        tk.Label(self.content_frame, text=welcome_msg, 
                font=("Arial", 12), bg="white").pack(pady=20)
    
    def load_data(self):
        """Load data from CSV file"""
        file_path = filedialog.askopenfilename(
            title="Pilih file CSV",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    self.data = pd.read_csv(file_path)
                elif file_path.endswith('.xlsx'):
                    self.data = pd.read_excel(file_path)
                else:
                    # Try CSV first
                    self.data = pd.read_csv(file_path)
                
                self.preprocessor = PreprocessingModule(self.data)
                self.status_var.set(f"Status: Data dimuat - {len(self.data)} baris, {len(self.data.columns)} kolom")
                messagebox.showinfo("Sukses", f"Data berhasil dimuat!\n\nBaris: {len(self.data)}\nKolom: {len(self.data.columns)}")
                self.show_data_preview()
                
            except Exception as e:
                messagebox.showerror("Error", f"Gagal memuat data:\n{str(e)}")
    
    def show_data_preview(self):
        """Show data preview"""
        self.clear_content()
        
        if self.data is None:
            tk.Label(self.content_frame, text="Tidak ada data yang dimuat", 
                    font=("Arial", 12), bg="white").pack(pady=50)
            return
        
        # Create notebook for tabs
        notebook = ttk.Notebook(self.content_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 1: Data Preview
        preview_frame = ttk.Frame(notebook)
        notebook.add(preview_frame, text="Preview Data")
        
        # Add scrollbars
        container = tk.Frame(preview_frame)
        container.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(container)
        scrollbar_y = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollbar_x = tk.Scrollbar(container, orient="horizontal", command=canvas.xview)
        
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        # Create treeview
        tree = ttk.Treeview(scrollable_frame)
        
        # Define columns
        tree["columns"] = list(self.data.columns)
        tree["show"] = "headings"
        
        # Create headings
        for col in self.data.columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, minwidth=50)
        
        # Add data (first 50 rows)
        for i, row in self.data.head(50).iterrows():
            tree.insert("", tk.END, values=list(row))
        
        tree.pack(fill=tk.BOTH, expand=True)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar_y.pack(side="right", fill="y")
        scrollbar_x.pack(side="bottom", fill="x")
        
        # Tab 2: Data Info
        info_frame = ttk.Frame(notebook)
        notebook.add(info_frame, text="Info Data")
        
        info_text = tk.Text(info_frame, wrap=tk.WORD, height=20)
        info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        info = "=== INFORMASI DATA ===\n\n"
        info += f"Shape: {self.data.shape}\n"
        info += f"Total Records: {len(self.data)}\n"
        info += f"Total Features: {len(self.data.columns)}\n\n"
        
        info += "=== KOLOM ===\n"
        for i, col in enumerate(self.data.columns, 1):
            info += f"{i}. {col} ({self.data[col].dtype})\n"
        
        info += "\n=== MISSING VALUES ===\n"
        missing = self.data.isnull().sum()
        for col in self.data.columns:
            if missing[col] > 0:
                info += f"{col}: {missing[col]} missing values\n"
        
        info += "\n=== DATA TYPES ===\n"
        dtypes = self.data.dtypes.value_counts()
        for dtype, count in dtypes.items():
            info += f"{dtype}: {count} columns\n"
        
        info_text.insert(tk.END, info)
        info_text.config(state=tk.DISABLED)
        
        # Tab 3: Statistics
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="Statistik")
        
        stats_text = tk.Text(stats_frame, wrap=tk.WORD, height=20)
        stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats = self.data[numeric_cols].describe()
            stats_text.insert(tk.END, str(stats))
        else:
            stats_text.insert(tk.END, "Tidak ada kolom numerik")
        
        stats_text.config(state=tk.DISABLED)
    
    def show_preprocessing(self):
        """Show preprocessing options (teacher only)"""
        if not self.is_teacher:
            messagebox.showwarning("Akses Ditolak", "Hanya guru yang dapat mengakses fitur ini")
            return
        
        if self.data is None:
            messagebox.showwarning("Peringatan", "Load data terlebih dahulu")
            return
        
        self.clear_content()
        
        tk.Label(self.content_frame, text="PREPROCESSING DATA", 
                font=("Arial", 16, "bold"), bg="white").pack(pady=10)
        
        # Create notebook for preprocessing steps
        notebook = ttk.Notebook(self.content_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 1. Data Cleaning
        cleaning_frame = ttk.Frame(notebook)
        notebook.add(cleaning_frame, text="Pembersihan Data")
        
        tk.Label(cleaning_frame, text="Pembersihan Data", 
                font=("Arial", 14, "bold")).pack(pady=10)
        
        # Missing values
        missing_frame = tk.LabelFrame(cleaning_frame, text="Missing Values", padx=10, pady=10)
        missing_frame.pack(fill=tk.X, padx=10, pady=5)
        
        missing_count = self.data.isnull().sum().sum()
        tk.Label(missing_frame, text=f"Total missing values: {missing_count}").pack(anchor=tk.W)
        
        tk.Button(missing_frame, text="Hapus Rows dengan Missing Values",
                 command=self.handle_remove_missing, width=30).pack(pady=5)
        tk.Button(missing_frame, text="Isi dengan Median/Mode",
                 command=self.handle_fill_missing, width=30).pack(pady=5)
        
        # Duplicates
        dup_frame = tk.LabelFrame(cleaning_frame, text="Duplikat", padx=10, pady=10)
        dup_frame.pack(fill=tk.X, padx=10, pady=5)
        
        dup_count = self.data.duplicated().sum()
        tk.Label(dup_frame, text=f"Jumlah duplikat: {dup_count}").pack(anchor=tk.W)
        
        tk.Button(dup_frame, text="Hapus Duplikat",
                 command=self.handle_remove_duplicates, width=30).pack(pady=5)
        
        # Outliers
        outlier_frame = tk.LabelFrame(cleaning_frame, text="Outliers", padx=10, pady=10)
        outlier_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(outlier_frame, text="Deteksi Outliers (IQR Method)",
                 command=self.handle_detect_outliers, width=30).pack(pady=5)
        tk.Button(outlier_frame, text="Hapus Outliers",
                 command=self.handle_remove_outliers, width=30).pack(pady=5)
        
        # 2. Data Transformation
        transform_frame = ttk.Frame(notebook)
        notebook.add(transform_frame, text="Transformasi Data")
        
        tk.Label(transform_frame, text="Transformasi Data", 
                font=("Arial", 14, "bold")).pack(pady=10)
        
        # Column selection
        col_frame = tk.Frame(transform_frame)
        col_frame.pack(pady=10)
        
        tk.Label(col_frame, text="Pilih Kolom:").pack(side=tk.LEFT)
        
        self.transform_col_var = tk.StringVar()
        col_combo = ttk.Combobox(col_frame, textvariable=self.transform_col_var,
                                values=list(self.data.columns), width=30)
        col_combo.pack(side=tk.LEFT, padx=10)
        col_combo.set(self.data.columns[0] if len(self.data.columns) > 0 else "")
        
        # Encoding
        encode_frame = tk.LabelFrame(transform_frame, text="Encoding", padx=10, pady=10)
        encode_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(encode_frame, text="Label Encoding",
                 command=self.handle_label_encode, width=20).pack(side=tk.LEFT, padx=5)
        tk.Button(encode_frame, text="One-Hot Encoding",
                 command=self.handle_onehot_encode, width=20).pack(side=tk.LEFT, padx=5)
        
        # Scaling
        scale_frame = tk.LabelFrame(transform_frame, text="Scaling", padx=10, pady=10)
        scale_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(scale_frame, text="Standard Scaling",
                 command=self.handle_standard_scale, width=20).pack(side=tk.LEFT, padx=5)
        tk.Button(scale_frame, text="Min-Max Scaling",
                 command=self.handle_minmax_scale, width=20).pack(side=tk.LEFT, padx=5)
        
        # 3. Data Reduction
        reduction_frame = ttk.Frame(notebook)
        notebook.add(reduction_frame, text="Reduksi Data")
        
        tk.Label(reduction_frame, text="Reduksi Data", 
                font=("Arial", 14, "bold")).pack(pady=10)
        
        # PCA
        pca_frame = tk.LabelFrame(reduction_frame, text="PCA", padx=10, pady=10)
        pca_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(pca_frame, text="Jumlah Komponen:").pack(side=tk.LEFT)
        
        self.pca_n_var = tk.IntVar(value=2)
        pca_spin = tk.Spinbox(pca_frame, from_=1, to=min(10, len(self.data.columns)), 
                             textvariable=self.pca_n_var, width=10)
        pca_spin.pack(side=tk.LEFT, padx=10)
        
        tk.Button(pca_frame, text="Apply PCA",
                 command=self.handle_pca, width=20).pack(side=tk.LEFT)
        
        # 4. Data Splitting
        split_frame = ttk.Frame(notebook)
        notebook.add(split_frame, text="Pembagian Data")
        
        tk.Label(split_frame, text="Pembagian Data Train-Test", 
                font=("Arial", 14, "bold")).pack(pady=10)
        
        split_controls = tk.Frame(split_frame)
        split_controls.pack(pady=20)
        
        # Target column selection
        tk.Label(split_controls, text="Target Column:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.target_col_var = tk.StringVar()
        target_combo = ttk.Combobox(split_controls, textvariable=self.target_col_var,
                                   values=list(self.data.columns), width=30)
        target_combo.grid(row=0, column=1, padx=10, pady=5)
        target_combo.set(self.data.columns[-1] if len(self.data.columns) > 0 else "")
        
        # Test size
        tk.Label(split_controls, text="Test Size:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.test_size_var = tk.DoubleVar(value=0.2)
        test_size_entry = tk.Entry(split_controls, textvariable=self.test_size_var, width=10)
        test_size_entry.grid(row=1, column=1, padx=10, pady=5, sticky=tk.W)
        
        # Random state
        tk.Label(split_controls, text="Random State:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.random_state_var = tk.IntVar(value=42)
        random_state_entry = tk.Entry(split_controls, textvariable=self.random_state_var, width=10)
        random_state_entry.grid(row=2, column=1, padx=10, pady=5, sticky=tk.W)
        
        # Split button
        tk.Button(split_frame, text="Split Data", 
                 command=self.handle_split_data,
                 width=20, height=2, bg="green", fg="white").pack(pady=10)
        
        # Show current split status
        if self.X_train is not None:
            status_text = f"Data sudah terbagi:\n"
            status_text += f"Train: {len(self.X_train)} samples\n"
            status_text += f"Test: {len(self.X_test)} samples\n"
            status_text += f"Features: {len(self.X_train.columns)}"
            
            status_label = tk.Label(split_frame, text=status_text, justify=tk.LEFT)
            status_label.pack(pady=10)
    
    def handle_remove_missing(self):
        """Handle remove missing values"""
        if self.preprocessor:
            result = self.preprocessor.remove_missing()
            self.data = self.preprocessor.data
            messagebox.showinfo("Sukses", result)
            self.show_preprocessing()
    
    def handle_fill_missing(self):
        """Handle fill missing values"""
        if self.preprocessor:
            result = self.preprocessor.fill_missing()
            self.data = self.preprocessor.data
            messagebox.showinfo("Sukses", result)
            self.show_preprocessing()
    
    def handle_remove_duplicates(self):
        """Handle remove duplicates"""
        if self.preprocessor:
            result = self.preprocessor.remove_duplicates()
            self.data = self.preprocessor.data
            messagebox.showinfo("Sukses", result)
            self.show_preprocessing()
    
    def handle_detect_outliers(self):
        """Handle detect outliers"""
        if self.preprocessor:
            outliers = self.preprocessor.detect_outliers_iqr()
            
            if outliers:
                result = "Outliers ditemukan:\n"
                for col, count in outliers.items():
                    result += f"{col}: {count} outliers\n"
            else:
                result = "Tidak ditemukan outliers"
            
            messagebox.showinfo("Deteksi Outliers", result)
    
    def handle_remove_outliers(self):
        """Handle remove outliers"""
        if self.preprocessor:
            result = self.preprocessor.remove_outliers_iqr()
            self.data = self.preprocessor.data
            messagebox.showinfo("Sukses", result)
            self.show_preprocessing()
    
    def handle_label_encode(self):
        """Handle label encoding"""
        if self.preprocessor:
            col = self.transform_col_var.get()
            if col:
                result = self.preprocessor.label_encode_column(col)
                self.data = self.preprocessor.data
                messagebox.showinfo("Sukses", result)
                self.show_preprocessing()
    
    def handle_onehot_encode(self):
        """Handle one-hot encoding"""
        if self.preprocessor:
            col = self.transform_col_var.get()
            if col:
                result = self.preprocessor.onehot_encode_column(col)
                self.data = self.preprocessor.data
                messagebox.showinfo("Sukses", result)
                self.show_preprocessing()
    
    def handle_standard_scale(self):
        """Handle standard scaling"""
        if self.preprocessor:
            col = self.transform_col_var.get()
            if col:
                result = self.preprocessor.standard_scale_column(col)
                self.data = self.preprocessor.data
                messagebox.showinfo("Sukses", result)
                self.show_preprocessing()
    
    def handle_minmax_scale(self):
        """Handle min-max scaling"""
        if self.preprocessor:
            col = self.transform_col_var.get()
            if col:
                result = self.preprocessor.minmax_scale_column(col)
                self.data = self.preprocessor.data
                messagebox.showinfo("Sukses", result)
                self.show_preprocessing()
    
    def handle_pca(self):
        """Handle PCA"""
        if self.preprocessor:
            n_components = self.pca_n_var.get()
            result = self.preprocessor.apply_pca(n_components)
            self.data = self.preprocessor.data
            messagebox.showinfo("Sukses", result)
            self.show_preprocessing()
    
    def handle_split_data(self):
        """Handle data splitting"""
        if self.preprocessor:
            target_col = self.target_col_var.get()
            test_size = self.test_size_var.get()
            random_state = self.random_state_var.get()
            
            if not target_col:
                messagebox.showwarning("Peringatan", "Pilih kolom target terlebih dahulu")
                return
            
            self.X_train, self.X_test, self.y_train, self.y_test, result = \
                self.preprocessor.split_data(target_col, test_size, random_state)
            
            if self.X_train is not None:
                messagebox.showinfo("Sukses", 
                                  f"{result}\n\n"
                                  f"Train samples: {len(self.X_train)}\n"
                                  f"Test samples: {len(self.X_test)}")
                self.show_preprocessing()
            else:
                messagebox.showerror("Error", result)
    
    def show_analysis(self):
        """Show data analysis"""
        self.clear_content()
        
        if self.data is None:
            tk.Label(self.content_frame, text="Load data terlebih dahulu", 
                    font=("Arial", 12), bg="white").pack(pady=50)
            return
        
        notebook = ttk.Notebook(self.content_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 1. Basic Statistics
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="Statistik")
        
        stats_text = tk.Text(stats_frame, wrap=tk.WORD)
        stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats = self.data[numeric_cols].describe()
            stats_text.insert(tk.END, str(stats))
        else:
            stats_text.insert(tk.END, "Tidak ada kolom numerik untuk analisis statistik")
        
        stats_text.config(state=tk.DISABLED)
        
        # 2. Correlation Analysis
        corr_frame = ttk.Frame(notebook)
        notebook.add(corr_frame, text="Korelasi")
        
        tk.Button(corr_frame, text="Tampilkan Heatmap Korelasi",
                 command=self.show_correlation_heatmap, width=30, height=2).pack(pady=20)
        
        # 3. Distribution Analysis
        dist_frame = ttk.Frame(notebook)
        notebook.add(dist_frame, text="Distribusi")
        
        tk.Label(dist_frame, text="Pilih Kolom untuk Analisis Distribusi:", 
                font=("Arial", 12)).pack(pady=10)
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            self.dist_col_var = tk.StringVar(value=numeric_cols[0])
            dist_combo = ttk.Combobox(dist_frame, textvariable=self.dist_col_var,
                                     values=list(numeric_cols), width=30)
            dist_combo.pack(pady=5)
            
            tk.Button(dist_frame, text="Tampilkan Histogram",
                     command=self.show_histogram, width=20).pack(pady=5)
            tk.Button(dist_frame, text="Tampilkan Box Plot",
                     command=self.show_boxplot, width=20).pack(pady=5)
        else:
            tk.Label(dist_frame, text="Tidak ada kolom numerik").pack(pady=20)
    
    def show_correlation_heatmap(self):
        """Show correlation heatmap"""
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            messagebox.showwarning("Peringatan", "Minimal 2 kolom numerik diperlukan")
            return
        
        # Create new window
        heatmap_window = tk.Toplevel(self.root)
        heatmap_window.title("Heatmap Korelasi")
        heatmap_window.geometry("800x600")
        
        # Calculate correlation
        corr_matrix = numeric_data.corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                   square=True, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title("Heatmap Korelasi")
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, heatmap_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add close button
        tk.Button(heatmap_window, text="Tutup", 
                 command=heatmap_window.destroy).pack(pady=10)
    
    def show_histogram(self):
        """Show histogram of selected column"""
        col = self.dist_col_var.get()
        
        if col not in self.data.columns:
            return
        
        # Create new window
        hist_window = tk.Toplevel(self.root)
        hist_window.title(f"Histogram - {col}")
        hist_window.geometry("600x500")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        self.data[col].hist(bins=30, ax=ax, edgecolor='black', color='skyblue')
        ax.set_title(f"Distribusi {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frekuensi")
        
        # Add statistics
        mean_val = self.data[col].mean()
        median_val = self.data[col].median()
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
        ax.legend()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, hist_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add close button
        tk.Button(hist_window, text="Tutup", 
                 command=hist_window.destroy).pack(pady=10)
    
    def show_boxplot(self):
        """Show box plot of selected column"""
        col = self.dist_col_var.get()
        
        if col not in self.data.columns:
            return
        
        # Create new window
        box_window = tk.Toplevel(self.root)
        box_window.title(f"Box Plot - {col}")
        box_window.geometry("600x500")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        self.data[[col]].boxplot(ax=ax)
        ax.set_title(f"Box Plot {col}")
        ax.set_ylabel(col)
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, box_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add close button
        tk.Button(box_window, text="Tutup", 
                 command=box_window.destroy).pack(pady=10)
    
    def train_model(self):
        """Train Random Forest model (teacher only)"""
        if not self.is_teacher:
            messagebox.showwarning("Akses Ditolak", "Hanya guru yang dapat melatih model")
            return
        
        if self.X_train is None or self.y_train is None:
            messagebox.showwarning("Peringatan", "Split data terlebih dahulu")
            return
        
        self.clear_content()
        
        tk.Label(self.content_frame, text="TRAINING RANDOM FOREST MODEL", 
                font=("Arial", 16, "bold"), bg="white").pack(pady=10)
        
        # Model configuration
        config_frame = tk.Frame(self.content_frame, bg="white")
        config_frame.pack(pady=20)
        
        tk.Label(config_frame, text="Parameter Model:", 
                font=("Arial", 12, "bold"), bg="white").grid(row=0, column=0, columnspan=2, pady=10)
        
        # n_estimators
        tk.Label(config_frame, text="n_estimators:", bg="white").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.n_estimators_var = tk.IntVar(value=100)
        tk.Entry(config_frame, textvariable=self.n_estimators_var, width=15).grid(row=1, column=1, pady=5, padx=10)
        
        # max_depth
        tk.Label(config_frame, text="max_depth (None untuk unlimited):", bg="white").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.max_depth_var = tk.StringVar(value="None")
        tk.Entry(config_frame, textvariable=self.max_depth_var, width=15).grid(row=2, column=1, pady=5, padx=10)
        
        # random_state
        tk.Label(config_frame, text="random_state:", bg="white").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.rf_random_state_var = tk.IntVar(value=42)
        tk.Entry(config_frame, textvariable=self.rf_random_state_var, width=15).grid(row=3, column=1, pady=5, padx=10)
        
        # Problem type detection
        unique_classes = len(np.unique(self.y_train))
        problem_type = "Klasifikasi" if unique_classes < 10 else "Regresi"
        tk.Label(config_frame, text=f"Tipe Problem: {problem_type}", 
                font=("Arial", 10, "italic"), bg="white").grid(row=4, column=0, columnspan=2, pady=10)
        
        # Train button
        tk.Button(self.content_frame, text="Train Model", 
                 command=self.execute_training,
                 width=20, height=2, bg="green", fg="white",
                 font=("Arial", 12)).pack(pady=20)
        
        # Result area
        self.result_text = tk.Text(self.content_frame, height=20, width=80)
        self.result_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(self.result_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.result_text.yview)
    
    def execute_training(self):
        """Execute model training"""
        try:
            # Get parameters
            n_estimators = self.n_estimators_var.get()
            max_depth = None if self.max_depth_var.get() == "None" else int(self.max_depth_var.get())
            random_state = self.rf_random_state_var.get()
            
            # Train model
            result = self.model.train(self.X_train, self.y_train, 
                                     n_estimators, max_depth, random_state)
            
            # Make predictions
            predictions, _ = self.model.predict(self.X_test)
            
            # Display results
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "=== HASIL TRAINING ===\n\n")
            self.result_text.insert(tk.END, f"{result}\n\n")
            self.result_text.insert(tk.END, f"n_estimators: {n_estimators}\n")
            self.result_text.insert(tk.END, f"max_depth: {max_depth}\n")
            self.result_text.insert(tk.END, f"Training samples: {len(self.X_train)}\n")
            self.result_text.insert(tk.END, f"Test samples: {len(self.X_test)}\n\n")
            
            # Feature importance
            feature_importance = self.model.get_feature_importance(self.X_train.columns)
            if feature_importance is not None:
                self.result_text.insert(tk.END, "=== FEATURE IMPORTANCE (Top 10) ===\n")
                for i, row in feature_importance.head(10).iterrows():
                    self.result_text.insert(tk.END, f"{row['feature']}: {row['importance']:.4f}\n")
            
            self.status_var.set("Status: Model berhasil dilatih")
            
        except Exception as e:
            messagebox.showerror("Error Training", f"Terjadi error:\n{str(e)}")
    
    def evaluate_model(self):
        """Evaluate model (teacher only)"""
        if not self.is_teacher:
            messagebox.showwarning("Akses Ditolak", "Hanya guru yang dapat mengevaluasi model")
            return
        
        if self.model.model is None:
            messagebox.showwarning("Peringatan", "Train model terlebih dahulu")
            return
        
        self.clear_content()
        
        tk.Label(self.content_frame, text="EVALUASI MODEL", 
                font=("Arial", 16, "bold"), bg="white").pack(pady=10)
        
        # Evaluate model
        results = self.model.evaluate(self.X_test, self.y_test)
        
        # Display results
        result_text = tk.Text(self.content_frame, height=25, width=80)
        result_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        result_text.insert(tk.END, "=== HASIL EVALUASI ===\n\n")
        
        if self.model.problem_type == "classification":
            # Classification metrics
            metrics = results['metrics']
            result_text.insert(tk.END, "=== METRIK KLASIFIKASI ===\n")
            result_text.insert(tk.END, f"Akurasi: {metrics['accuracy']:.4f}\n")
            result_text.insert(tk.END, f"Presisi: {metrics['precision']:.4f}\n")
            result_text.insert(tk.END, f"Recall: {metrics['recall']:.4f}\n")
            result_text.insert(tk.END, f"F1-Score: {metrics['f1_score']:.4f}\n\n")
            
            # Classification report
            report = results['classification_report']
            result_text.insert(tk.END, "=== LAPORAN KLASIFIKASI ===\n")
            for key, value in report.items():
                if isinstance(value, dict):
                    result_text.insert(tk.END, f"\n{key}:\n")
                    for k, v in value.items():
                        result_text.insert(tk.END, f"  {k}: {v:.4f}\n")
                else:
                    result_text.insert(tk.END, f"{key}: {value:.4f}\n")
            
            # Button to show confusion matrix
            tk.Button(self.content_frame, text="Tampilkan Confusion Matrix",
                     command=self.show_confusion_matrix, width=25).pack(pady=10)
            
        else:
            # Regression metrics
            metrics = results['metrics']
            result_text.insert(tk.END, "=== METRIK REGRESI ===\n")
            result_text.insert(tk.END, f"MSE: {metrics['mse']:.4f}\n")
            result_text.insert(tk.END, f"MAE: {metrics['mae']:.4f}\n")
            result_text.insert(tk.END, f"RMSE: {metrics['rmse']:.4f}\n\n")
            
            # Button to show residual plot
            tk.Button(self.content_frame, text="Tampilkan Residual Plot",
                     command=self.show_residual_plot, width=25).pack(pady=10)
        
        result_text.config(state=tk.DISABLED)
        
        # Feature importance visualization button
        tk.Button(self.content_frame, text="Tampilkan Feature Importance",
                 command=self.show_feature_importance, width=25).pack(pady=10)
    
    def show_confusion_matrix(self):
        """Show confusion matrix visualization"""
        if self.model.problem_type != "classification":
            return
        
        results = self.model.evaluate(self.X_test, self.y_test)
        cm = np.array(results['confusion_matrix'])
        
        # Create new window
        cm_window = tk.Toplevel(self.root)
        cm_window.title("Confusion Matrix")
        cm_window.geometry("700x600")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, cm_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add close button
        tk.Button(cm_window, text="Tutup", 
                 command=cm_window.destroy).pack(pady=10)
    
    def show_residual_plot(self):
        """Show residual plot for regression"""
        if self.model.problem_type != "regression":
            return
        
        results = self.model.evaluate(self.X_test, self.y_test)
        predictions = np.array(results['predictions'])
        residuals = np.array(results['residuals'])
        
        # Create new window
        residual_window = tk.Toplevel(self.root)
        residual_window.title("Residual Analysis")
        residual_window.geometry("900x600")
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Residuals vs Predicted
        axes[0].scatter(predictions, residuals, alpha=0.6)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted')
        
        # Histogram of residuals
        axes[1].hist(residuals, bins=30, edgecolor='black', color='skyblue')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Residuals')
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, residual_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add close button
        tk.Button(residual_window, text="Tutup", 
                 command=residual_window.destroy).pack(pady=10)
    
    def show_feature_importance(self):
        """Show feature importance visualization"""
        feature_importance = self.model.get_feature_importance(self.X_train.columns)
        
        if feature_importance is None:
            return
        
        # Create new window
        fi_window = tk.Toplevel(self.root)
        fi_window.title("Feature Importance")
        fi_window.geometry("800x600")
        
        # Get top 15 features
        top_features = feature_importance.head(15)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(top_features))
        
        ax.barh(y_pos, top_features['importance'], color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'])
        ax.invert_yaxis()  # highest importance at top
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance (Top 15)')
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, fi_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add close button
        tk.Button(fi_window, text="Tutup", 
                 command=fi_window.destroy).pack(pady=10)
    
    def show_prediction(self):
        """Show prediction interface"""
        self.clear_content()
        
        if self.model.model is None:
            tk.Label(self.content_frame, text="Model belum dilatih. Guru harus melatih model terlebih dahulu.", 
                    font=("Arial", 12), bg="white").pack(pady=50)
            return
        
        tk.Label(self.content_frame, text="PREDIKSI DATA BARU", 
                font=("Arial", 16, "bold"), bg="white").pack(pady=10)
        
        # Create input form
        input_frame = tk.Frame(self.content_frame, bg="white")
        input_frame.pack(pady=20)
        
        self.input_vars = {}
        
        # Get sample from training data
        sample_row = self.X_train.iloc[0] if len(self.X_train) > 0 else pd.Series()
        
        # Create input fields for each feature
        for i, col in enumerate(self.X_train.columns):
            tk.Label(input_frame, text=f"{col}:", bg="white").grid(row=i, column=0, sticky=tk.W, pady=5, padx=5)
            
            # Get sample value
            sample_val = sample_row[col] if col in sample_row else 0
            
            # Create appropriate input widget
            if self.X_train[col].dtype == 'object' or len(self.X_train[col].unique()) < 10:
                # Categorical - use combobox
                unique_vals = self.X_train[col].unique()[:20]
                var = tk.StringVar(value=str(sample_val))
                combo = ttk.Combobox(input_frame, textvariable=var, 
                                    values=list(unique_vals), width=30)
                combo.grid(row=i, column=1, pady=5, padx=5)
            else:
                # Numerical - use entry
                var = tk.DoubleVar(value=float(sample_val) if pd.notna(sample_val) else 0)
                entry = tk.Entry(input_frame, textvariable=var, width=33)
                entry.grid(row=i, column=1, pady=5, padx=5)
            
            self.input_vars[col] = var
        
        # Predict button
        tk.Button(self.content_frame, text="Predict", 
                 command=self.make_prediction,
                 width=20, height=2, bg="blue", fg="white",
                 font=("Arial", 12)).pack(pady=20)
        
        # Result display
        self.prediction_result = tk.Text(self.content_frame, height=8, width=60)
        self.prediction_result.pack(pady=10, padx=10)
        self.prediction_result.insert(tk.END, "Hasil prediksi akan muncul di sini...")
        self.prediction_result.config(state=tk.DISABLED)
        
        # Batch prediction
        batch_frame = tk.Frame(self.content_frame, bg="white")
        batch_frame.pack(pady=10)
        
        tk.Button(batch_frame, text="Prediksi dari File CSV", 
                 command=self.predict_from_file, width=20).pack(side=tk.LEFT, padx=10)
        tk.Button(batch_frame, text="Reset Input", 
                 command=self.reset_prediction_input, width=20).pack(side=tk.LEFT, padx=10)
    
    def make_prediction(self):
        """Make prediction based on input values"""
        try:
            # Prepare input data
            input_data = {}
            for col, var in self.input_vars.items():
                value = var.get()
                
                # Convert to appropriate type
                if self.X_train[col].dtype == 'object':
                    input_data[col] = str(value)
                else:
                    try:
                        input_data[col] = float(value)
                    except:
                        input_data[col] = 0.0
            
            # Make prediction
            prediction, probabilities = self.model.predict_single(input_data)
            
            # Display result
            self.prediction_result.config(state=tk.NORMAL)
            self.prediction_result.delete(1.0, tk.END)
            
            if self.model.problem_type == "classification":
                self.prediction_result.insert(tk.END, f"Hasil Prediksi: Kelas {prediction}\n\n")
                
                if probabilities is not None:
                    self.prediction_result.insert(tk.END, "Probabilitas Kelas:\n")
                    for i, prob in enumerate(probabilities):
                        self.prediction_result.insert(tk.END, f"  Kelas {i}: {prob:.4f}\n")
            else:
                self.prediction_result.insert(tk.END, f"Hasil Prediksi: {prediction:.4f}\n")
            
            self.prediction_result.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Error Prediksi", f"Terjadi error:\n{str(e)}")
    
    def predict_from_file(self):
        """Make predictions from CSV file"""
        if self.model.model is None:
            messagebox.showwarning("Peringatan", "Model belum dilatih")
            return
        
        file_path = filedialog.askopenfilename(
            title="Pilih file CSV untuk prediksi",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if file_path:
            try:
                # Load data
                new_data = pd.read_csv(file_path)
                
                # Check if all required columns are present
                missing_cols = set(self.X_train.columns) - set(new_data.columns)
                if missing_cols:
                    messagebox.showwarning("Peringatan", 
                                         f"Kolom berikut tidak ditemukan: {missing_cols}")
                    return
                
                # Ensure correct column order
                new_data = new_data[self.X_train.columns]
                
                # Make predictions
                predictions, _ = self.model.predict(new_data)
                
                # Add predictions to data
                result_data = new_data.copy()
                result_data['Prediction'] = predictions
                
                # Ask for save location
                save_path = filedialog.asksaveasfilename(
                    title="Simpan hasil prediksi",
                    defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv")]
                )
                
                if save_path:
                    result_data.to_csv(save_path, index=False)
                    messagebox.showinfo("Sukses", 
                                      f"Prediksi berhasil disimpan di:\n{save_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Gagal membuat prediksi:\n{str(e)}")
    
    def reset_prediction_input(self):
        """Reset prediction input fields"""
        if self.X_train is not None:
            sample_row = self.X_train.iloc[0] if len(self.X_train) > 0 else pd.Series()
            
            for col, var in self.input_vars.items():
                sample_val = sample_row[col] if col in sample_row else 0
                
                if isinstance(var, tk.StringVar):
                    var.set(str(sample_val))
                elif isinstance(var, tk.DoubleVar):
                    var.set(float(sample_val) if pd.notna(sample_val) else 0)
        
        self.prediction_result.config(state=tk.NORMAL)
        self.prediction_result.delete(1.0, tk.END)
        self.prediction_result.insert(tk.END, "Input telah direset. Hasil prediksi akan muncul di sini...")
        self.prediction_result.config(state=tk.DISABLED)
    
    def export_data(self):
        """Export data and results to CSV"""
        if self.data is None:
            messagebox.showwarning("Peringatan", "Tidak ada data untuk diekspor")
            return
        
        # Create export directory
        export_dir = "hasil_export"
        os.makedirs(export_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # 1. Export original data
            original_path = f"{export_dir}/data_asli_{timestamp}.csv"
            self.data.to_csv(original_path, index=False, encoding='utf-8-sig')
            
            # 2. Export preprocessed data (if available)
            if self.X_train is not None:
                # Combine train and test data
                train_data = self.X_train.copy()
                train_data['target'] = self.y_train
                train_data['dataset'] = 'train'
                
                test_data = self.X_test.copy()
                test_data['target'] = self.y_test
                test_data['dataset'] = 'test'
                
                preprocessed_data = pd.concat([train_data, test_data], ignore_index=True)
                preprocessed_path = f"{export_dir}/data_preprocessed_{timestamp}.csv"
                preprocessed_data.to_csv(preprocessed_path, index=False, encoding='utf-8-sig')
            
            # 3. Export model results (if available)
            if self.model.model is not None:
                # Export predictions
                if self.X_test is not None:
                    predictions, _ = self.model.predict(self.X_test)
                    
                    predictions_df = self.X_test.copy()
                    predictions_df['Aktual'] = self.y_test
                    predictions_df['Prediksi'] = predictions
                    
                    predictions_path = f"{export_dir}/hasil_prediksi_{timestamp}.csv"
                    predictions_df.to_csv(predictions_path, index=False, encoding='utf-8-sig')
                
                # Export feature importance
                feature_importance = self.model.get_feature_importance(self.X_train.columns)
                if feature_importance is not None:
                    fi_path = f"{export_dir}/feature_importance_{timestamp}.csv"
                    feature_importance.to_csv(fi_path, index=False, encoding='utf-8-sig')
                
                # Export metrics
                if hasattr(self.model, 'metrics') and self.model.metrics:
                    metrics_df = pd.DataFrame([self.model.metrics])
                    metrics_path = f"{export_dir}/metrics_model_{timestamp}.csv"
                    metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
            
            # Show success message
            messagebox.showinfo("Ekspor Berhasil", 
                              f"Semua data telah berhasil diekspor ke folder:\n{export_dir}")
            
        except Exception as e:
            messagebox.showerror("Error Ekspor", f"Gagal mengekspor data:\n{str(e)}")
    
    def logout(self):
        """Logout and return to login screen"""
        self.data = None
        self.preprocessor = None
        self.model = RandomForestModel()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.setup_login_screen()
    
    def clear_content(self):
        """Clear content frame"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = StudentAIAnalysisApp(root)
    
    # Center the window
    window_width = 1400
    window_height = 800
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width/2 - window_width/2)
    center_y = int(screen_height/2 - window_height/2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    
    root.mainloop()

if __name__ == "__main__":
    main()
