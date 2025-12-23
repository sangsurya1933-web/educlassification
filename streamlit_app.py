import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
import json
from datetime import datetime

class StudentAIAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analisis AI dalam Pendidikan")
        self.root.geometry("1400x800")
        
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = StandardScaler()
        self.is_teacher = False
        self.le_dict = {}
        
        # Setup GUI
        self.setup_login_screen()
        
    def setup_login_screen(self):
        """Setup login screen to choose role"""
        self.clear_window()
        
        tk.Label(self.root, text="SISTEM ANALISIS AI DALAM PENDIDIKAN", 
                font=("Arial", 20, "bold")).pack(pady=20)
        
        tk.Label(self.root, text="Pilih Peran Anda:", 
                font=("Arial", 14)).pack(pady=10)
        
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=20)
        
        tk.Button(btn_frame, text="GURU/ADMIN", 
                 command=self.login_as_teacher,
                 width=20, height=3, bg="blue", fg="white",
                 font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=20)
        
        tk.Button(btn_frame, text="SISWA", 
                 command=self.login_as_student,
                 width=20, height=3, bg="green", fg="white",
                 font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=20)
        
    def login_as_teacher(self):
        """Login as teacher/admin"""
        password = tk.simpledialog.askstring("Login Guru", "Masukkan Password:", show='*')
        if password == "guru123":  # Simple password for demo
            self.is_teacher = True
            self.setup_main_interface()
        else:
            messagebox.showerror("Error", "Password salah!")
            
    def login_as_student(self):
        """Login as student"""
        self.is_teacher = False
        self.setup_main_interface()
        
    def clear_window(self):
        """Clear all widgets from window"""
        for widget in self.root.winfo_children():
            widget.destroy()
            
    def setup_main_interface(self):
        """Setup main interface based on role"""
        self.clear_window()
        
        # Title
        title = "PANEL GURU" if self.is_teacher else "PANEL SISWA"
        tk.Label(self.root, text=title, 
                font=("Arial", 18, "bold")).pack(pady=10)
        
        # Menu Frame
        menu_frame = tk.Frame(self.root)
        menu_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Buttons based on role
        if self.is_teacher:
            buttons = [
                ("Load Data", self.load_data),
                ("Preprocessing", self.show_preprocessing),
                ("Analisis Data", self.show_analysis),
                ("Train Model", self.train_model),
                ("Evaluasi Model", self.evaluate_model),
                ("Export Data", self.export_data),
                ("Logout", self.logout)
            ]
        else:
            buttons = [
                ("Load Data", self.load_data),
                ("Analisis Data", self.show_analysis),
                ("Prediksi", self.predict_data),
                ("Export Data", self.export_data),
                ("Logout", self.logout)
            ]
        
        for text, command in buttons:
            tk.Button(menu_frame, text=text, command=command,
                     width=15, height=2).pack(side=tk.LEFT, padx=5)
        
        # Main Content Area
        self.content_frame = tk.Frame(self.root)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Status: Ready")
        tk.Label(self.root, textvariable=self.status_var, 
                bd=1, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)
    
    def logout(self):
        """Logout and return to login screen"""
        self.data = None
        self.model = None
        self.setup_login_screen()
    
    def load_data(self):
        """Load data from CSV file"""
        file_path = filedialog.askopenfilename(
            title="Pilih file data",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                self.status_var.set(f"Status: Data loaded - {len(self.data)} records")
                messagebox.showinfo("Success", "Data berhasil dimuat!")
                self.show_data_preview()
            except Exception as e:
                messagebox.showerror("Error", f"Gagal memuat data: {str(e)}")
    
    def show_data_preview(self):
        """Show data preview in content area"""
        self.clear_content()
        
        if self.data is None:
            tk.Label(self.content_frame, text="Tidak ada data yang dimuat").pack()
            return
        
        # Create notebook for tabs
        notebook = ttk.Notebook(self.content_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Preview tab
        preview_frame = ttk.Frame(notebook)
        notebook.add(preview_frame, text="Preview Data")
        
        # Treeview for data
        tree_frame = tk.Frame(preview_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add scrollbars
        tree_scroll_y = tk.Scrollbar(tree_frame)
        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        tree_scroll_x = tk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create treeview
        tree = ttk.Treeview(tree_frame, 
                           yscrollcommand=tree_scroll_y.set,
                           xscrollcommand=tree_scroll_x.set)
        tree.pack(fill=tk.BOTH, expand=True)
        
        tree_scroll_y.config(command=tree.yview)
        tree_scroll_x.config(command=tree.xview)
        
        # Define columns
        tree["columns"] = list(self.data.columns)
        tree["show"] = "headings"
        
        # Create headings
        for col in self.data.columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)
        
        # Add data
        for i, row in self.data.head(100).iterrows():  # Show first 100 rows
            tree.insert("", tk.END, values=list(row))
        
        # Info tab
        info_frame = ttk.Frame(notebook)
        notebook.add(info_frame, text="Data Info")
        
        info_text = tk.Text(info_frame, wrap=tk.NONE)
        info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        info = f"Shape: {self.data.shape}\n\n"
        info += f"Columns: {list(self.data.columns)}\n\n"
        info += "Data Types:\n"
        info += str(self.data.dtypes) + "\n\n"
        info += "Missing Values:\n"
        info += str(self.data.isnull().sum())
        
        info_text.insert(tk.END, info)
        info_text.config(state=tk.DISABLED)
        
        # Statistics tab
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="Statistics")
        
        stats_text = tk.Text(stats_frame, wrap=tk.NONE)
        stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats = self.data[numeric_cols].describe()
            stats_text.insert(tk.END, str(stats))
        else:
            stats_text.insert(tk.END, "No numeric columns found")
        
        stats_text.config(state=tk.DISABLED)
    
    def show_preprocessing(self):
        """Show preprocessing options (teacher only)"""
        if not self.is_teacher:
            messagebox.showwarning("Access Denied", "Hanya guru yang dapat mengakses fitur ini")
            return
        
        self.clear_content()
        
        if self.data is None:
            tk.Label(self.content_frame, text="Load data terlebih dahulu").pack()
            return
        
        # Preprocessing notebook
        notebook = ttk.Notebook(self.content_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # 1. Data Cleaning
        cleaning_frame = ttk.Frame(notebook)
        notebook.add(cleaning_frame, text="Pembersihan Data")
        
        tk.Label(cleaning_frame, text="Pembersihan Data", 
                font=("Arial", 14, "bold")).pack(pady=10)
        
        # Missing values
        missing_frame = tk.LabelFrame(cleaning_frame, text="Missing Values")
        missing_frame.pack(fill=tk.X, padx=10, pady=5)
        
        missing_info = tk.Text(missing_frame, height=5)
        missing_info.pack(fill=tk.X, padx=5, pady=5)
        missing_info.insert(tk.END, str(self.data.isnull().sum()))
        missing_info.config(state=tk.DISABLED)
        
        tk.Button(missing_frame, text="Hapus Rows dengan Missing Values",
                 command=self.remove_missing).pack(pady=5)
        tk.Button(missing_frame, text="Isi dengan Mean/Median",
                 command=self.fill_missing).pack(pady=5)
        
        # Duplicates
        dup_frame = tk.LabelFrame(cleaning_frame, text="Duplikat")
        dup_frame.pack(fill=tk.X, padx=10, pady=5)
        
        dup_count = len(self.data[self.data.duplicated()])
        tk.Label(dup_frame, text=f"Jumlah duplikat: {dup_count}").pack()
        
        tk.Button(dup_frame, text="Hapus Duplikat",
                 command=self.remove_duplicates).pack(pady=5)
        
        # Outliers
        outlier_frame = tk.LabelFrame(cleaning_frame, text="Outliers")
        outlier_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(outlier_frame, text="Deteksi Outliers (IQR Method)",
                 command=self.detect_outliers).pack(pady=5)
        tk.Button(outlier_frame, text="Hapus Outliers",
                 command=self.remove_outliers).pack(pady=5)
        
        # 2. Data Transformation
        transform_frame = ttk.Frame(notebook)
        notebook.add(transform_frame, text="Transformasi Data")
        
        tk.Label(transform_frame, text="Transformasi Data", 
                font=("Arial", 14, "bold")).pack(pady=10)
        
        # Encoding
        encode_frame = tk.LabelFrame(transform_frame, text="Encoding Kategorikal")
        encode_frame.pack(fill=tk.X, padx=10, pady=5)
        
        cat_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        if cat_cols:
            tk.Label(encode_frame, text=f"Kolom kategorikal: {cat_cols}").pack()
            
            self.encode_var = tk.StringVar(value=cat_cols[0] if cat_cols else "")
            encode_combo = ttk.Combobox(encode_frame, textvariable=self.encode_var,
                                       values=cat_cols)
            encode_combo.pack(pady=5)
            
            tk.Button(encode_frame, text="Label Encoding",
                     command=self.label_encode).pack(pady=2)
            tk.Button(encode_frame, text="One-Hot Encoding",
                     command=self.onehot_encode).pack(pady=2)
        else:
            tk.Label(encode_frame, text="Tidak ada kolom kategorikal").pack()
        
        # Normalization
        norm_frame = tk.LabelFrame(transform_frame, text="Normalisasi/Standarisasi")
        norm_frame.pack(fill=tk.X, padx=10, pady=5)
        
        num_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            tk.Label(norm_frame, text=f"Kolom numerik: {num_cols}").pack()
            
            self.norm_var = tk.StringVar(value=num_cols[0] if num_cols else "")
            norm_combo = ttk.Combobox(norm_frame, textvariable=self.norm_var,
                                     values=num_cols)
            norm_combo.pack(pady=5)
            
            tk.Button(norm_frame, text="StandardScaler",
                     command=self.standard_scaler).pack(pady=2)
            tk.Button(norm_frame, text="MinMaxScaler",
                     command=self.minmax_scaler).pack(pady=2)
        else:
            tk.Label(norm_frame, text="Tidak ada kolom numerik").pack()
        
        # 3. Data Reduction
        reduction_frame = ttk.Frame(notebook)
        notebook.add(reduction_frame, text="Reduksi Data")
        
        tk.Label(reduction_frame, text="Reduksi Data", 
                font=("Arial", 14, "bold")).pack(pady=10)
        
        # PCA
        pca_frame = tk.LabelFrame(reduction_frame, text="Principal Component Analysis (PCA)")
        pca_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.pca_var = tk.IntVar(value=2)
        tk.Label(pca_frame, text="Jumlah Komponen:").pack()
        tk.Entry(pca_frame, textvariable=self.pca_var, width=10).pack()
        
        tk.Button(pca_frame, text="Apply PCA",
                 command=self.apply_pca).pack(pady=5)
        
        # Feature Selection
        feat_frame = tk.LabelFrame(reduction_frame, text="Seleksi Fitur")
        feat_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(feat_frame, text="Importance-based Selection",
                 command=self.feature_selection).pack(pady=5)
        
        # 4. Data Splitting
        split_frame = ttk.Frame(notebook)
        notebook.add(split_frame, text="Pembagian Data")
        
        tk.Label(split_frame, text="Pembagian Data Train-Test", 
                font=("Arial", 14, "bold")).pack(pady=10)
        
        split_control = tk.Frame(split_frame)
        split_control.pack(pady=10)
        
        tk.Label(split_control, text="Test Size:").pack(side=tk.LEFT)
        self.test_size_var = tk.DoubleVar(value=0.2)
        tk.Entry(split_control, textvariable=self.test_size_var, width=10).pack(side=tk.LEFT, padx=5)
        
        tk.Label(split_control, text="Random State:").pack(side=tk.LEFT, padx=10)
        self.random_state_var = tk.IntVar(value=42)
        tk.Entry(split_control, textvariable=self.random_state_var, width=10).pack(side=tk.LEFT)
        
        tk.Button(split_frame, text="Split Data",
                 command=self.split_data,
                 width=20, height=2).pack(pady=10)
        
        if self.X_train is not None:
            info = f"Data sudah terbagi:\n"
            info += f"X_train shape: {self.X_train.shape}\n"
            info += f"X_test shape: {self.X_test.shape}\n"
            info += f"y_train shape: {self.y_train.shape}\n"
            info += f"y_test shape: {self.y_test.shape}"
            tk.Label(split_frame, text=info, justify=tk.LEFT).pack(pady=10)
    
    def remove_missing(self):
        """Remove rows with missing values"""
        if self.data is not None:
            original_len = len(self.data)
            self.data = self.data.dropna()
            new_len = len(self.data)
            messagebox.showinfo("Success", 
                              f"Dihapus {original_len - new_len} rows dengan missing values")
            self.show_preprocessing()
    
    def fill_missing(self):
        """Fill missing values with mean/median"""
        if self.data is not None:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if self.data[col].isnull().sum() > 0:
                    self.data[col].fillna(self.data[col].median(), inplace=True)
            
            # For categorical columns, fill with mode
            cat_cols = self.data.select_dtypes(include=['object']).columns
            for col in cat_cols:
                if self.data[col].isnull().sum() > 0:
                    self.data[col].fillna(self.data[col].mode()[0], inplace=True)
            
            messagebox.showinfo("Success", "Missing values telah diisi")
            self.show_preprocessing()
    
    def remove_duplicates(self):
        """Remove duplicate rows"""
        if self.data is not None:
            original_len = len(self.data)
            self.data = self.data.drop_duplicates()
            new_len = len(self.data)
            messagebox.showinfo("Success", 
                              f"Dihapus {original_len - new_len} duplikat")
            self.show_preprocessing()
    
    def detect_outliers(self):
        """Detect outliers using IQR method"""
        if self.data is None:
            return
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        outlier_info = "Outlier Detection:\n\n"
        
        for col in numeric_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]
            outlier_info += f"{col}: {len(outliers)} outliers\n"
        
        messagebox.showinfo("Outlier Detection", outlier_info)
    
    def remove_outliers(self):
        """Remove outliers using IQR method"""
        if self.data is None:
            return
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        original_len = len(self.data)
        
        for col in numeric_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            self.data = self.data[(self.data[col] >= lower_bound) & (self.data[col] <= upper_bound)]
        
        new_len = len(self.data)
        messagebox.showinfo("Success", 
                          f"Dihapus {original_len - new_len} outliers")
        self.show_preprocessing()
    
    def label_encode(self):
        """Apply label encoding to selected column"""
        if self.data is None:
            return
        
        col = self.encode_var.get()
        if col not in self.data.columns:
            return
        
        le = LabelEncoder()
        self.data[col] = le.fit_transform(self.data[col].astype(str))
        self.le_dict[col] = le
        
        messagebox.showinfo("Success", f"Label encoding applied to {col}")
        self.show_preprocessing()
    
    def onehot_encode(self):
        """Apply one-hot encoding to selected column"""
        if self.data is None:
            return
        
        col = self.encode_var.get()
        if col not in self.data.columns:
            return
        
        # Create dummy variables
        dummies = pd.get_dummies(self.data[col], prefix=col)
        
        # Add to dataframe and drop original column
        self.data = pd.concat([self.data, dummies], axis=1)
        self.data = self.data.drop(col, axis=1)
        
        messagebox.showinfo("Success", f"One-hot encoding applied to {col}")
        self.show_preprocessing()
    
    def standard_scaler(self):
        """Apply standard scaler to selected column"""
        if self.data is None:
            return
        
        col = self.norm_var.get()
        if col not in self.data.columns:
            return
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        self.data[col] = scaler.fit_transform(self.data[[col]])
        
        messagebox.showinfo("Success", f"StandardScaler applied to {col}")
        self.show_preprocessing()
    
    def minmax_scaler(self):
        """Apply min-max scaler to selected column"""
        if self.data is None:
            return
        
        col = self.norm_var.get()
        if col not in self.data.columns:
            return
        
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        self.data[col] = scaler.fit_transform(self.data[[col]])
        
        messagebox.showinfo("Success", f"MinMaxScaler applied to {col}")
        self.show_preprocessing()
    
    def apply_pca(self):
        """Apply PCA for dimensionality reduction"""
        if self.data is None:
            return
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            messagebox.showwarning("Warning", "Minimal 2 kolom numerik diperlukan untuk PCA")
            return
        
        n_components = self.pca_var.get()
        if n_components >= len(numeric_cols):
            messagebox.showwarning("Warning", "Jumlah komponen harus kurang dari jumlah fitur")
            return
        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        
        # Fit and transform
        pca_result = pca.fit_transform(self.data[numeric_cols])
        
        # Create new column names
        pca_cols = [f'PCA_{i+1}' for i in range(n_components)]
        
        # Create DataFrame with PCA results
        pca_df = pd.DataFrame(pca_result, columns=pca_cols)
        
        # Drop original numeric columns and add PCA columns
        self.data = self.data.drop(numeric_cols, axis=1)
        self.data = pd.concat([self.data, pca_df], axis=1)
        
        messagebox.showinfo("Success", 
                          f"PCA applied: {len(numeric_cols)} features → {n_components} components")
        self.show_preprocessing()
    
    def feature_selection(self):
        """Select important features using Random Forest"""
        if self.data is None:
            return
        
        # For simplicity, let's assume last column is target
        if len(self.data.columns) < 2:
            return
        
        target_col = self.data.columns[-1]
        X = self.data.drop(target_col, axis=1)
        y = self.data[target_col]
        
        # Only numeric features for RF importance
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            messagebox.showwarning("Warning", "Tidak ada fitur numerik untuk seleksi")
            return
        
        from sklearn.ensemble import RandomForestClassifier
        
        # Convert categorical target if needed
        if y.dtype == 'object':
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            rf = RandomForestClassifier()
        else:
            rf = RandomForestRegressor()
            y_encoded = y
        
        rf.fit(X[numeric_cols], y_encoded)
        
        # Get feature importances
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Show importance ranking
        importance_text = "Feature Importances:\n\n"
        for i, idx in enumerate(indices[:10]):  # Top 10 features
            importance_text += f"{i+1}. {numeric_cols[idx]}: {importances[idx]:.4f}\n"
        
        messagebox.showinfo("Feature Importance", importance_text)
        
        # Ask user how many features to keep
        top_n = tk.simpledialog.askinteger("Feature Selection", 
                                          "Berapa banyak fitur terbaik yang akan disimpan?",
                                          minvalue=1, maxvalue=len(numeric_cols))
        
        if top_n:
            # Select top N features
            top_features = numeric_cols[indices[:top_n]].tolist()
            
            # Keep only selected features plus any categorical columns
            cat_cols = X.select_dtypes(include=['object']).columns.tolist()
            cols_to_keep = top_features + cat_cols + [target_col]
            
            self.data = self.data[cols_to_keep]
            messagebox.showinfo("Success", f"Selected {top_n} most important features")
            self.show_preprocessing()
    
    def split_data(self):
        """Split data into train and test sets"""
        if self.data is None:
            return
        
        # Assume last column is target
        target_col = self.data.columns[-1]
        X = self.data.drop(target_col, axis=1)
        y = self.data[target_col]
        
        # Encode target if categorical
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        # Split data
        test_size = self.test_size_var.get()
        random_state = self.random_state_var.get()
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if len(set(y)) < 10 else None
        )
        
        # Scale features
        self.scaler.fit(self.X_train.select_dtypes(include=[np.number]))
        
        messagebox.showinfo("Success", 
                          f"Data split completed:\n"
                          f"Train: {len(self.X_train)} samples\n"
                          f"Test: {len(self.X_test)} samples")
        self.show_preprocessing()
    
    def show_analysis(self):
        """Show data analysis visualizations"""
        self.clear_content()
        
        if self.data is None:
            tk.Label(self.content_frame, text="Load data terlebih dahulu").pack()
            return
        
        notebook = ttk.Notebook(self.content_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Distribution Analysis
        dist_frame = ttk.Frame(notebook)
        notebook.add(dist_frame, text="Distribusi Data")
        
        # Select column for distribution
        dist_control = tk.Frame(dist_frame)
        dist_control.pack(pady=10)
        
        tk.Label(dist_control, text="Pilih Kolom:").pack(side=tk.LEFT)
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self.dist_col_var = tk.StringVar(value=numeric_cols[0] if numeric_cols else "")
        
        dist_combo = ttk.Combobox(dist_control, textvariable=self.dist_col_var,
                                 values=numeric_cols, width=30)
        dist_combo.pack(side=tk.LEFT, padx=5)
        
        tk.Button(dist_control, text="Tampilkan Histogram",
                 command=lambda: self.show_histogram(dist_frame)).pack(side=tk.LEFT, padx=5)
        
        tk.Button(dist_control, text="Tampilkan Box Plot",
                 command=lambda: self.show_boxplot(dist_frame)).pack(side=tk.LEFT, padx=5)
        
        # Correlation Analysis
        corr_frame = ttk.Frame(notebook)
        notebook.add(corr_frame, text="Korelasi")
        
        tk.Button(corr_frame, text="Tampilkan Heatmap Korelasi",
                 command=self.show_correlation).pack(pady=20)
        
        # Target Analysis
        target_frame = ttk.Frame(notebook)
        notebook.add(target_frame, text="Analisis Target")
        
        if len(self.data.columns) > 0:
            target_col = self.data.columns[-1]
            tk.Label(target_frame, text=f"Target Column: {target_col}", 
                    font=("Arial", 12, "bold")).pack(pady=10)
            
            tk.Button(target_frame, text="Distribusi Target",
                     command=self.show_target_distribution).pack(pady=5)
            
            # For categorical target
            if self.data[target_col].dtype == 'object' or len(self.data[target_col].unique()) < 10:
                tk.Button(target_frame, text="Count Plot Target",
                         command=self.show_target_count).pack(pady=5)
        
        # Feature Relationships
        rel_frame = ttk.Frame(notebook)
        notebook.add(rel_frame, text="Hubungan Fitur")
        
        tk.Button(rel_frame, text="Scatter Plot (2 Fitur)",
                 command=self.show_scatter).pack(pady=10)
        
        # AI Tools Usage Analysis
        ai_frame = ttk.Frame(notebook)
        notebook.add(ai_frame, text="Analisis AI Tools")
        
        tk.Button(ai_frame, text="Distribusi Penggunaan AI Tools",
                 command=self.show_ai_tools_dist).pack(pady=10)
        
        tk.Button(ai_frame, text="Impact vs Trust Analysis",
                 command=self.show_impact_trust).pack(pady=10)
    
    def show_histogram(self, parent):
        """Show histogram of selected column"""
        col = self.dist_col_var.get()
        if col not in self.data.columns:
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        self.data[col].hist(ax=ax, bins=30, edgecolor='black')
        ax.set_title(f'Distribusi {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frekuensi')
        
        self.display_plot(fig, parent)
    
    def show_boxplot(self, parent):
        """Show box plot of selected column"""
        col = self.dist_col_var.get()
        if col not in self.data.columns:
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        self.data[[col]].boxplot(ax=ax)
        ax.set_title(f'Box Plot {col}')
        
        self.display_plot(fig, parent)
    
    def show_correlation(self):
        """Show correlation heatmap"""
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            messagebox.showwarning("Warning", "Perlu minimal 2 kolom numerik untuk korelasi")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = numeric_data.corr()
        
        # Create heatmap
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Set labels
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr_matrix.columns)
        
        # Add correlation values
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="w")
        
        ax.set_title('Heatmap Korelasi')
        plt.tight_layout()
        
        # Create new window for plot
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Correlation Heatmap")
        
        canvas = FigureCanvasTkAgg(fig, plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_target_distribution(self):
        """Show target variable distribution"""
        target_col = self.data.columns[-1]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if self.data[target_col].dtype == 'object' or len(self.data[target_col].unique()) < 10:
            # For categorical target
            self.data[target_col].value_counts().plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
            ax.set_title(f'Distribusi Target ({target_col})')
            ax.set_xlabel(target_col)
            ax.set_ylabel('Count')
            plt.xticks(rotation=45)
        else:
            # For continuous target
            self.data[target_col].hist(ax=ax, bins=30, edgecolor='black', color='lightgreen')
            ax.set_title(f'Distribusi Target ({target_col})')
            ax.set_xlabel(target_col)
            ax.set_ylabel('Frekuensi')
        
        plt.tight_layout()
        
        # Create new window for plot
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Target Distribution")
        
        canvas = FigureCanvasTkAgg(fig, plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_target_count(self):
        """Show count plot for categorical target"""
        target_col = self.data.columns[-1]
        
        if self.data[target_col].dtype != 'object' and len(self.data[target_col].unique()) >= 10:
            messagebox.showinfo("Info", "Target bukan kategorikal")
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        self.data[target_col].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
        ax.set_title(f'Persentase Target ({target_col})')
        ax.set_ylabel('')
        
        # Create new window for plot
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Target Percentage")
        
        canvas = FigureCanvasTkAgg(fig, plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_scatter(self):
        """Show scatter plot between two features"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            messagebox.showwarning("Warning", "Perlu minimal 2 kolom numerik")
            return
        
        # Dialog to select two columns
        dialog = tk.Toplevel(self.root)
        dialog.title("Pilih Dua Fitur")
        dialog.geometry("300x150")
        
        tk.Label(dialog, text="Fitur X:").pack()
        x_var = tk.StringVar(value=numeric_cols[0])
        ttk.Combobox(dialog, textvariable=x_var, values=numeric_cols).pack()
        
        tk.Label(dialog, text="Fitur Y:").pack()
        y_var = tk.StringVar(value=numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0])
        ttk.Combobox(dialog, textvariable=y_var, values=numeric_cols).pack()
        
        def plot_scatter():
            x_col = x_var.get()
            y_col = y_var.get()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(self.data[x_col], self.data[y_col], alpha=0.6)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f'Scatter Plot: {x_col} vs {y_col}')
            
            # Add regression line
            try:
                z = np.polyfit(self.data[x_col], self.data[y_col], 1)
                p = np.poly1d(z)
                ax.plot(self.data[x_col], p(self.data[x_col]), "r--", alpha=0.8)
            except:
                pass
            
            plt.tight_layout()
            
            # Create new window for plot
            plot_window = tk.Toplevel(self.root)
            plot_window.title("Scatter Plot")
            
            canvas = FigureCanvasTkAgg(fig, plot_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            dialog.destroy()
        
        tk.Button(dialog, text="Plot", command=plot_scatter).pack(pady=10)
    
    def show_ai_tools_dist(self):
        """Show distribution of AI tools usage"""
        # Look for AI tools column (assuming column names contain 'AI' or 'Tools')
        ai_cols = [col for col in self.data.columns if any(keyword in col.lower() 
                   for keyword in ['ai', 'tool', 'usage', 'gemini', 'chatgpt', 'copilot'])]
        
        if not ai_cols:
            messagebox.showinfo("Info", "Kolom AI tools tidak ditemukan")
            return
        
        ai_col = ai_cols[0]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if self.data[ai_col].dtype == 'object' or len(self.data[ai_col].unique()) < 10:
            # Categorical AI tools
            self.data[ai_col].value_counts().plot(kind='bar', ax=ax, color='orange', edgecolor='black')
            ax.set_title('Distribusi Penggunaan AI Tools')
            ax.set_xlabel('AI Tools')
            ax.set_ylabel('Jumlah Pengguna')
            plt.xticks(rotation=45)
        else:
            # Numerical usage data
            self.data[ai_col].hist(ax=ax, bins=30, edgecolor='black', color='lightblue')
            ax.set_title('Distribusi Penggunaan AI Tools')
            ax.set_xlabel('Intensitas Penggunaan')
            ax.set_ylabel('Frekuensi')
        
        plt.tight_layout()
        
        # Create new window for plot
        plot_window = tk.Toplevel(self.root)
        plot_window.title("AI Tools Distribution")
        
        canvas = FigureCanvasTkAgg(fig, plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_impact_trust(self):
        """Show relationship between impact and trust"""
        # Look for impact and trust columns
        impact_cols = [col for col in self.data.columns if 'impact' in col.lower()]
        trust_cols = [col for col in self.data.columns if 'trust' in col.lower()]
        
        if not impact_cols or not trust_cols:
            messagebox.showinfo("Info", "Kolom impact atau trust tidak ditemukan")
            return
        
        impact_col = impact_cols[0]
        trust_col = trust_cols[0]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Scatter plot
        scatter = ax.scatter(self.data[trust_col], self.data[impact_col], 
                           alpha=0.6, c='purple')
        ax.set_xlabel(trust_col)
        ax.set_ylabel(impact_col)
        ax.set_title(f'Hubungan {trust_col} vs {impact_col}')
        
        # Add regression line
        try:
            z = np.polyfit(self.data[trust_col], self.data[impact_col], 1)
            p = np.poly1d(z)
            ax.plot(self.data[trust_col], p(self.data[trust_col]), "r--", alpha=0.8)
        except:
            pass
        
        plt.tight_layout()
        
        # Create new window for plot
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Impact vs Trust Analysis")
        
        canvas = FigureCanvasTkAgg(fig, plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def display_plot(self, fig, parent):
        """Display matplotlib figure in Tkinter frame"""
        # Clear previous plot
        for widget in parent.winfo_children():
            if widget not in [child for child in parent.winfo_children() 
                            if isinstance(child, tk.Frame)]:
                widget.destroy()
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def train_model(self):
        """Train Random Forest model"""
        if not self.is_teacher:
            messagebox.showwarning("Access Denied", "Hanya guru yang dapat melatih model")
            return
        
        if self.X_train is None or self.y_train is None:
            messagebox.showwarning("Warning", "Split data terlebih dahulu")
            return
        
        self.clear_content()
        
        tk.Label(self.content_frame, text="Training Random Forest Model", 
                font=("Arial", 14, "bold")).pack(pady=10)
        
        # Model configuration
        config_frame = tk.Frame(self.content_frame)
        config_frame.pack(pady=10)
        
        tk.Label(config_frame, text="n_estimators:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.n_estimators_var = tk.IntVar(value=100)
        tk.Entry(config_frame, textvariable=self.n_estimators_var, width=10).grid(row=0, column=1, padx=5)
        
        tk.Label(config_frame, text="max_depth:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.max_depth_var = tk.StringVar(value="None")
        tk.Entry(config_frame, textvariable=self.max_depth_var, width=10).grid(row=1, column=1, padx=5)
        
        tk.Label(config_frame, text="random_state:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.rf_random_state_var = tk.IntVar(value=42)
        tk.Entry(config_frame, textvariable=self.rf_random_state_var, width=10).grid(row=2, column=1, padx=5)
        
        # Check if classification or regression
        target_unique = len(np.unique(self.y_train))
        
        if target_unique < 10:  # Classification
            self.problem_type = "classification"
            tk.Label(config_frame, text="Problem Type: Classification").grid(row=3, column=0, columnspan=2, pady=5)
        else:  # Regression
            self.problem_type = "regression"
            tk.Label(config_frame, text="Problem Type: Regression").grid(row=3, column=0, columnspan=2, pady=5)
        
        # Train button
        tk.Button(self.content_frame, text="Train Model", 
                 command=self.execute_training,
                 width=20, height=2, bg="green", fg="white").pack(pady=20)
        
        # Progress/result area
        self.result_text = tk.Text(self.content_frame, height=15, width=80)
        self.result_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(self.result_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.result_text.yview)
    
    def execute_training(self):
        """Execute the model training"""
        try:
            # Get parameters
            n_estimators = self.n_estimators_var.get()
            max_depth = None if self.max_depth_var.get() == "None" else int(self.max_depth_var.get())
            random_state = self.rf_random_state_var.get()
            
            # Scale the features
            X_train_scaled = self.X_train.copy()
            X_test_scaled = self.X_test.copy()
            
            numeric_cols = self.X_train.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                X_train_scaled[numeric_cols] = self.scaler.transform(self.X_train[numeric_cols])
                X_test_scaled[numeric_cols] = self.scaler.transform(self.X_test[numeric_cols])
            
            # Train model
            if self.problem_type == "classification":
                self.model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=-1
                )
            else:
                self.model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=-1
                )
            
            self.model.fit(X_train_scaled, self.y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            
            # Display results
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "=== TRAINING RESULTS ===\n\n")
            self.result_text.insert(tk.END, f"Model: Random Forest ({self.problem_type})\n")
            self.result_text.insert(tk.END, f"n_estimators: {n_estimators}\n")
            self.result_text.insert(tk.END, f"max_depth: {max_depth}\n")
            self.result_text.insert(tk.END, f"Training samples: {len(self.X_train)}\n")
            self.result_text.insert(tk.END, f"Test samples: {len(self.X_test)}\n\n")
            
            if self.problem_type == "classification":
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
                
                self.result_text.insert(tk.END, "=== CLASSIFICATION METRICS ===\n")
                self.result_text.insert(tk.END, f"Accuracy: {accuracy:.4f}\n")
                self.result_text.insert(tk.END, f"Precision: {precision:.4f}\n")
                self.result_text.insert(tk.END, f"Recall: {recall:.4f}\n")
                self.result_text.insert(tk.END, f"F1-Score: {f1:.4f}\n")
                
                # Store metrics for later use
                self.metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
            else:
                mse = mean_squared_error(self.y_test, y_pred)
                mae = mean_absolute_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                
                self.result_text.insert(tk.END, "=== REGRESSION METRICS ===\n")
                self.result_text.insert(tk.END, f"MSE: {mse:.4f}\n")
                self.result_text.insert(tk.END, f"MAE: {mae:.4f}\n")
                self.result_text.insert(tk.END, f"RMSE: {rmse:.4f}\n")
                
                # Store metrics
                self.metrics = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse
                }
            
            # Feature importance
            self.result_text.insert(tk.END, "\n=== FEATURE IMPORTANCE ===\n")
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            for i, row in feature_importance.head(10).iterrows():
                self.result_text.insert(tk.END, f"{row['feature']}: {row['importance']:.4f}\n")
            
            self.status_var.set("Status: Model training completed successfully")
            
        except Exception as e:
            messagebox.showerror("Training Error", str(e))
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        if not self.is_teacher:
            messagebox.showwarning("Access Denied", "Hanya guru yang dapat mengevaluasi model")
            return
        
        if self.model is None:
            messagebox.showwarning("Warning", "Train model terlebih dahulu")
            return
        
        self.clear_content()
        
        tk.Label(self.content_frame, text="Model Evaluation", 
                font=("Arial", 14, "bold")).pack(pady=10)
        
        # Create notebook for different evaluation views
        notebook = ttk.Notebook(self.content_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Metrics tab
        metrics_frame = ttk.Frame(notebook)
        notebook.add(metrics_frame, text="Metrics")
        
        metrics_text = tk.Text(metrics_frame, wrap=tk.NONE)
        metrics_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scale test data
        X_test_scaled = self.X_test.copy()
        numeric_cols = self.X_test.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X_test_scaled[numeric_cols] = self.scaler.transform(self.X_test[numeric_cols])
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        if self.problem_type == "classification":
            # Detailed classification report
            report = classification_report(self.y_test, y_pred, output_dict=False)
            metrics_text.insert(tk.END, "=== CLASSIFICATION REPORT ===\n\n")
            metrics_text.insert(tk.END, report)
            
            # Confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            
            # Create confusion matrix visualization
            cm_frame = ttk.Frame(notebook)
            notebook.add(cm_frame, text="Confusion Matrix")
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            
            canvas = FigureCanvasTkAgg(fig, cm_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        else:
            # Regression metrics
            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            
            metrics_text.insert(tk.END, "=== REGRESSION METRICS ===\n\n")
            metrics_text.insert(tk.END, f"MSE: {mse:.4f}\n")
            metrics_text.insert(tk.END, f"MAE: {mae:.4f}\n")
            metrics_text.insert(tk.END, f"RMSE: {rmse:.4f}\n\n")
            
            # Residual plot
            residuals = self.y_test - y_pred
            
            residual_frame = ttk.Frame(notebook)
            notebook.add(residual_frame, text="Residual Analysis")
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Residuals vs Predicted
            axes[0].scatter(y_pred, residuals, alpha=0.6)
            axes[0].axhline(y=0, color='r', linestyle='--')
            axes[0].set_xlabel('Predicted Values')
            axes[0].set_ylabel('Residuals')
            axes[0].set_title('Residuals vs Predicted')
            
            # Histogram of residuals
            axes[1].hist(residuals, bins=30, edgecolor='black')
            axes[1].set_xlabel('Residuals')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Distribution of Residuals')
            
            plt.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, residual_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        metrics_text.config(state=tk.DISABLED)
        
        # Feature importance visualization
        if hasattr(self.model, 'feature_importances_'):
            feat_frame = ttk.Frame(notebook)
            notebook.add(feat_frame, text="Feature Importance")
            
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(range(len(feature_importance)), feature_importance['importance'])
            ax.set_yticks(range(len(feature_importance)))
            ax.set_yticklabels(feature_importance['feature'])
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance')
            
            plt.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, feat_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def predict_data(self):
        """Make predictions on new data (for students)"""
        self.clear_content()
        
        if self.model is None:
            tk.Label(self.content_frame, text="Model belum dilatih. Guru harus melatih model terlebih dahulu.").pack(pady=20)
            return
        
        tk.Label(self.content_frame, text="Prediksi Data Baru", 
                font=("Arial", 14, "bold")).pack(pady=10)
        
        # Create input form based on features
        input_frame = tk.Frame(self.content_frame)
        input_frame.pack(pady=10)
        
        self.input_vars = {}
        
        # Get feature names from training data
        for i, col in enumerate(self.X_train.columns):
            tk.Label(input_frame, text=f"{col}:").grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            
            # Get sample value for placeholder
            sample_val = self.X_train[col].iloc[0] if len(self.X_train) > 0 else 0
            
            if self.X_train[col].dtype == 'object' or len(self.X_train[col].unique()) < 10:
                # Categorical feature - use combobox
                unique_vals = self.X_train[col].unique()[:20]  # Limit to 20 values
                var = tk.StringVar(value=str(sample_val))
                combo = ttk.Combobox(input_frame, textvariable=var, values=list(unique_vals), width=30)
                combo.grid(row=i, column=1, padx=5, pady=2)
                self.input_vars[col] = var
            else:
                # Numerical feature - use entry
                var = tk.DoubleVar(value=float(sample_val) if pd.notna(sample_val) else 0)
                tk.Entry(input_frame, textvariable=var, width=30).grid(row=i, column=1, padx=5, pady=2)
                self.input_vars[col] = var
        
        # Predict button
        tk.Button(self.content_frame, text="Predict", 
                 command=self.make_prediction,
                 width=20, height=2, bg="blue", fg="white").pack(pady=20)
        
        # Result display
        self.prediction_result = tk.Text(self.content_frame, height=5, width=60)
        self.prediction_result.pack(pady=10, padx=10)
        self.prediction_result.insert(tk.END, "Hasil prediksi akan ditampilkan di sini...")
        self.prediction_result.config(state=tk.DISABLED)
        
        # Batch prediction option
        batch_frame = tk.Frame(self.content_frame)
        batch_frame.pack(pady=10)
        
        tk.Button(batch_frame, text="Prediksi dari File CSV", 
                 command=self.predict_from_file).pack(pady=5)
    
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
            
            # Create DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Handle categorical encoding
            for col in input_df.columns:
                if col in self.le_dict:
                    # Transform using saved label encoder
                    try:
                        input_df[col] = self.le_dict[col].transform(input_df[col].astype(str))
                    except:
                        # If new category, assign -1
                        input_df[col] = -1
            
            # Scale numerical features
            numeric_cols = self.X_train.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                input_df[numeric_cols] = self.scaler.transform(input_df[numeric_cols])
            
            # Ensure all columns are present
            for col in self.X_train.columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            input_df = input_df[self.X_train.columns]
            
            # Make prediction
            prediction = self.model.predict(input_df)
            
            # Display result
            self.prediction_result.config(state=tk.NORMAL)
            self.prediction_result.delete(1.0, tk.END)
            
            if self.problem_type == "classification":
                # For classification, show class probabilities if available
                self.prediction_result.insert(tk.END, f"Predicted Class: {prediction[0]}\n\n")
                
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(input_df)[0]
                    self.prediction_result.insert(tk.END, "Class Probabilities:\n")
                    for i, prob in enumerate(probabilities):
                        self.prediction_result.insert(tk.END, f"  Class {i}: {prob:.4f}\n")
            else:
                # For regression
                self.prediction_result.insert(tk.END, f"Predicted Value: {prediction[0]:.4f}\n")
            
            self.prediction_result.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
    
    def predict_from_file(self):
        """Make predictions from CSV file"""
        file_path = filedialog.askopenfilename(
            title="Pilih file data untuk prediksi",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Load data
                new_data = pd.read_csv(file_path)
                
                # Preprocess similar to training data
                processed_data = new_data.copy()
                
                # Handle categorical encoding
                for col in processed_data.columns:
                    if col in self.le_dict:
                        try:
                            processed_data[col] = self.le_dict[col].transform(processed_data[col].astype(str))
                        except:
                            processed_data[col] = -1
                
                # Scale numerical features
                numeric_cols = self.X_train.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    processed_data[numeric_cols] = self.scaler.transform(processed_data[numeric_cols])
                
                # Ensure all columns are present
                for col in self.X_train.columns:
                    if col not in processed_data.columns:
                        processed_data[col] = 0
                
                processed_data = processed_data[self.X_train.columns]
                
                # Make predictions
                predictions = self.model.predict(processed_data)
                
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
                    messagebox.showinfo("Success", f"Prediksi disimpan di: {save_path}")
                
            except Exception as e:
                messagebox.showerror("Prediction Error", str(e))
    
    def export_data(self):
        """Export data and results to CSV files"""
        if self.data is None:
            messagebox.showwarning("Warning", "Tidak ada data untuk diekspor")
            return
        
        # Create export directory
        import os
        export_dir = "export_results"
        os.makedirs(export_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export original data
        if self.data is not None:
            original_path = f"{export_dir}/original_data_{timestamp}.csv"
            self.data.to_csv(original_path, index=False)
        
        # Export preprocessed data (if split)
        if self.X_train is not None:
            train_data = self.X_train.copy()
            train_data['target'] = self.y_train
            train_data['set'] = 'train'
            
            test_data = self.X_test.copy()
            test_data['target'] = self.y_test
            test_data['set'] = 'test'
            
            combined_data = pd.concat([train_data, test_data], ignore_index=True)
            combined_path = f"{export_dir}/preprocessed_data_{timestamp}.csv"
            combined_data.to_csv(combined_path, index=False)
        
        # Export model predictions (if model exists)
        if self.model is not None:
            # Scale test data
            X_test_scaled = self.X_test.copy()
            numeric_cols = self.X_test.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                X_test_scaled[numeric_cols] = self.scaler.transform(self.X_test[numeric_cols])
            
            y_pred = self.model.predict(X_test_scaled)
            
            predictions_df = self.X_test.copy()
            predictions_df['Actual'] = self.y_test
            predictions_df['Predicted'] = y_pred
            
            if self.problem_type == "classification" and hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X_test_scaled)
                for i in range(proba.shape[1]):
                    predictions_df[f'Prob_Class_{i}'] = proba[:, i]
            
            predictions_path = f"{export_dir}/predictions_{timestamp}.csv"
            predictions_df.to_csv(predictions_path, index=False)
        
        # Export metrics (if available)
        if hasattr(self, 'metrics'):
            metrics_df = pd.DataFrame([self.metrics])
            metrics_path = f"{export_dir}/metrics_{timestamp}.csv"
            metrics_df.to_csv(metrics_path, index=False)
        
        # Export feature importance
        if self.model is not None and hasattr(self.model, 'feature_importances_'):
            feat_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            feat_path = f"{export_dir}/feature_importance_{timestamp}.csv"
            feat_importance.to_csv(feat_path, index=False)
        
        messagebox.showinfo("Export Success", 
                          f"Semua data berhasil diekspor ke folder: {export_dir}")
    
    def clear_content(self):
        """Clear content frame"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()

def main():
    root = tk.Tk()
    app = StudentAIAnalysisApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
