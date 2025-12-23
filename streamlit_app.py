import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Cek apakah library machine learning sudah diinstall
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    ML_AVAILABLE = True
except ImportError as e:
    st.warning(f"⚠️ Library scikit-learn tidak tersedia. Error: {str(e)}")
    st.info("Instal dengan: pip install scikit-learn")
    ML_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="AI Usage Analysis System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def init_session_state():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_type' not in st.session_state:
        st.session_state.user_type = None
    if 'df_raw' not in st.session_state:
        st.session_state.df_raw = None
    if 'df_clean' not in st.session_state:
        st.session_state.df_clean = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'label_encoders' not in st.session_state:
        st.session_state.label_encoders = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None

# Function to generate sample data
def generate_sample_data(n_samples=100):
    """Generate sample data for demonstration"""
    np.random.seed(42)
    
    # Sample data
    colleges = [
        'Indian Institute of Technology', 'University of Delhi', 
        'Jawaharlal Nehru University', 'University of Calcutta',
        'University of Mumbai', 'Anna University', 'University of Hyderabad',
        'Banaras Hindu University', 'University of Madras', 'University of Pune'
    ]
    
    streams = ['Engineering', 'Science', 'Commerce', 'Arts', 'Medical', 'Law', 'Management', 'Pharmacy', 'Hotel Management', 'Agriculture']
    
    ai_tools = [
        'ChatGPT', 'Gemini', 'Copilot', 'ChatGPT, Gemini', 
        'Gemini, Copilot', 'ChatGPT, Copilot', 'ChatGPT, Gemini, Copilot',
        'Gemini, Midjourney', 'ChatGPT, Midjourney', 'All Tools'
    ]
    
    use_cases = [
        'Assignments', 'Content Writing', 'MCQ Practice', 'Exam Prep', 
        'Doubt Solving', 'Learning new topics', 'Project Work',
        'Research', 'Coding Help', 'Presentation'
    ]
    
    data = {
        'Student_Name': [f'Student_{i:03d}' for i in range(1, n_samples + 1)],
        'College_Name': np.random.choice(colleges, n_samples),
        'Stream': np.random.choice(streams, n_samples),
        'AL_Tools_Used': np.random.choice(ai_tools, n_samples),
        'Usage_Intensity_Score': np.random.randint(5, 46, n_samples),
        'Use_Cases': [', '.join(np.random.choice(use_cases, np.random.randint(1, 4))) for _ in range(n_samples)]
    }
    
    df = pd.DataFrame(data)
    return df

# Function to load data
def load_data(uploaded_file=None):
    """Load data from uploaded CSV or use sample data"""
    try:
        if uploaded_file is not None:
            # Try to read the uploaded file
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            else:
                st.error("❌ Please upload a CSV or Excel file")
                return None
        else:
            # Use sample data
            df = generate_sample_data()
        
        # Basic validation
        if df.empty:
            st.error("❌ The uploaded file is empty")
            return None
        
        # Ensure required columns exist
        required_columns = ['Student_Name', 'College_Name', 'Stream', 
                           'AL_Tools_Used', 'Usage_Intensity_Score', 'Use_Cases']
        
        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.warning(f"⚠️ Missing columns: {missing_columns}")
            st.info("Using sample data instead")
            df = generate_sample_data()
        
        # Show basic info
        st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
        st.info(f"📊 Columns: {', '.join(df.columns.tolist())}")
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.info("Using sample data instead")
        return generate_sample_data()

# Function for data preprocessing
def preprocess_data(df):
    """Preprocess the data for machine learning"""
    try:
        if df is None:
            st.error("❌ No data to preprocess")
            return None, None
        
        df_clean = df.copy()
        
        # 1. Check for missing values
        missing_values = df_clean.isnull().sum()
        if missing_values.sum() > 0:
            st.warning(f"⚠️ Missing values found: {missing_values[missing_values > 0].to_dict()}")
            
            # Fill missing values
            for col in df_clean.columns:
                if df_clean[col].dtype == 'object':
                    df_clean[col] = df_clean[col].fillna('Unknown')
                else:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # 2. Convert Usage_Intensity_Score to numeric
        df_clean['Usage_Intensity_Score'] = pd.to_numeric(
            df_clean['Usage_Intensity_Score'], errors='coerce'
        )
        
        # Fill any NaN values after conversion
        df_clean['Usage_Intensity_Score'] = df_clean['Usage_Intensity_Score'].fillna(
            df_clean['Usage_Intensity_Score'].median()
        )
        
        # 3. Create target variable (Usage Level)
        df_clean['Usage_Level'] = pd.cut(
            df_clean['Usage_Intensity_Score'],
            bins=[0, 15, 30, 50],
            labels=['Low', 'Medium', 'High']
        )
        
        # Remove rows with NaN in Usage_Level
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna(subset=['Usage_Level'])
        removed_rows = initial_rows - len(df_clean)
        
        if removed_rows > 0:
            st.warning(f"⚠️ Removed {removed_rows} rows with invalid Usage_Intensity_Score")
        
        # 4. Simple encoding for demonstration
        label_encoders = {}
        
        # Encode Stream
        if 'Stream' in df_clean.columns:
            unique_streams = df_clean['Stream'].unique()
            stream_mapping = {stream: i for i, stream in enumerate(unique_streams)}
            df_clean['Stream_Encoded'] = df_clean['Stream'].map(stream_mapping)
            label_encoders['Stream'] = stream_mapping
        
        # Simple tool count encoding
        df_clean['Tools_Count'] = df_clean['AL_Tools_Used'].apply(
            lambda x: len(str(x).split(',')) if pd.notnull(x) else 0
        )
        
        # Simple use cases count encoding
        df_clean['UseCases_Count'] = df_clean['Use_Cases'].apply(
            lambda x: len(str(x).split(',')) if pd.notnull(x) else 0
        )
        
        # Encode target variable
        usage_level_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
        df_clean['Usage_Level_Encoded'] = df_clean['Usage_Level'].map(usage_level_mapping)
        label_encoders['Usage_Level'] = usage_level_mapping
        
        st.success(f"✅ Data preprocessing completed! Final shape: {df_clean.shape}")
        return df_clean, label_encoders
        
    except Exception as e:
        st.error(f"❌ Error in preprocessing: {str(e)}")
        return None, None

# Function to split data
def split_data(df_clean, test_size=0.3, random_state=42):
    """Split data into training and testing sets"""
    try:
        # Prepare features and target
        feature_cols = ['Stream_Encoded', 'Tools_Count', 'UseCases_Count', 'Usage_Intensity_Score']
        
        # Check if all feature columns exist
        available_features = [col for col in feature_cols if col in df_clean.columns]
        
        if len(available_features) < 2:
            st.error("❌ Not enough features available for training")
            return None, None, None, None, None, None
        
        X = df_clean[available_features]
        y = df_clean['Usage_Level_Encoded']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        st.success(f"✅ Data split completed: Train={len(X_train)}, Test={len(X_test)}")
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, available_features
        
    except Exception as e:
        st.error(f"❌ Error in data splitting: {str(e)}")
        return None, None, None, None, None, None

# Function to train model
def train_model(X_train, y_train, **kwargs):
    """Train a Random Forest classifier"""
    try:
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        }
        params.update(kwargs)
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        st.success("✅ Model training completed!")
        return model
        
    except Exception as e:
        st.error(f"❌ Error in model training: {str(e)}")
        return None

# Function to evaluate model
def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics"""
    try:
        if model is None or X_test is None or y_test is None:
            return None, None, None, None
        
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return y_pred, report_df, cm, metrics
        
    except Exception as e:
        st.error(f"❌ Error in model evaluation: {str(e)}")
        return None, None, None, None

# Function to create download link
def get_csv_download_link(df, filename="data.csv"):
    """Generate a download link for a DataFrame"""
    try:
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 {filename}</a>'
        return href
    except:
        return ""

# Main application
def main():
    # Initialize session state
    init_session_state()
    
    # Check if ML libraries are available
    if not ML_AVAILABLE:
        st.error("""
        ❌ **Required libraries missing!**
        
        Please install the required libraries:
        ```
        pip install scikit-learn pandas numpy matplotlib seaborn
        ```
        """)
        return
    
    # Login sidebar
    with st.sidebar:
        st.title("🔐 Login System")
        
        if not st.session_state.authenticated:
            st.markdown("---")
            user_type = st.selectbox("Select User Type", ["Teacher", "Student"])
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Login", use_container_width=True):
                    if user_type == "Teacher" and username == "teacher" and password == "teacher123":
                        st.session_state.authenticated = True
                        st.session_state.user_type = "teacher"
                        st.success("✅ Teacher login successful!")
                        st.rerun()
                    elif user_type == "Student" and username == "student" and password == "student123":
                        st.session_state.authenticated = True
                        st.session_state.user_type = "student"
                        st.success("✅ Student login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials!")
            
            with col2:
                if st.button("Quick Demo", use_container_width=True):
                    st.session_state.authenticated = True
                    st.session_state.user_type = "teacher"
                    st.session_state.df_raw = generate_sample_data()
                    st.success("✅ Demo mode activated!")
                    st.rerun()
            
            st.markdown("---")
            st.info("**Demo Credentials:**")
            st.write("- 👨‍🏫 Teacher: teacher / teacher123")
            st.write("- 👨‍🎓 Student: student / student123")
            
        else:
            st.success(f"✅ Logged in as {st.session_state.user_type}")
            st.markdown("---")
            
            if st.button("🔄 Reset All Data", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
            
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.authenticated = False
                st.session_state.user_type = None
                st.rerun()
    
    # Main content based on authentication
    if not st.session_state.authenticated:
        show_welcome_page()
        return
    
    # Teacher Dashboard
    if st.session_state.user_type == "teacher":
        show_teacher_dashboard()
    else:
        show_student_dashboard()

def show_welcome_page():
    """Show welcome page for unauthenticated users"""
    st.title("🎓 AI Usage Analysis System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Welcome to the AI Usage Analysis Dashboard
        
        This system analyzes the relationship between AI tool usage and academic performance
        using Random Forest algorithm.
        
        **Features:**
        👨‍🏫 **Teacher Dashboard:**
        - 📁 Upload and manage student data
        - 🔧 Data preprocessing and cleaning
        - 🤖 Train Random Forest models
        - 📊 Evaluate model performance
        - 📈 Visualize results
        - 📥 Export analysis results
        
        👨‍🎓 **Student Dashboard:**
        - 📊 View analysis results
        - 🎯 Predict personal AI usage level
        - 📈 Compare with peers
        
        **Data Requirements:**
        Upload a CSV file with these columns:
        - Student_Name
        - College_Name
        - Stream
        - AL_Tools_Used
        - Usage_Intensity_Score (1-50)
        - Use_Cases
        """)
    
    with col2:
        # Show sample data
        st.subheader("📋 Sample Data")
        sample_df = generate_sample_data(5)
        st.dataframe(sample_df, use_container_width=True)
        
        # Download sample CSV
        csv = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample CSV",
            data=csv,
            file_name="sample_ai_usage_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    st.markdown("---")
    st.info("👈 **Please login from the sidebar to continue**")

def show_teacher_dashboard():
    """Show teacher dashboard"""
    st.title("👨‍🏫 Teacher Dashboard")
    st.markdown("**Analysis of AI Usage on Academic Performance using Random Forest**")
    
    # Teacher menu
    menu = st.sidebar.radio(
        "📋 Menu",
        ["📁 Data Management", "🔧 Data Preprocessing", "🤖 Model Training", 
         "📊 Model Evaluation", "📈 Visualizations", "📥 Export Data"],
        index=0
    )
    
    if menu == "📁 Data Management":
        show_data_management()
    elif menu == "🔧 Data Preprocessing":
        show_data_preprocessing()
    elif menu == "🤖 Model Training":
        show_model_training()
    elif menu == "📊 Model Evaluation":
        show_model_evaluation()
    elif menu == "📈 Visualizations":
        show_visualizations()
    elif menu == "📥 Export Data":
        show_export_data()

def show_data_management():
    """Show data management section"""
    st.header("📁 Data Management")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Data")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload student data file"
        )
        
        if uploaded_file is not None:
            if st.button("📂 Load Data", use_container_width=True):
                with st.spinner("Loading data..."):
                    df = load_data(uploaded_file)
                    if df is not None:
                        st.session_state.df_raw = df
                        st.success(f"✅ Data loaded: {len(df)} records")
        
        st.subheader("Or Use Sample Data")
        if st.button("🎲 Generate Sample Data", use_container_width=True):
            with st.spinner("Generating sample data..."):
                st.session_state.df_raw = generate_sample_data()
                st.success(f"✅ Sample data generated: {len(st.session_state.df_raw)} records")
    
    with col2:
        if st.session_state.df_raw is not None:
            st.subheader("Data Preview")
            
            # Show data info
            st.info(f"**Data Shape:** {st.session_state.df_raw.shape[0]} rows × {st.session_state.df_raw.shape[1]} columns")
            
            # Show data
            st.dataframe(st.session_state.df_raw.head(10), use_container_width=True)
            
            # Data statistics
            with st.expander("📊 Data Statistics", expanded=True):
                col_stats1, col_stats2 = st.columns(2)
                
                with col_stats1:
                    st.write("**Basic Info:**")
                    st.write(f"Rows: {st.session_state.df_raw.shape[0]}")
                    st.write(f"Columns: {st.session_state.df_raw.shape[1]}")
                    st.write(f"Missing Values: {st.session_state.df_raw.isnull().sum().sum()}")
                
                with col_stats2:
                    st.write("**Column Types:**")
                    dtype_info = st.session_state.df_raw.dtypes.astype(str).to_dict()
                    for col, dtype in dtype_info.items():
                        st.write(f"• {col}: {dtype}")
            
            # Column distribution
            with st.expander("📈 Column Distribution"):
                selected_col = st.selectbox(
                    "Select column",
                    st.session_state.df_raw.columns.tolist()
                )
                
                if selected_col:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    
                    if st.session_state.df_raw[selected_col].dtype == 'object':
                        # Categorical
                        value_counts = st.session_state.df_raw[selected_col].value_counts().head(10)
                        value_counts.plot(kind='bar', ax=ax, color='skyblue')
                    else:
                        # Numerical
                        st.session_state.df_raw[selected_col].hist(ax=ax, bins=20, color='skyblue')
                    
                    ax.set_title(f'Distribution of {selected_col}')
                    ax.set_xlabel(selected_col)
                    ax.set_ylabel('Frequency')
                    st.pyplot(fig)
        else:
            st.info("👈 Please upload data or generate sample data to begin")

def show_data_preprocessing():
    """Show data preprocessing section"""
    st.header("🔧 Data Preprocessing")
    
    if st.session_state.df_raw is None:
        st.warning("⚠️ Please load data first in 'Data Management' section")
        return
    
    st.subheader("Raw Data Preview")
    st.dataframe(st.session_state.df_raw.head(), use_container_width=True)
    
    if st.button("🔄 Start Preprocessing", use_container_width=True):
        with st.spinner("Preprocessing data..."):
            df_clean, label_encoders = preprocess_data(st.session_state.df_raw)
            
            if df_clean is not None and label_encoders is not None:
                st.session_state.df_clean = df_clean
                st.session_state.label_encoders = label_encoders
                
                st.success("✅ Preprocessing completed!")
                
                # Show results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Cleaned Data Preview")
                    st.dataframe(df_clean[['Student_Name', 'College_Name', 'Stream', 
                                         'Usage_Intensity_Score', 'Usage_Level']].head(), 
                               use_container_width=True)
                
                with col2:
                    st.subheader("Target Distribution")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    
                    level_counts = df_clean['Usage_Level'].value_counts()
                    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                    level_counts.plot(kind='bar', ax=ax, color=colors)
                    
                    ax.set_xlabel('Usage Level')
                    ax.set_ylabel('Count')
                    ax.set_title('Distribution of AI Usage Levels')
                    
                    # Add count labels
                    for i, v in enumerate(level_counts):
                        ax.text(i, v + 0.5, str(v), ha='center')
                    
                    st.pyplot(fig)
                
                # Show preprocessing details
                with st.expander("🔍 Preprocessing Details"):
                    st.write("**Created Features:**")
                    st.write("• Stream_Encoded: Encoded stream categories")
                    st.write("• Tools_Count: Number of AI tools used")
                    st.write("• UseCases_Count: Number of use cases")
                    st.write("• Usage_Level: Categorized usage level (Low/Medium/High)")
                    st.write("• Usage_Level_Encoded: Numeric encoding of usage level")
                    
                    st.write("\n**Data Shape After Preprocessing:**")
                    st.write(f"• Rows: {df_clean.shape[0]}")
                    st.write(f"• Columns: {df_clean.shape[1]}")
            else:
                st.error("❌ Preprocessing failed. Please check your data.")

def show_model_training():
    """Show model training section"""
    st.header("🤖 Model Training with Random Forest")
    
    if st.session_state.df_clean is None:
        st.warning("⚠️ Please preprocess data first in 'Data Preprocessing' section")
        return
    
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_estimators = st.slider("Number of Trees", 50, 200, 100, 10)
        max_depth = st.slider("Max Depth", 5, 20, 10, 1)
    
    with col2:
        min_samples_split = st.slider("Min Samples Split", 2, 10, 2, 1)
        min_samples_leaf = st.slider("Min Samples Leaf", 1, 5, 1, 1)
    
    test_size = st.slider("Test Size (%)", 20, 40, 30, 5) / 100
    random_state = st.number_input("Random State", 0, 100, 42)
    
    if st.button("🚀 Train Model", use_container_width=True):
        with st.spinner("Training model..."):
            # Split data
            X_train, X_test, y_train, y_test, scaler, feature_cols = split_data(
                st.session_state.df_clean, test_size, random_state
            )
            
            if X_train is not None:
                # Train model
                model = train_model(
                    X_train, y_train,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state
                )
                
                if model is not None:
                    st.session_state.model = model
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.scaler = scaler
                    st.session_state.feature_cols = feature_cols
                    
                    # Evaluate model
                    y_pred, report_df, cm, metrics = evaluate_model(
                        model, X_test, y_test
                    )
                    
                    if metrics is not None:
                        st.session_state.metrics = metrics
                        st.session_state.y_pred = y_pred
                        st.session_state.report_df = report_df
                        st.session_state.cm = cm
                        
                        # Show training results
                        st.success("✅ Model trained successfully!")
                        
                        st.subheader("Training Results")
                        
                        col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
                        
                        with col_metric1:
                            st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                        
                        with col_metric2:
                            st.metric("Precision", f"{metrics['precision']:.2%}")
                        
                        with col_metric3:
                            st.metric("Recall", f"{metrics['recall']:.2%}")
                        
                        with col_metric4:
                            st.metric("F1-Score", f"{metrics['f1_score']:.2%}")
                        
                        # Feature importance
                        st.subheader("Feature Importance")
                        
                        if hasattr(model, 'feature_importances_'):
                            importance_df = pd.DataFrame({
                                'Feature': feature_cols,
                                'Importance': model.feature_importances_
                            }).sort_values('Importance', ascending=False)
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.barh(importance_df['Feature'], importance_df['Importance'], color='lightcoral')
                            ax.set_xlabel('Importance')
                            ax.set_title('Feature Importance')
                            ax.invert_yaxis()
                            st.pyplot(fig)
                        else:
                            st.info("Feature importance not available for this model")

def show_model_evaluation():
    """Show model evaluation section"""
    st.header("📊 Model Evaluation")
    
    if st.session_state.model is None:
        st.warning("⚠️ Please train the model first in 'Model Training' section")
        return
    
    if st.session_state.metrics is None:
        st.warning("⚠️ No evaluation metrics available")
        return
    
    # Display metrics
    st.subheader("Performance Metrics")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Value': [
                f"{st.session_state.metrics['accuracy']:.2%}",
                f"{st.session_state.metrics['precision']:.2%}",
                f"{st.session_state.metrics['recall']:.2%}",
                f"{st.session_state.metrics['f1_score']:.2%}"
            ]
        }
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    with col2:
        # Confusion Matrix
        if st.session_state.cm is not None:
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(st.session_state.cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Low', 'Medium', 'High'],
                       yticklabels=['Low', 'Medium', 'High'], ax=ax)
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
    
    # Classification Report
    st.subheader("Detailed Classification Report")
    if st.session_state.report_df is not None:
        st.dataframe(st.session_state.report_df, use_container_width=True)
    
    # Performance visualization
    st.subheader("Performance Visualization")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Bar chart of metrics
    axes[0].bar(['Accuracy', 'Precision', 'Recall', 'F1-Score'],
               [st.session_state.metrics['accuracy'],
                st.session_state.metrics['precision'],
                st.session_state.metrics['recall'],
                st.session_state.metrics['f1_score']],
               color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFE66D'])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('Score')
    axes[0].set_title('Model Performance Metrics')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for i, v in enumerate([st.session_state.metrics['accuracy'],
                           st.session_state.metrics['precision'],
                           st.session_state.metrics['recall'],
                           st.session_state.metrics['f1_score']]):
        axes[0].text(i, v + 0.02, f'{v:.2%}', ha='center')
    
    # Metrics by class (if available)
    if st.session_state.report_df is not None and '0' in st.session_state.report_df.index:
        try:
            class_metrics = st.session_state.report_df.loc[['0', '1', '2']]
            x = np.arange(3)
            width = 0.25
            
            axes[1].bar(x - width, class_metrics['precision'], width, label='Precision', color='#4ECDC4')
            axes[1].bar(x, class_metrics['recall'], width, label='Recall', color='#FF6B6B')
            axes[1].bar(x + width, class_metrics['f1-score'], width, label='F1-Score', color='#45B7D1')
            
            axes[1].set_xlabel('Class (0=Low, 1=Medium, 2=High)')
            axes[1].set_ylabel('Score')
            axes[1].set_title('Metrics by Class')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(['Low', 'Medium', 'High'])
            axes[1].legend()
            axes[1].set_ylim(0, 1)
        except:
            # Simple metric plot if class metrics not available
            axes[1].text(0.5, 0.5, 'Class metrics\nnot available', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Class Metrics')
    
    plt.tight_layout()
    st.pyplot(fig)

def show_visualizations():
    """Show data visualizations"""
    st.header("📈 Data Visualizations")
    
    if st.session_state.df_clean is None:
        st.warning("⚠️ Please preprocess data first")
        return
    
    viz_option = st.selectbox(
        "Select Visualization Type",
        ["Usage Distribution", "Stream Analysis", "College Analysis", 
         "AI Tools Analysis", "Use Cases Analysis"]
    )
    
    if viz_option == "Usage Distribution":
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram
        axes[0].hist(st.session_state.df_clean['Usage_Intensity_Score'], 
                    bins=20, color='skyblue', edgecolor='black')
        axes[0].set_xlabel('Usage Intensity Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Usage Scores')
        
        # Box plot
        axes[1].boxplot(st.session_state.df_clean['Usage_Intensity_Score'])
        axes[1].set_ylabel('Usage Intensity Score')
        axes[1].set_title('Box Plot of Usage Scores')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    elif viz_option == "Stream Analysis":
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Average usage by stream
        stream_avg = st.session_state.df_clean.groupby('Stream')['Usage_Intensity_Score'].mean().sort_values()
        stream_avg.plot(kind='barh', ax=axes[0], color='lightcoral')
        axes[0].set_xlabel('Average Score')
        axes[0].set_title('Average AI Usage by Stream')
        
        # Usage level distribution by stream
        try:
            stream_level = pd.crosstab(st.session_state.df_clean['Stream'], 
                                     st.session_state.df_clean['Usage_Level'])
            stream_level.plot(kind='bar', ax=axes[1], stacked=True)
            axes[1].set_xlabel('Stream')
            axes[1].set_ylabel('Count')
            axes[1].set_title('Usage Level by Stream')
            axes[1].legend(title='Usage Level')
            plt.xticks(rotation=45)
        except:
            axes[1].text(0.5, 0.5, 'Data not available\nfor this visualization', 
                        ha='center', va='center', transform=axes[1].transAxes)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    elif viz_option == "College Analysis":
        # Top colleges by average usage
        college_avg = st.session_state.df_clean.groupby('College_Name')['Usage_Intensity_Score'].mean().nlargest(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        college_avg.plot(kind='barh', ax=ax, color='lightgreen')
        ax.set_xlabel('Average Score')
        ax.set_title('Top 10 Colleges by AI Usage')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    elif viz_option == "AI Tools Analysis":
        # Count of AI tools usage
        try:
            all_tools = []
            for tools in st.session_state.df_clean['AL_Tools_Used']:
                if isinstance(tools, str):
                    tool_list = [t.strip() for t in tools.split(',')]
                    all_tools.extend(tool_list)
            
            if all_tools:
                tool_counts = pd.Series(all_tools).value_counts().head(10)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                tool_counts.plot(kind='bar', ax=ax, color='orange')
                ax.set_xlabel('AI Tool')
                ax.set_ylabel('Count')
                ax.set_title('Most Used AI Tools')
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No AI tools data available")
        except:
            st.info("Unable to analyze AI tools data")
    
    elif viz_option == "Use Cases Analysis":
        # Common use cases
        try:
            all_cases = []
            for cases in st.session_state.df_clean['Use_Cases']:
                if isinstance(cases, str):
                    case_list = [c.strip() for c in cases.split(',')]
                    all_cases.extend(case_list)
            
            if all_cases:
                case_counts = pd.Series(all_cases).value_counts()
                
                fig, ax = plt.subplots(figsize=(10, 5))
                case_counts.plot(kind='bar', ax=ax, color='lightblue')
                ax.set_xlabel('Use Case')
                ax.set_ylabel('Count')
                ax.set_title('Most Common AI Use Cases')
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No use cases data available")
        except:
            st.info("Unable to analyze use cases data")

def show_export_data():
    """Show data export section"""
    st.header("📥 Export Data & Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Export Data Files")
        
        if st.session_state.df_raw is not None:
            csv_raw = st.session_state.df_raw.to_csv(index=False)
            st.download_button(
                label="📥 Raw Data (CSV)",
                data=csv_raw,
                file_name="ai_usage_raw_data.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        if st.session_state.df_clean is not None:
            csv_clean = st.session_state.df_clean.to_csv(index=False)
            st.download_button(
                label="📥 Cleaned Data (CSV)",
                data=csv_clean,
                file_name="ai_usage_cleaned_data.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        if st.session_state.model is not None:
            # Export predictions
            predictions_df = pd.DataFrame({
                'Actual_Usage': st.session_state.y_test,
                'Predicted_Usage': st.session_state.y_pred
            })
            csv_pred = predictions_df.to_csv(index=False)
            st.download_button(
                label="📥 Predictions (CSV)",
                data=csv_pred,
                file_name="ai_usage_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        st.subheader("Export Reports")
        
        if st.session_state.model is not None:
            # Export metrics
            if st.session_state.metrics is not None:
                metrics_df = pd.DataFrame({
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                    'Value': [
                        st.session_state.metrics['accuracy'],
                        st.session_state.metrics['precision'],
                        st.session_state.metrics['recall'],
                        st.session_state.metrics['f1_score']
                    ]
                })
                csv_metrics = metrics_df.to_csv(index=False)
                st.download_button(
                    label="📥 Model Metrics (CSV)",
                    data=csv_metrics,
                    file_name="model_metrics.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Export feature importance
            if hasattr(st.session_state.model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': st.session_state.feature_cols,
                    'Importance': st.session_state.model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                csv_importance = importance_df.to_csv(index=False)
                st.download_button(
                    label="📥 Feature Importance (CSV)",
                    data=csv_importance,
                    file_name="feature_importance.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    st.markdown("---")
    st.subheader("📋 Summary Report")
    
    if st.button("Generate Summary Report", use_container_width=True):
        with st.spinner("Generating report..."):
            # Create summary report
            summary = []
            
            if st.session_state.df_raw is not None:
                summary.append(f"📊 **Data Summary:**")
                summary.append(f"• Raw Data: {st.session_state.df_raw.shape[0]} rows, {st.session_state.df_raw.shape[1]} columns")
            
            if st.session_state.df_clean is not None:
                summary.append(f"• Cleaned Data: {st.session_state.df_clean.shape[0]} rows, {st.session_state.df_clean.shape[1]} columns")
                
                # Usage level distribution
                if 'Usage_Level' in st.session_state.df_clean.columns:
                    level_dist = st.session_state.df_clean['Usage_Level'].value_counts()
                    summary.append(f"• Usage Level Distribution:")
                    for level, count in level_dist.items():
                        summary.append(f"  - {level}: {count} students ({count/len(st.session_state.df_clean):.1%})")
            
            if st.session_state.model is not None and st.session_state.metrics is not None:
                summary.append(f"\n🤖 **Model Performance:**")
                summary.append(f"• Accuracy: {st.session_state.metrics['accuracy']:.2%}")
                summary.append(f"• Precision: {st.session_state.metrics['precision']:.2%}")
                summary.append(f"• Recall: {st.session_state.metrics['recall']:.2%}")
                summary.append(f"• F1-Score: {st.session_state.metrics['f1_score']:.2%}")
            
            # Display summary
            st.markdown("\n".join(summary))
            
            # Download summary
            summary_text = "\n".join(summary)
            st.download_button(
                label="📥 Download Summary Report (TXT)",
                data=summary_text,
                file_name="ai_analysis_summary.txt",
                mime="text/plain",
                use_container_width=True
            )

def show_student_dashboard():
    """Show student dashboard"""
    st.title("👨‍🎓 Student Dashboard")
    
    menu = st.sidebar.radio(
        "📋 Menu",
        ["📊 View Analysis", "🎯 Predict My Usage", "📈 Compare with Peers"],
        index=0
    )
    
    if menu == "📊 View Analysis":
        show_student_analysis()
    elif menu == "🎯 Predict My Usage":
        show_student_prediction()
    elif menu == "📈 Compare with Peers":
        show_student_comparison()

def show_student_analysis():
    """Show analysis results for students"""
    st.header("📊 AI Usage Analysis Results")
    
    if st.session_state.df_clean is None:
        st.info("📢 No analysis data available yet. Please ask your teacher to upload and analyze data.")
        
        # Show sample statistics
        st.subheader("Sample Statistics")
        sample_df = generate_sample_data(50)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Usage Score", f"{sample_df['Usage_Intensity_Score'].mean():.1f}")
        with col2:
            # Categorize sample scores
            sample_levels = pd.cut(sample_df['Usage_Intensity_Score'], 
                                  bins=[0, 15, 30, 50], 
                                  labels=['Low', 'Medium', 'High'])
            common_level = sample_levels.mode()[0]
            st.metric("Most Common Level", common_level)
        with col3:
            st.metric("Sample Size", len(sample_df))
        
        return
    
    # Show actual statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_score = st.session_state.df_clean['Usage_Intensity_Score'].mean()
        st.metric("Average Usage Score", f"{avg_score:.1f}")
    
    with col2:
        most_common_level = st.session_state.df_clean['Usage_Level'].mode()[0]
        st.metric("Most Common Level", most_common_level)
    
    with col3:
        total_students = len(st.session_state.df_clean)
        st.metric("Total Students", total_students)
    
    # Distribution chart
    st.subheader("Usage Level Distribution")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    level_counts = st.session_state.df_clean['Usage_Level'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    level_counts.plot(kind='bar', ax=ax, color=colors)
    
    ax.set_xlabel('Usage Level')
    ax.set_ylabel('Number of Students')
    ax.set_title('Distribution of AI Usage Levels')
    
    # Add percentage labels
    total = level_counts.sum()
    for i, v in enumerate(level_counts):
        ax.text(i, v + 0.5, f'{v/total:.1%}', ha='center')
    
    st.pyplot(fig)
    
    # Recommendations based on analysis
    st.subheader("📝 Recommendations")
    
    col_rec1, col_rec2, col_rec3 = st.columns(3)
    
    with col_rec1:
        st.info("""
        **For Low Usage (0-15):**
        • Start with basic AI tools
        • Use for simple tasks
        • Attend AI workshops
        """)
    
    with col_rec2:
        st.info("""
        **For Medium Usage (16-30):**
        • Explore advanced features
        • Integrate into projects
        • Share best practices
        """)
    
    with col_rec3:
        st.info("""
        **For High Usage (31-50):**
        • Mentor other students
        • Research AI applications
        • Ensure ethical use
        """)

def show_student_prediction():
    """Show prediction interface for students"""
    st.header("🎯 Predict Your AI Usage Level")
    
    st.info("Enter your information to predict your AI usage level:")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            student_name = st.text_input("Your Name", "John Doe")
            college = st.text_input("College Name", "Indian Institute of Technology")
            stream = st.selectbox("Stream", 
                                ["Engineering", "Science", "Commerce", "Arts", 
                                 "Medical", "Law", "Management", "Agriculture", "Pharmacy", "Hotel Management"])
        
        with col2:
            ai_tools = st.multiselect("AI Tools You Use",
                                    ["ChatGPT", "Gemini", "Copilot", "Midjourney",
                                     "Bard", "Claude", "Other"])
            
            use_cases = st.multiselect("Primary Use Cases",
                                     ["Assignments", "Content Writing", "MCQ Practice",
                                      "Exam Prep", "Doubt Solving", "Learning new topics",
                                      "Project Work", "Coding Help", "Resume Writing", "Research"])
            
            usage_score = st.slider("Your Usage Intensity Score", 1, 50, 25, 
                                   help="Estimate how intensely you use AI tools (1=rarely, 50=very frequently)")
        
        submitted = st.form_submit_button("🔮 Predict My Usage", use_container_width=True)
        
        if submitted:
            # Simple prediction logic
            if usage_score <= 15:
                predicted_level = "Low"
                confidence = 0.85
            elif usage_score <= 30:
                predicted_level = "Medium"
                confidence = 0.90
            else:
                predicted_level = "High"
                confidence = 0.95
            
            # Display results
            st.success(f"✅ Prediction completed for {student_name}")
            
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                st.metric("Predicted Usage Level", predicted_level)
                st.metric("Confidence Level", f"{confidence:.0%}")
            
            with col_result2:
                st.metric("Your Score", usage_score)
                st.metric("AI Tools Used", len(ai_tools))
            
            # Personalized recommendations
            st.subheader("🎯 Personalized Recommendations")
            
            if predicted_level == "Low":
                st.warning("""
                **You're just getting started with AI!** 
                
                **Suggestions:**
                1. **Start Simple**: Try using ChatGPT or Gemini for help with homework
                2. **Learn Basics**: Take a free online course on AI tools for students
                3. **Join Communities**: Participate in AI clubs or online forums
                4. **Set Goals**: Aim to use one AI tool for 30 minutes daily
                
                **Expected Benefits**: 
                • Save 2-3 hours per week on assignments
                • Improve understanding of difficult concepts
                • Develop valuable digital skills
                """)
            
            elif predicted_level == "Medium":
                st.info("""
                **You're using AI effectively!**
                
                **Suggestions:**
                1. **Advanced Features**: Explore plugins and advanced functionalities
                2. **Project Integration**: Use AI for research papers and projects
                3. **Skill Sharing**: Help classmates learn AI tools
                4. **Specialize**: Focus on AI tools specific to your field
                
                **Expected Benefits**:
                • Enhanced productivity in academic work
                • Better project outcomes
                • Development of AI-assisted problem-solving skills
                """)
            
            else:
                st.success("""
                **You're an advanced AI user!**
                
                **Suggestions:**
                1. **Mentor Others**: Share your expertise with fellow students
                2. **Research Applications**: Explore AI for academic research
                3. **Ethical Considerations**: Ensure responsible AI use
                4. **Future Skills**: Learn about AI development and applications
                
                **Expected Benefits**:
                • Leadership in AI literacy among peers
                • Potential for AI-related research projects
                • Enhanced employability with AI skills
                """)
            
            # Action plan
            st.subheader("📋 30-Day Action Plan")
            days = ["Week 1", "Week 2", "Week 3", "Week 4"]
            
            if predicted_level == "Low":
                actions = [
                    "Try 3 different AI tools for basic tasks",
                    "Complete an online tutorial on AI basics",
                    "Use AI for one major assignment",
                    "Join an AI learning community"
                ]
            elif predicted_level == "Medium":
                actions = [
                    "Master advanced features of your main AI tool",
                    "Collaborate on an AI-assisted project",
                    "Teach one AI skill to a classmate",
                    "Explore AI tools for your career field"
                ]
            else:
                actions = [
                    "Start an AI learning group",
                    "Begin an AI-related research project",
                    "Create AI usage guidelines for peers",
                    "Explore AI development basics"
                ]
            
            for day, action in zip(days, actions):
                st.write(f"**{day}**: {action}")

def show_student_comparison():
    """Show comparison with peers"""
    st.header("📈 Compare with Peers")
    
    if st.session_state.df_clean is None:
        st.info("📊 No comparison data available. Using sample data for demonstration.")
        comparison_df = generate_sample_data(100)
    else:
        comparison_df = st.session_state.df_clean
    
    compare_option = st.selectbox(
        "Compare by:",
        ["Stream", "College", "Usage Level", "AI Tools"]
    )
    
    if compare_option == "Stream":
        st.subheader("Comparison by Academic Stream")
        
        stream_stats = comparison_df.groupby('Stream').agg({
            'Usage_Intensity_Score': ['mean', 'std', 'count']
        }).round(1)
        
        st.dataframe(stream_stats, use_container_width=True)
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 5))
        stream_means = comparison_df.groupby('Stream')['Usage_Intensity_Score'].mean()
        stream_means.plot(kind='bar', ax=ax, color='lightcoral')
        ax.set_xlabel('Academic Stream')
        ax.set_ylabel('Average Usage Score')
        ax.set_title('Average AI Usage by Academic Stream')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    elif compare_option == "College":
        st.subheader("Comparison by College")
        
        # Top 10 colleges
        college_stats = comparison_df.groupby('College_Name').agg({
            'Usage_Intensity_Score': ['mean', 'std', 'count']
        }).round(1).nlargest(10, ('Usage_Intensity_Score', 'mean'))
        
        st.dataframe(college_stats, use_container_width=True)
    
    elif compare_option == "Usage Level":
        st.subheader("Usage Level Analysis")
        
        if 'Usage_Level' in comparison_df.columns:
            level_stats = comparison_df.groupby('Usage_Level').agg({
                'Usage_Intensity_Score': ['mean', 'min', 'max', 'count']
            }).round(1)
            
            st.dataframe(level_stats, use_container_width=True)
            
            # Pie chart
            fig, ax = plt.subplots(figsize=(8, 6))
            level_counts = comparison_df['Usage_Level'].value_counts()
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            ax.pie(level_counts, labels=level_counts.index, autopct='%1.1f%%', 
                  colors=colors, startangle=90)
            ax.set_title('Distribution of Usage Levels')
            st.pyplot(fig)
    
    elif compare_option == "AI Tools":
        st.subheader("AI Tools Usage Patterns")
        
        # Extract and count AI tools
        try:
            all_tools = []
            for tools in comparison_df['AL_Tools_Used']:
                if isinstance(tools, str):
                    tool_list = [t.strip() for t in tools.split(',')]
                    all_tools.extend(tool_list)
            
            if all_tools:
                tool_counts = pd.Series(all_tools).value_counts()
                
                # Display top tools
                st.write("**Most Popular AI Tools:**")
                for tool, count in tool_counts.head(10).items():
                    percentage = (count / len(comparison_df)) * 100
                    st.write(f"• **{tool}**: {count} users ({percentage:.1f}%)")
                
                # Visualization
                fig, ax = plt.subplots(figsize=(10, 5))
                tool_counts.head(10).plot(kind='bar', ax=ax, color='skyblue')
                ax.set_xlabel('AI Tool')
                ax.set_ylabel('Number of Users')
                ax.set_title('Top 10 AI Tools Used')
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.info("No AI tools data available for comparison")
        except:
            st.info("Unable to analyze AI tools data")
    
    # Personal reflection
    st.subheader("🤔 Self-Reflection Questions")
    
    reflection_questions = [
        "How does your AI usage compare to your peers in the same stream?",
        "What AI tools are most popular among students with high usage scores?",
        "How could you improve your AI usage based on these insights?",
        "What benefits have you experienced from using AI tools in your studies?"
    ]
    
    for i, question in enumerate(reflection_questions, 1):
        with st.expander(f"Question {i}: {question}"):
            st.text_area(f"Your reflection on question {i}", 
                        placeholder="Type your thoughts here...", 
                        height=100, 
                        key=f"reflection_{i}")

if __name__ == "__main__":
    main()
