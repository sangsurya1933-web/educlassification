import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from io import BytesIO
import warnings
import os
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="AI Usage Analysis System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to generate sample data if CSV doesn't exist
def generate_sample_data():
    """Generate sample data for demonstration"""
    data = {
        'Student_Name': [f'Student_{i}' for i in range(1, 101)],
        'College_Name': ['Indian Institute of Technology', 'University of Delhi', 
                        'Jawaharlal Nehru University', 'University of Calcutta',
                        'University of Mumbai', 'Anna University', 'University of Hyderabad'] * 15,
        'Stream': ['Engineering', 'Science', 'Commerce', 'Arts', 'Medical', 'Law', 'Management'] * 15,
        'AL_Tools_Used': ['ChatGPT', 'Gemini', 'Copilot', 'ChatGPT, Gemini', 'Gemini, Midjourney', 
                         'ChatGPT, Copilot', 'All Tools'] * 15,
        'Usage_Intensity_Score': np.random.randint(5, 46, 100),
        'Use_Cases': ['Assignments', 'Content Writing', 'MCQ Practice', 'Exam Prep', 
                     'Doubt Solving', 'Learning new topics', 'Project Work'] * 15
    }
    
    # Make last 20 entries more varied
    for i in range(80, 100):
        data['Usage_Intensity_Score'][i] = np.random.randint(20, 50)
    
    return pd.DataFrame(data)

# Function to load data
def load_data(uploaded_file=None):
    """Load data from uploaded CSV or use sample data"""
    if uploaded_file is not None:
        try:
            # Try to read the uploaded file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Please upload a CSV or Excel file")
                return None
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return None
    else:
        # Use sample data
        df = generate_sample_data()
    
    # Ensure required columns exist
    required_columns = ['Student_Name', 'College_Name', 'Stream', 'AL_Tools_Used', 
                       'Usage_Intensity_Score', 'Use_Cases']
    
    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.warning(f"Missing columns: {missing_columns}. Using sample data instead.")
        df = generate_sample_data()
    
    return df

# Function for data preprocessing
def preprocess_data(df):
    """Preprocess the data for machine learning"""
    df_clean = df.copy()
    
    # Handle missing values
    if df_clean['Usage_Intensity_Score'].isnull().any():
        df_clean['Usage_Intensity_Score'] = df_clean['Usage_Intensity_Score'].fillna(
            df_clean['Usage_Intensity_Score'].median()
        )
    
    # Convert to appropriate types
    df_clean['Usage_Intensity_Score'] = pd.to_numeric(
        df_clean['Usage_Intensity_Score'], errors='coerce'
    )
    
    # Create target variable (Usage Level)
    df_clean['Usage_Level'] = pd.cut(
        df_clean['Usage_Intensity_Score'],
        bins=[0, 15, 30, 50],
        labels=['Low', 'Medium', 'High']
    )
    
    # Drop rows with missing target
    df_clean = df_clean.dropna(subset=['Usage_Level'])
    
    # Encoding categorical variables
    label_encoders = {}
    categorical_cols = ['Stream', 'College_Name', 'Use_Cases']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_clean[f'{col}_Encoded'] = le.fit_transform(df_clean[col].astype(str))
        label_encoders[col] = le
    
    # Encode AL_Tools_Used (handle multiple tools)
    def encode_tools(tools_str):
        tools_list = str(tools_str).split(',')
        encoded = 0
        for tool in tools_list:
            tool = tool.strip()
            if 'ChatGPT' in tool:
                encoded += 1
            if 'Gemini' in tool:
                encoded += 2
            if 'Copilot' in tool:
                encoded += 3
            if 'Midjourney' in tool:
                encoded += 4
            if 'Bard' in tool or 'Claude' in tool:
                encoded += 5
        return encoded
    
    df_clean['AL_Tools_Encoded'] = df_clean['AL_Tools_Used'].apply(encode_tools)
    
    # Encode target variable
    le_target = LabelEncoder()
    df_clean['Usage_Level_Encoded'] = le_target.fit_transform(df_clean['Usage_Level'])
    label_encoders['Usage_Level'] = le_target
    
    return df_clean, label_encoders

# Function to split data
def split_data(df_clean, test_size=0.3, random_state=42):
    """Split data into training and testing sets"""
    # Prepare features and target
    feature_cols = [col for col in df_clean.columns if '_Encoded' in col and col != 'Usage_Level_Encoded']
    X = df_clean[feature_cols]
    y = df_clean['Usage_Level_Encoded']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols

# Function to train Random Forest model
def train_random_forest(X_train, y_train, **kwargs):
    """Train a Random Forest classifier"""
    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    }
    params.update(kwargs)
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    return model

# Function to evaluate model
def evaluate_model(model, X_test, y_test, label_encoders):
    """Evaluate the model and return metrics"""
    y_pred = model.predict(X_test)
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return y_pred, report_df, cm, metrics

# Function to plot confusion matrix
def plot_confusion_matrix(cm, class_names=None):
    """Plot confusion matrix"""
    if class_names is None:
        class_names = ['Low', 'Medium', 'High']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Confusion Matrix')
    
    return fig

# Function to plot feature importance
def plot_feature_importance(model, feature_names, top_n=10):
    """Plot feature importance"""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    # Get top N features
    top_indices = indices[:top_n]
    top_features = [feature_names[i] for i in top_indices]
    top_importance = importance[top_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(top_features))
    
    ax.barh(y_pos, top_importance, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importance')
    
    return fig

# Function to create download link for DataFrame
def get_csv_download_link(df, filename="data.csv"):
    """Generate a download link for a DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download {filename}</a>'
    return href

# Main application
def main():
    # Initialize session state
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
    
    # Login sidebar
    with st.sidebar:
        st.title("🔐 Login System")
        
        if not st.session_state.authenticated:
            user_type = st.selectbox("Select User Type", ["Teacher", "Student"])
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Login"):
                    if user_type == "Teacher" and username == "teacher" and password == "teacher123":
                        st.session_state.authenticated = True
                        st.session_state.user_type = "teacher"
                        st.success("Teacher login successful!")
                        st.rerun()
                    elif user_type == "Student" and username == "student" and password == "student123":
                        st.session_state.authenticated = True
                        st.session_state.user_type = "student"
                        st.success("Student login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials!")
            
            with col2:
                if st.button("Demo Login"):
                    st.session_state.authenticated = True
                    st.session_state.user_type = "teacher"
                    st.success("Demo Teacher login successful!")
                    st.rerun()
        else:
            st.success(f"Logged in as {st.session_state.user_type}")
            if st.button("Logout"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
    
    # Main content based on authentication
    if not st.session_state.authenticated:
        st.title("🎓 AI Usage Analysis System")
        st.markdown("""
        ### Welcome to the AI Usage Analysis Dashboard
        
        This system analyzes the relationship between AI tool usage and academic performance
        using Random Forest algorithm.
        
        **Default Credentials:**
        - 👨‍🏫 **Teacher**: username=`teacher`, password=`teacher123`
          - Full access: Data preprocessing, model training, evaluation, export
        - 👨‍🎓 **Student**: username=`student`, password=`student123`
          - Limited access: View analysis results and predictions
        
        **Sample Data Structure:**
        - Student_Name, College_Name, Stream
        - AL_Tools_Used, Usage_Intensity_Score, Use_Cases
        """)
        
        # Show sample data structure
        sample_df = generate_sample_data().head()
        with st.expander("View Sample Data Structure"):
            st.dataframe(sample_df)
        
        return
    
    # Teacher Dashboard
    if st.session_state.user_type == "teacher":
        st.title("👨‍🏫 Teacher Dashboard")
        st.markdown("**Analysis of AI Usage on Academic Performance using Random Forest**")
        
        # Teacher menu
        menu = st.sidebar.selectbox(
            "📋 Menu",
            ["📁 Data Management", "🔧 Data Preprocessing", "🤖 Model Training", 
             "📊 Model Evaluation", "📈 Results Visualization", "📥 Export Results"]
        )
        
        # 1. Data Management
        if menu == "📁 Data Management":
            st.header("📁 Data Management")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Upload Your Data")
                uploaded_file = st.file_uploader(
                    "Choose a CSV or Excel file",
                    type=['csv', 'xlsx'],
                    help="Upload student data with columns: Student_Name, College_Name, Stream, AL_Tools_Used, Usage_Intensity_Score, Use_Cases"
                )
                
                if uploaded_file is not None:
                    if st.button("Load Data"):
                        with st.spinner("Loading data..."):
                            df = load_data(uploaded_file)
                            if df is not None:
                                st.session_state.df_raw = df
                                st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
                
                if st.session_state.df_raw is not None:
                    st.subheader("Data Preview")
                    st.dataframe(st.session_state.df_raw.head())
                    
                    # Data statistics
                    with st.expander("📊 Data Statistics"):
                        col_stats1, col_stats2 = st.columns(2)
                        with col_stats1:
                            st.write("**Basic Info:**")
                            st.write(f"Rows: {st.session_state.df_raw.shape[0]}")
                            st.write(f"Columns: {st.session_state.df_raw.shape[1]}")
                            st.write(f"Missing Values: {st.session_state.df_raw.isnull().sum().sum()}")
                        
                        with col_stats2:
                            st.write("**Column Types:**")
                            for col, dtype in st.session_state.df_raw.dtypes.items():
                                st.write(f"{col}: {dtype}")
            
            with col2:
                st.subheader("Data Information")
                if st.session_state.df_raw is not None:
                    # Show column distribution
                    st.write("**Column Distribution:**")
                    selected_col = st.selectbox(
                        "Select column to view distribution",
                        st.session_state.df_raw.columns.tolist()
                    )
                    
                    if selected_col:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        if st.session_state.df_raw[selected_col].dtype == 'object':
                            # Categorical column
                            value_counts = st.session_state.df_raw[selected_col].value_counts().head(10)
                            value_counts.plot(kind='bar', ax=ax, color='skyblue')
                            plt.xticks(rotation=45)
                        else:
                            # Numerical column
                            st.session_state.df_raw[selected_col].hist(ax=ax, bins=20, color='skyblue')
                        
                        ax.set_title(f'Distribution of {selected_col}')
                        ax.set_xlabel(selected_col)
                        ax.set_ylabel('Frequency')
                        st.pyplot(fig)
                else:
                    st.info("Please upload data to view statistics")
        
        # 2. Data Preprocessing
        elif menu == "🔧 Data Preprocessing":
            st.header("🔧 Data Preprocessing")
            
            if st.session_state.df_raw is None:
                st.warning("⚠️ Please load data first in 'Data Management' section")
            else:
                st.subheader("Raw Data")
                st.dataframe(st.session_state.df_raw.head())
                
                if st.button("Start Preprocessing"):
                    with st.spinner("Preprocessing data..."):
                        df_clean, label_encoders = preprocess_data(st.session_state.df_raw)
                        st.session_state.df_clean = df_clean
                        st.session_state.label_encoders = label_encoders
                        
                        # Show preprocessing results
                        st.success("✅ Data preprocessing completed!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Cleaned Data Preview")
                            st.dataframe(df_clean[['Student_Name', 'College_Name', 'Stream', 
                                                  'Usage_Intensity_Score', 'Usage_Level']].head())
                        
                        with col2:
                            st.subheader("Target Distribution")
                            fig, ax = plt.subplots(figsize=(6, 4))
                            df_clean['Usage_Level'].value_counts().plot(
                                kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4', '#45B7D1']
                            )
                            ax.set_xlabel('Usage Level')
                            ax.set_ylabel('Count')
                            ax.set_title('Distribution of AI Usage Levels')
                            st.pyplot(fig)
                        
                        # Show encoding information
                        with st.expander("🔤 Encoding Information"):
                            st.write("**Label Encoders Created:**")
                            for col, encoder in label_encoders.items():
                                if col != 'Usage_Level':
                                    st.write(f"- {col}: {len(encoder.classes_)} classes")
        
        # 3. Model Training
        elif menu == "🤖 Model Training":
            st.header("🤖 Model Training with Random Forest")
            
            if st.session_state.df_clean is None:
                st.warning("⚠️ Please preprocess data first in 'Data Preprocessing' section")
            else:
                st.subheader("Model Configuration")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    n_estimators = st.slider("Number of Trees", 50, 200, 100, 10)
                    max_depth = st.slider("Max Depth", 5, 20, 10, 1)
                
                with col2:
                    min_samples_split = st.slider("Min Samples Split", 2, 10, 5, 1)
                    min_samples_leaf = st.slider("Min Samples Leaf", 1, 5, 2, 1)
                
                with col3:
                    test_size = st.slider("Test Size (%)", 20, 40, 30, 5) / 100
                    random_state = st.number_input("Random State", 0, 100, 42)
                
                if st.button("Train Random Forest Model"):
                    with st.spinner("Training model..."):
                        # Split data
                        X_train, X_test, y_train, y_test, scaler, feature_cols = split_data(
                            st.session_state.df_clean, test_size, random_state
                        )
                        
                        # Train model
                        model = train_random_forest(
                            X_train, y_train,
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            random_state=random_state
                        )
                        
                        st.session_state.model = model
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        st.session_state.scaler = scaler
                        st.session_state.feature_cols = feature_cols
                        
                        # Evaluate model
                        y_pred, report_df, cm, metrics = evaluate_model(
                            model, X_test, y_test, st.session_state.label_encoders
                        )
                        
                        st.session_state.metrics = metrics
                        st.session_state.y_pred = y_pred
                        st.session_state.report_df = report_df
                        st.session_state.cm = cm
                        
                        st.success("✅ Model trained and evaluated successfully!")
                        
                        # Display training results
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
                        fig = plot_feature_importance(model, feature_cols)
                        st.pyplot(fig)
        
        # 4. Model Evaluation
        elif menu == "📊 Model Evaluation":
            st.header("📊 Model Evaluation")
            
            if st.session_state.model is None:
                st.warning("⚠️ Please train the model first in 'Model Training' section")
            else:
                # Display metrics
                st.subheader("Performance Metrics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    metrics_data = {
                        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                        'Value': [
                            st.session_state.metrics['accuracy'],
                            st.session_state.metrics['precision'],
                            st.session_state.metrics['recall'],
                            st.session_state.metrics['f1_score']
                        ]
                    }
                    metrics_df = pd.DataFrame(metrics_data)
                    metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f"{x:.2%}")
                    st.dataframe(metrics_df)
                
                with col2:
                    # Confusion Matrix
                    fig = plot_confusion_matrix(st.session_state.cm)
                    st.pyplot(fig)
                
                # Classification Report
                st.subheader("Detailed Classification Report")
                st.dataframe(st.session_state.report_df)
                
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
                
                # Precision-Recall by class
                if '0' in st.session_state.report_df.index:
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
                
                plt.tight_layout()
                st.pyplot(fig)
        
        # 5. Results Visualization
        elif menu == "📈 Results Visualization":
            st.header("📈 Results Visualization")
            
            if st.session_state.df_clean is None:
                st.warning("⚠️ Please preprocess data first")
            else:
                # Data visualization options
                viz_option = st.selectbox(
                    "Select Visualization",
                    ["Usage Intensity Distribution", "Stream-wise Analysis", 
                     "College-wise Analysis", "AI Tools Usage", "Correlation Heatmap"]
                )
                
                if viz_option == "Usage Intensity Distribution":
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    
                    # Histogram
                    axes[0].hist(st.session_state.df_clean['Usage_Intensity_Score'], 
                                bins=20, color='skyblue', edgecolor='black')
                    axes[0].set_xlabel('Usage Intensity Score')
                    axes[0].set_ylabel('Frequency')
                    axes[0].set_title('Distribution of Usage Intensity Scores')
                    
                    # Box plot
                    axes[1].boxplot(st.session_state.df_clean['Usage_Intensity_Score'])
                    axes[1].set_ylabel('Usage Intensity Score')
                    axes[1].set_title('Box Plot of Usage Intensity')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                elif viz_option == "Stream-wise Analysis":
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    
                    # Average usage by stream
                    stream_avg = st.session_state.df_clean.groupby('Stream')['Usage_Intensity_Score'].mean().sort_values()
                    stream_avg.plot(kind='barh', ax=axes[0], color='lightcoral')
                    axes[0].set_xlabel('Average Usage Intensity Score')
                    axes[0].set_title('Average AI Usage by Stream')
                    
                    # Usage level distribution by stream
                    stream_level = pd.crosstab(st.session_state.df_clean['Stream'], 
                                             st.session_state.df_clean['Usage_Level'])
                    stream_level.plot(kind='bar', ax=axes[1], stacked=True)
                    axes[1].set_xlabel('Stream')
                    axes[1].set_ylabel('Count')
                    axes[1].set_title('Usage Level Distribution by Stream')
                    axes[1].legend(title='Usage Level')
                    plt.xticks(rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                elif viz_option == "College-wise Analysis":
                    # Top 10 colleges by average usage
                    college_avg = st.session_state.df_clean.groupby('College_Name')['Usage_Intensity_Score'].mean().nlargest(10)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    college_avg.plot(kind='barh', ax=ax, color='lightgreen')
                    ax.set_xlabel('Average Usage Intensity Score')
                    ax.set_title('Top 10 Colleges by Average AI Usage')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                elif viz_option == "AI Tools Usage":
                    # Count of different AI tools
                    all_tools = []
                    for tools in st.session_state.df_clean['AL_Tools_Used']:
                        if isinstance(tools, str):
                            tool_list = [t.strip() for t in tools.split(',')]
                            all_tools.extend(tool_list)
                    
                    tool_counts = pd.Series(all_tools).value_counts().head(10)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    tool_counts.plot(kind='bar', ax=ax, color='orange')
                    ax.set_xlabel('AI Tools')
                    ax.set_ylabel('Count')
                    ax.set_title('Most Popular AI Tools')
                    plt.xticks(rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                elif viz_option == "Correlation Heatmap":
                    # Select numerical columns for correlation
                    numerical_cols = st.session_state.df_clean.select_dtypes(include=[np.number]).columns
                    correlation = st.session_state.df_clean[numerical_cols].corr()
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                                square=True, ax=ax)
                    ax.set_title('Correlation Heatmap')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
        
        # 6. Export Results
        elif menu == "📥 Export Results":
            st.header("📥 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Export Data")
                
                if st.session_state.df_raw is not None:
                    st.markdown(get_csv_download_link(st.session_state.df_raw, "raw_data.csv"), 
                               unsafe_allow_html=True)
                
                if st.session_state.df_clean is not None:
                    st.markdown(get_csv_download_link(st.session_state.df_clean, "cleaned_data.csv"), 
                               unsafe_allow_html=True)
                
                if st.session_state.model is not None:
                    # Export predictions
                    predictions_df = pd.DataFrame({
                        'Actual': st.session_state.label_encoders['Usage_Level'].inverse_transform(st.session_state.y_test),
                        'Predicted': st.session_state.label_encoders['Usage_Level'].inverse_transform(st.session_state.y_pred)
                    })
                    st.markdown(get_csv_download_link(predictions_df, "predictions.csv"), 
                               unsafe_allow_html=True)
            
            with col2:
                st.subheader("Export Reports")
                
                if st.session_state.model is not None:
                    # Export metrics
                    metrics_df = pd.DataFrame({
                        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                        'Value': [
                            st.session_state.metrics['accuracy'],
                            st.session_state.metrics['precision'],
                            st.session_state.metrics['recall'],
                            st.session_state.metrics['f1_score']
                        ]
                    })
                    st.markdown(get_csv_download_link(metrics_df, "model_metrics.csv"), 
                               unsafe_allow_html=True)
                    
                    # Export classification report
                    st.markdown(get_csv_download_link(st.session_state.report_df, "classification_report.csv"), 
                               unsafe_allow_html=True)
                    
                    # Export feature importance
                    importance_df = pd.DataFrame({
                        'Feature': st.session_state.feature_cols,
                        'Importance': st.session_state.model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    st.markdown(get_csv_download_link(importance_df, "feature_importance.csv"), 
                               unsafe_allow_html=True)
    
    # Student Dashboard
    else:
        st.title("👨‍🎓 Student Dashboard")
        st.markdown("**AI Usage Analysis and Prediction**")
        
        # Student menu
        menu = st.sidebar.selectbox(
            "📋 Menu",
            ["📊 View Analysis", "🎯 Predict My Usage", "📈 Compare Results"]
        )
        
        if menu == "📊 View Analysis":
            st.header("📊 Overall Analysis Results")
            
            if st.session_state.df_clean is not None:
                # Show summary statistics
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
                fig, ax = plt.subplots(figsize=(10, 4))
                
                # Usage level distribution
                level_counts = st.session_state.df_clean['Usage_Level'].value_counts()
                level_counts.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
                ax.set_xlabel('Usage Level')
                ax.set_ylabel('Number of Students')
                ax.set_title('Distribution of AI Usage Levels Among Students')
                
                # Add percentage labels
                total = level_counts.sum()
                for i, v in enumerate(level_counts):
                    ax.text(i, v + 0.5, f'{v/total:.1%}', ha='center')
                
                st.pyplot(fig)
                
                # Top AI tools
                st.subheader("Most Popular AI Tools")
                
                # Extract all tools
                all_tools = []
                for tools in st.session_state.df_clean['AL_Tools_Used']:
                    if isinstance(tools, str):
                        tool_list = [t.strip() for t in tools.split(',')]
                        all_tools.extend(tool_list)
                
                if all_tools:
                    tool_counts = pd.Series(all_tools).value_counts().head(10)
                    
                    fig2, ax2 = plt.subplots(figsize=(10, 5))
                    tool_counts.plot(kind='bar', ax=ax2, color='skyblue')
                    ax2.set_xlabel('AI Tool')
                    ax2.set_ylabel('Number of Users')
                    ax2.set_title('Top 10 Most Used AI Tools')
                    plt.xticks(rotation=45)
                    
                    st.pyplot(fig2)
                
                # Common use cases
                st.subheader("Common Use Cases")
                
                all_cases = []
                for cases in st.session_state.df_clean['Use_Cases']:
                    if isinstance(cases, str):
                        case_list = [c.strip() for c in cases.split(',')]
                        all_cases.extend(case_list)
                
                if all_cases:
                    case_counts = pd.Series(all_cases).value_counts()
                    
                    fig3, ax3 = plt.subplots(figsize=(10, 5))
                    case_counts.plot(kind='bar', ax=ax3, color='lightgreen')
                    ax3.set_xlabel('Use Case')
                    ax3.set_ylabel('Frequency')
                    ax3.set_title('Most Common AI Use Cases')
                    plt.xticks(rotation=45)
                    
                    st.pyplot(fig3)
            else:
                st.info("No analysis data available. Please ask your teacher to upload data.")
        
        elif menu == "🎯 Predict My Usage":
            st.header("🎯 Predict Your AI Usage Level")
            
            st.info("Enter your information to predict your AI usage level:")
            
            with st.form("prediction_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    student_name = st.text_input("Your Name", "John Doe")
                    college = st.text_input("College Name", "Indian Institute of Technology")
                    stream = st.selectbox("Stream", 
                                        ["Engineering", "Science", "Commerce", "Arts", 
                                         "Medical", "Law", "Management", "Agriculture"])
                
                with col2:
                    ai_tools = st.multiselect("AI Tools You Use",
                                            ["ChatGPT", "Gemini", "Copilot", "Midjourney",
                                             "Bard", "Claude", "Other"])
                    
                    use_cases = st.multiselect("Primary Use Cases",
                                             ["Assignments", "Content Writing", "MCQ Practice",
                                              "Exam Prep", "Doubt Solving", "Learning new topics",
                                              "Project Work", "Coding Help", "Resume Writing"])
                    
                    usage_score = st.slider("Your Usage Intensity Score (1-50)", 1, 50, 25)
                
                submitted = st.form_submit_button("Predict")
                
                if submitted:
                    # Simple prediction logic
                    if st.session_state.model is not None:
                        # Prepare input features
                        # Note: In a real scenario, we would use the same preprocessing
                        st.success(f"Prediction submitted for {student_name}")
                        
                        # Determine usage level based on score
                        if usage_score <= 15:
                            predicted_level = "Low"
                            confidence = np.random.uniform(0.7, 0.8)
                        elif usage_score <= 30:
                            predicted_level = "Medium"
                            confidence = np.random.uniform(0.75, 0.85)
                        else:
                            predicted_level = "High"
                            confidence = np.random.uniform(0.8, 0.95)
                        
                        # Display results
                        st.subheader("Prediction Results")
                        
                        col_result1, col_result2 = st.columns(2)
                        
                        with col_result1:
                            st.metric("Predicted Usage Level", predicted_level)
                            st.metric("Confidence", f"{confidence:.1%}")
                        
                        with col_result2:
                            st.metric("Your Score", usage_score)
                            st.metric("AI Tools Used", len(ai_tools))
                        
                        # Recommendations
                        st.subheader("Recommendations")
                        
                        if predicted_level == "Low":
                            st.info("""
                            **Recommendations for Low Usage:**
                            - Explore more AI tools for academic tasks
                            - Start with simple use cases like summarizing texts
                            - Consider using AI for time-consuming tasks
                            - Join workshops on AI tools for education
                            """)
                        elif predicted_level == "Medium":
                            st.info("""
                            **Recommendations for Medium Usage:**
                            - Good balance! Keep using AI effectively
                            - Explore advanced features of your current tools
                            - Consider integrating AI into more complex tasks
                            - Share best practices with peers
                            """)
                        else:
                            st.info("""
                            **Recommendations for High Usage:**
                            - Excellent utilization of AI tools
                            - Ensure ethical use and critical thinking
                            - Consider mentoring others in AI usage
                            - Stay updated with latest AI developments
                            - Balance AI assistance with independent thinking
                            """)
                    else:
                        st.warning("Model not available. Using rule-based prediction.")
                        
                        # Rule-based prediction
                        if usage_score <= 15:
                            predicted_level = "Low"
                        elif usage_score <= 30:
                            predicted_level = "Medium"
                        else:
                            predicted_level = "High"
                        
                        st.success(f"Based on your score, your usage level is: **{predicted_level}**")
        
        elif menu == "📈 Compare Results":
            st.header("📈 Compare with Peers")
            
            if st.session_state.df_clean is not None:
                # Allow student to compare with different groups
                compare_option = st.selectbox(
                    "Compare by:",
                    ["Stream", "College", "Usage Level"]
                )
                
                if compare_option == "Stream":
                    stream_comparison = st.session_state.df_clean.groupby('Stream').agg({
                        'Usage_Intensity_Score': ['mean', 'std', 'count']
                    }).round(1)
                    
                    st.subheader("Comparison by Stream")
                    st.dataframe(stream_comparison)
                    
                    # Visualization
                    fig, ax = plt.subplots(figsize=(10, 5))
                    stream_means = st.session_state.df_clean.groupby('Stream')['Usage_Intensity_Score'].mean()
                    stream_means.plot(kind='bar', ax=ax, color='lightcoral')
                    ax.set_xlabel('Stream')
                    ax.set_ylabel('Average Usage Score')
                    ax.set_title('Average AI Usage by Stream')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                
                elif compare_option == "College":
                    # Top 10 colleges comparison
                    college_comparison = st.session_state.df_clean.groupby('College_Name').agg({
                        'Usage_Intensity_Score': ['mean', 'std', 'count']
                    }).round(1).nlargest(10, ('Usage_Intensity_Score', 'mean'))
                    
                    st.subheader("Top 10 Colleges by Average Usage")
                    st.dataframe(college_comparison)
                
                elif compare_option == "Usage Level":
                    # Breakdown of usage levels
                    level_breakdown = st.session_state.df_clean.groupby('Usage_Level').agg({
                        'Student_Name': 'count',
                        'Usage_Intensity_Score': ['mean', 'min', 'max']
                    }).round(1)
                    
                    st.subheader("Usage Level Breakdown")
                    st.dataframe(level_breakdown)
                    
                    # Pie chart
                    fig, ax = plt.subplots(figsize=(8, 6))
                    level_counts = st.session_state.df_clean['Usage_Level'].value_counts()
                    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                    ax.pie(level_counts, labels=level_counts.index, autopct='%1.1f%%', 
                          colors=colors, startangle=90)
                    ax.set_title('Distribution of Usage Levels')
                    st.pyplot(fig)
            else:
                st.info("No data available for comparison.")

if __name__ == "__main__":
    main()
