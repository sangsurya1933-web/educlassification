import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="AI Analytics Dashboard - UMMgl",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state untuk login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'preprocessing_done' not in st.session_state:
    st.session_state.preprocessing_done = False

# Custom CSS untuk gaya admin LTE futuristik
st.markdown("""
<style>
    /* Main Styles */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stApp {
        background: #f5f7fb;
    }
    
    /* Header */
    .header-title {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.2rem;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-1vq4p4l {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid #334155;
    }
    
    .sidebar .sidebar-content {
        background: transparent;
    }
    
    /* Cards */
    .custom-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 50px rgba(0,0,0,0.12);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    
    .primary-btn {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
    }
    
    .secondary-btn {
        background: linear-gradient(45deg, #10b981, #059669) !important;
    }
    
    .danger-btn {
        background: linear-gradient(45deg, #ef4444, #dc2626) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #f1f5f9;
        padding: 4px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Dataframes */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }
    
    /* Loading */
    .stSpinner > div {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(45deg, #667eea, #764ba2);
    }
    
    /* Selectbox, Slider, etc */
    .stSelectbox, .stSlider, .stNumberInput {
        background: white;
        border-radius: 8px;
        padding: 5px;
        border: 1px solid #e2e8f0;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .status-success {
        background: linear-gradient(45deg, #10b981, #059669);
        color: white;
    }
    
    .status-warning {
        background: linear-gradient(45deg, #f59e0b, #d97706);
        color: white;
    }
    
    .status-danger {
        background: linear-gradient(45deg, #ef4444, #dc2626);
        color: white;
    }
    
    /* Gradient Text */
    .gradient-text {
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #64748b;
        font-size: 0.9rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

def load_sample_data():
    """Load dataset sampel yang disederhanakan"""
    # Data sampel yang lebih lengkap
    data = {
        'Nama': [
            'Althaf Rayyan Putra', 'Ayesha Kinanti', 'Salsabila Nurfadila',
            'Anindya Safira', 'Iqbal Ramadhan', 'Muhammad Rizky Pratama',
            'Fikri Alfarizi', 'Citra Maharani', 'Zidan Harits',
            'Rizky Kurniawan Putra', 'Raka Bimantara', 'Zahra Alya Safitri',
            'Muhammad Naufal Haidar', 'Ammar Zaky Firmansyah', 'Ilham Nurhadi'
        ],
        'Studi_Jurusan': [
            'Teknologi Informasi', 'Teknologi Informasi', 'Teknik Informatika',
            'Teknik Informatika', 'Farmasi', 'Teknologi Informasi',
            'Teknologi Informasi', 'Keperawatan', 'Farmasi',
            'Teknik Informatika', 'Farmasi', 'Teknik Informatika',
            'Farmasi', 'Farmasi', 'Teknologi Informasi'
        ],
        'Semester': [7, 3, 1, 5, 1, 5, 1, 5, 7, 5, 3, 3, 1, 3, 1],
        'AI_Tools': ['Gemini', 'Gemini', 'Gemini', 'Gemini', 'Gemini', 
                    'Gemini', 'ChatGPT', 'Multiple', 'Gemini',
                    'ChatGPT', 'ChatGPT', 'Gemini', 'Gemini', 'ChatGPT', 'Gemini'],
        'Trust_Level': [4, 4, 5, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5],
        'Usage_Intensity_Score': [8, 9, 3, 6, 10, 4, 7, 2, 4, 3, 4, 10, 3, 7, 9]
    }
    
    df = pd.DataFrame(data)
    # Handle '10+' values
    df['Usage_Intensity_Score'] = df['Usage_Intensity_Score'].apply(
        lambda x: 10 if str(x) == '10+' else int(x)
    )
    
    # Create target variable (kategori intensitas penggunaan)
    def categorize_intensity(score):
        if score <= 3:
            return 'Rendah'
        elif score <= 7:
            return 'Sedang'
        else:
            return 'Tinggi'
    
    df['Kategori_Intensitas'] = df['Usage_Intensity_Score'].apply(categorize_intensity)
    return df

def preprocessing_data(df):
    """Melakukan preprocessing data"""
    # Copy dataframe
    df_processed = df.copy()
    
    # 1. Handle missing values
    df_processed = df_processed.dropna()
    
    # 2. Encoding variabel kategorikal
    label_encoders = {}
    
    # Encoding untuk fitur kategorikal
    categorical_cols = ['Studi_Jurusan', 'AI_Tools']
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col + '_encoded'] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
    
    # Encoding target
    le_target = LabelEncoder()
    df_processed['Target_encoded'] = le_target.fit_transform(df_processed['Kategori_Intensitas'])
    label_encoders['Target'] = le_target
    
    # 3. Split features and target
    features = ['Semester', 'Trust_Level', 'Studi_Jurusan_encoded', 'AI_Tools_encoded']
    X = df_processed[features]
    y = df_processed['Target_encoded']
    
    # 4. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 5. Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'df_processed': df_processed,
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'label_encoders': label_encoders,
        'scaler': scaler,
        'features': features
    }

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10):
    """Melatih model Random Forest"""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, label_encoders):
    """Evaluasi model dan tampilkan metrics"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Decode labels
    y_test_decoded = label_encoders['Target'].inverse_transform(y_test)
    y_pred_decoded = label_encoders['Target'].inverse_transform(y_pred)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'y_test_decoded': y_test_decoded,
        'y_pred_decoded': y_pred_decoded,
        'y_pred_proba': y_pred_proba
    }

def create_gauge_chart(value, max_value, title):
    """Membuat gauge chart untuk metrics"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [0, max_value]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, max_value/3], 'color': "red"},
                {'range': [max_value/3, 2*max_value/3], 'color': "yellow"},
                {'range': [2*max_value/3, max_value], 'color': "green"}
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def login_page():
    """Halaman login futuristik"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 2.5rem; border-radius: 15px; 
                    box-shadow: 0 20px 60px rgba(0,0,0,0.1); text-align: center;">
            <h1 style="background: linear-gradient(45deg, #667eea, #764ba2);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       margin-bottom: 0.5rem;">🤖 AI ANALYTICS</h1>
            <p style="color: #64748b; margin-bottom: 2rem;">UMMgl Data Science Platform</p>
        """, unsafe_allow_html=True)
        
        # Login Options
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        
        role = st.radio(
            "Select Role",
            ["👨‍🏫 Administrator", "👨‍🎓 Student User"],
            horizontal=True
        )
        
        st.markdown("---")
        
        if role == "👨‍🏫 Administrator":
            if st.button("🔐 Login as Administrator", use_container_width=True, type="primary"):
                st.session_state.logged_in = True
                st.session_state.user_role = 'admin'
                st.rerun()
        else:
            if st.button("🎓 Login as Student", use_container_width=True, type="primary"):
                st.session_state.logged_in = True
                st.session_state.user_role = 'student'
                st.rerun()
        
        st.markdown("""
        <div style="margin-top: 2rem; padding: 1rem; background: #f8fafc; 
                    border-radius: 10px; border-left: 4px solid #667eea;">
            <p style="margin: 0; color: #475569; font-size: 0.9rem;">
            <strong>ℹ️ Platform Info:</strong><br>
            • AI Usage Analytics<br>
            • Random Forest Algorithm<br>
            • Real-time Predictions
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def admin_dashboard():
    """Dashboard untuk Administrator"""
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0;">
            <h2 style="background: linear-gradient(45deg, #667eea, #764ba2);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                🤖 AI Analytics
            </h2>
            <p style="color: #94a3b8; font-size: 0.9rem;">Administrator Panel</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        menu = st.radio(
            "MAIN NAVIGATION",
            ["📊 Dashboard Overview", "🔧 Data Preprocessing", 
             "🤖 Machine Learning", "📈 Model Evaluation", 
             "📤 Export Results", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Quick Stats
        if st.session_state.df is not None:
            df = st.session_state.df
            st.markdown("""
            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;">
                <p style="color: #cbd5e1; margin: 0; font-size: 0.9rem;">📊 QUICK STATS</p>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Students", len(df['Nama'].unique()))
            with col2:
                st.metric("Records", len(df))
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            st.session_state.logged_in = False
            st.session_state.user_role = None
            st.rerun()
    
    # Main Content
    if menu == "📊 Dashboard Overview":
        st.markdown('<h1 class="header-title">📊 Dashboard Overview</h1>', unsafe_allow_html=True)
        
        if st.session_state.df is None:
            st.session_state.df = load_sample_data()
        
        df = st.session_state.df
        
        # Top Metrics Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3 style="margin: 0; font-size: 2rem;">{}</h3>
                <p style="margin: 0; opacity: 0.9;">Total Students</p>
            </div>
            """.format(len(df['Nama'].unique())), unsafe_allow_html=True)
        
        with col2:
            avg_intensity = df['Usage_Intensity_Score'].mean()
            st.markdown("""
            <div class="metric-card">
                <h3 style="margin: 0; font-size: 2rem;">{:.1f}</h3>
                <p style="margin: 0; opacity: 0.9;">Avg Intensity</p>
            </div>
            """.format(avg_intensity), unsafe_allow_html=True)
        
        with col3:
            avg_trust = df['Trust_Level'].mean()
            st.markdown("""
            <div class="metric-card">
                <h3 style="margin: 0; font-size: 2rem;">{:.1f}</h3>
                <p style="margin: 0; opacity: 0.9;">Avg Trust Level</p>
            </div>
            """.format(avg_trust), unsafe_allow_html=True)
        
        with col4:
            popular_tool = df['AI_Tools'].mode()[0]
            st.markdown("""
            <div class="metric-card">
                <h3 style="margin: 0; font-size: 1.5rem;">{}</h3>
                <p style="margin: 0; opacity: 0.9;">Top AI Tool</p>
            </div>
            """.format(popular_tool), unsafe_allow_html=True)
        
        # Charts Section
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("📈 Distribution by Department")
            
            # Create pie chart
            dept_counts = df['Studi_Jurusan'].value_counts()
            fig = px.pie(
                values=dept_counts.values,
                names=dept_counts.index,
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("🔧 AI Tools Usage")
            
            # Create bar chart
            tool_counts = df['AI_Tools'].value_counts()
            fig = px.bar(
                x=tool_counts.index,
                y=tool_counts.values,
                color=tool_counts.values,
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=300, xaxis_title="AI Tools", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Data Preview
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("📋 Data Preview")
        
        tabs = st.tabs(["📊 Raw Data", "📈 Statistics", "🔍 Insights"])
        
        with tabs[0]:
            st.dataframe(df.head(10), use_container_width=True)
        
        with tabs[1]:
            st.dataframe(df.describe(), use_container_width=True)
        
        with tabs[2]:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Top 5 Students by Intensity:**")
                top_students = df.nlargest(5, 'Usage_Intensity_Score')[['Nama', 'Usage_Intensity_Score']]
                st.dataframe(top_students, use_container_width=True)
            with col2:
                st.write("**Trust Level Distribution:**")
                trust_dist = df['Trust_Level'].value_counts().sort_index()
                st.bar_chart(trust_dist)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif menu == "🔧 Data Preprocessing":
        st.markdown('<h1 class="header-title">🔧 Data Preprocessing</h1>', unsafe_allow_html=True)
        
        if st.session_state.df is None:
            st.session_state.df = load_sample_data()
        
        df = st.session_state.df
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("🔍 Original Data Preview")
            st.dataframe(df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("⚙️ Preprocessing Steps")
            
            steps = [
                "✅ Data Loading",
                "✅ Missing Value Handling",
                "✅ Categorical Encoding",
                "✅ Feature Selection",
                "✅ Train-Test Split",
                "✅ Feature Scaling"
            ]
            
            for step in steps:
                st.markdown(f'<p style="margin: 5px 0;">{step}</p>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            if st.button("🚀 Start Preprocessing", use_container_width=True, type="primary"):
                with st.spinner("Processing data..."):
                    preprocessing_results = preprocessing_data(df)
                    st.session_state.preprocessing_results = preprocessing_results
                    st.session_state.preprocessing_done = True
                    
                    st.success("✅ Preprocessing completed successfully!")
                    
                    # Show encoded data
                    st.markdown('<div style="margin-top: 1rem;">', unsafe_allow_html=True)
                    st.subheader("📊 Encoded Features")
                    encoded_df = preprocessing_results['df_processed'][
                        ['Studi_Jurusan', 'Studi_Jurusan_encoded', 
                         'AI_Tools', 'AI_Tools_encoded', 'Kategori_Intensitas', 'Target_encoded']
                    ].head()
                    st.dataframe(encoded_df, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif menu == "🤖 Machine Learning":
        st.markdown('<h1 class="header-title">🤖 Machine Learning</h1>', unsafe_allow_html=True)
        
        if not st.session_state.preprocessing_done:
            st.warning("⚠️ Please run preprocessing first!")
            return
        
        preprocessing_results = st.session_state.preprocessing_results
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("⚙️ Model Configuration")
            
            n_estimators = st.slider("Number of Trees", 10, 200, 100, 10,
                                    help="Number of decision trees in the forest")
            max_depth = st.slider("Max Depth", 3, 20, 10, 1,
                                help="Maximum depth of each tree")
            
            st.markdown("---")
            
            if st.button("🎯 Train Model", use_container_width=True, type="primary"):
                with st.spinner("Training Random Forest model..."):
                    model = train_random_forest(
                        preprocessing_results['X_train'],
                        preprocessing_results['y_train'],
                        n_estimators,
                        max_depth
                    )
                    st.session_state.model = model
                    
                    st.success("✅ Model trained successfully!")
                    
                    # Show feature importance
                    feature_importance = model.feature_importances_
                    features = preprocessing_results['features']
                    
                    importance_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': feature_importance
                    }).sort_values('Importance', ascending=True)
                    
                    fig = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        color='Importance',
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(height=300, title="Feature Importance")
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            if st.session_state.model is not None:
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.subheader("📊 Model Details")
                
                model = st.session_state.model
                
                # Model info cards
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Trees", model.n_estimators)
                with col2:
                    st.metric("Max Depth", model.max_depth)
                with col3:
                    st.metric("Classes", len(model.classes_))
                
                # Training sample predictions
                st.markdown("---")
                st.subheader("🔮 Sample Prediction")
                
                sample_idx = st.number_input("Select test sample index", 
                                          0, len(preprocessing_results['X_test'])-1, 0)
                
                if st.button("Predict Selected Sample", use_container_width=True):
                    sample = preprocessing_results['X_test'][sample_idx].reshape(1, -1)
                    prediction = model.predict(sample)
                    probability = model.predict_proba(sample)[0]
                    
                    label_decoder = preprocessing_results['label_encoders']['Target']
                    predicted_label = label_decoder.inverse_transform(prediction)[0]
                    
                    # Display prediction
                    st.markdown(f"""
                    <div style="background: #f8fafc; padding: 1rem; border-radius: 10px; 
                                border-left: 4px solid #667eea;">
                        <p><strong>Predicted Category:</strong> 
                        <span class="status-success">{predicted_label}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability chart
                    prob_df = pd.DataFrame({
                        'Category': label_decoder.classes_,
                        'Probability': probability
                    }).sort_values('Probability', ascending=False)
                    
                    fig = px.bar(
                        prob_df,
                        x='Probability',
                        y='Category',
                        orientation='h',
                        color='Probability',
                        color_continuous_scale='RdYlBu'
                    )
                    fig.update_layout(height=200)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    elif menu == "📈 Model Evaluation":
        st.markdown('<h1 class="header-title">📈 Model Evaluation</h1>', unsafe_allow_html=True)
        
        if st.session_state.model is None:
            st.warning("⚠️ Please train a model first!")
            return
        
        model = st.session_state.model
        preprocessing_results = st.session_state.preprocessing_results
        
        # Evaluate model
        evaluation_results = evaluate_model(
            model, 
            preprocessing_results['X_test'], 
            preprocessing_results['y_test'],
            preprocessing_results['label_encoders']
        )
        
        # Top Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{evaluation_results['accuracy']:.3f}")
        with col2:
            st.metric("Precision", f"{evaluation_results['classification_report']['macro avg']['precision']:.3f}")
        with col3:
            st.metric("Recall", f"{evaluation_results['classification_report']['macro avg']['recall']:.3f}")
        with col4:
            st.metric("F1-Score", f"{evaluation_results['classification_report']['macro avg']['f1-score']:.3f}")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("📊 Confusion Matrix")
            
            cm = evaluation_results['confusion_matrix']
            classes = preprocessing_results['label_encoders']['Target'].classes_
            
            fig = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale='Blues',
                labels=dict(x="Predicted", y="Actual"),
                x=classes,
                y=classes
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("📈 Performance Metrics")
            
            # Classification report
            report_df = pd.DataFrame(evaluation_results['classification_report']).transpose()
            st.dataframe(report_df, use_container_width=True)
            
            # Class-wise metrics chart
            classes = preprocessing_results['label_encoders']['Target'].classes_
            metrics_data = []
            
            for i, class_name in enumerate(classes):
                metrics_data.append({
                    'Class': class_name,
                    'Precision': evaluation_results['classification_report'][str(i)]['precision'],
                    'Recall': evaluation_results['classification_report'][str(i)]['recall'],
                    'F1-Score': evaluation_results['classification_report'][str(i)]['f1-score']
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            fig = px.bar(
                metrics_df.melt(id_vars=['Class'], var_name='Metric', value_name='Value'),
                x='Class',
                y='Value',
                color='Metric',
                barmode='group',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif menu == "📤 Export Results":
        st.markdown('<h1 class="header-title">📤 Export Results</h1>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("💾 Export Model")
            st.write("Save trained model for deployment")
            
            if st.session_state.model is not None:
                model_bytes = pickle.dumps(st.session_state.model)
                st.download_button(
                    label="Download Model (.pkl)",
                    data=model_bytes,
                    file_name="random_forest_model.pkl",
                    mime="application/octet-stream",
                    use_container_width=True
                )
            else:
                st.warning("No model available")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("📊 Export Data")
            st.write("Export processed data")
            
            if hasattr(st.session_state, 'preprocessing_results'):
                preprocessing_results = st.session_state.preprocessing_results
                df_bytes = preprocessing_results['df_processed'].to_csv(index=False).encode()
                st.download_button(
                    label="Download CSV",
                    data=df_bytes,
                    file_name="processed_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.warning("No processed data available")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("📈 Export Report")
            st.write("Generate comprehensive report")
            
            if st.session_state.model is not None:
                # Create report
                report_text = f"""
                ============================================
                AI ANALYTICS PLATFORM - MODEL EVALUATION REPORT
                ============================================
                
                Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                MODEL INFORMATION:
                - Algorithm: Random Forest Classifier
                - Number of Trees: {st.session_state.model.n_estimators}
                - Max Depth: {st.session_state.model.max_depth}
                
                DATASET INFORMATION:
                - Total Samples: {len(st.session_state.df) if st.session_state.df is not None else 'N/A'}
                - Features: {', '.join(st.session_state.preprocessing_results['features']) if hasattr(st.session_state, 'preprocessing_results') else 'N/A'}
                
                PERFORMANCE METRICS:
                - Accuracy: {evaluation_results['accuracy'] if 'evaluation_results' in locals() else 'N/A'}
                
                RECOMMENDATIONS:
                1. Model ready for production deployment
                2. Consider hyperparameter tuning for optimization
                3. Monitor model performance regularly
                
                ---
                Generated by AI Analytics Platform v1.0
                """
                
                st.download_button(
                    label="Download Report (.txt)",
                    data=report_text,
                    file_name="model_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            else:
                st.warning("No model available")
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif menu == "⚙️ Settings":
        st.markdown('<h1 class="header-title">⚙️ Settings</h1>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("🛠️ System Settings")
            
            st.selectbox("Theme Mode", ["Light", "Dark", "Auto"])
            st.selectbox("Language", ["English", "Indonesian", "Auto"])
            st.slider("Auto-save Interval (minutes)", 1, 60, 5)
            
            if st.button("💾 Save Settings", use_container_width=True, type="primary"):
                st.success("Settings saved successfully!")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("🔧 Advanced Options")
            
            st.checkbox("Enable real-time monitoring", value=True)
            st.checkbox("Auto-update models", value=False)
            st.checkbox("Log all predictions", value=True)
            st.checkbox("Email notifications", value=False)
            
            st.markdown("---")
            
            if st.button("🔄 Reset System", type="secondary", use_container_width=True):
                st.session_state.df = None
                st.session_state.model = None
                st.session_state.preprocessing_done = False
                st.success("System reset completed!")
            
            st.markdown('</div>', unsafe_allow_html=True)

def student_dashboard():
    """Dashboard untuk Student"""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0;">
            <h2 style="background: linear-gradient(45deg, #667eea, #764ba2);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                🎓 Student Portal
            </h2>
            <p style="color: #94a3b8; font-size: 0.9rem;">AI Usage Analytics</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        menu = st.radio(
            "STUDENT MENU",
            ["📊 My Analytics", "🤖 Predict My Usage", "📚 Learning Center", "👤 My Profile"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            st.session_state.logged_in = False
            st.session_state.user_role = None
            st.rerun()
    
    # Main Content
    if menu == "📊 My Analytics":
        st.markdown('<h1 class="header-title">📊 My AI Usage Analytics</h1>', unsafe_allow_html=True)
        
        if st.session_state.df is None:
            st.session_state.df = load_sample_data()
        
        df = st.session_state.df
        
        # Student Selector
        col1, col2 = st.columns([1, 2])
        with col1:
            student_name = st.selectbox("Select Student", df['Nama'].unique())
        
        if student_name:
            student_data = df[df['Nama'] == student_name]
            
            # Student Overview Cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin: 0; font-size: 2rem;">{student_data['Usage_Intensity_Score'].mean():.1f}</h3>
                    <p style="margin: 0; opacity: 0.9;">Avg Intensity</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin: 0; font-size: 2rem;">{student_data['Trust_Level'].mean():.1f}</h3>
                    <p style="margin: 0; opacity: 0.9;">Avg Trust Level</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                mode_tool = student_data['AI_Tools'].mode()[0] if not student_data.empty else "N/A"
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin: 0; font-size: 1.5rem;">{mode_tool}</h3>
                    <p style="margin: 0; opacity: 0.9;">Favorite Tool</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                avg_semester = student_data['Semester'].mean()
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin: 0; font-size: 2rem;">{avg_semester:.1f}</h3>
                    <p style="margin: 0; opacity: 0.9;">Avg Semester</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.subheader("📈 Usage Trend")
                
                # Create line chart for usage over time (simulated)
                fig = px.line(
                    student_data.sort_values('Semester'),
                    x='Semester',
                    y='Usage_Intensity_Score',
                    markers=True,
                    line_shape='spline'
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.subheader("🔧 Tool Preference")
                
                tool_counts = student_data['AI_Tools'].value_counts()
                fig = px.pie(
                    values=tool_counts.values,
                    names=tool_counts.index,
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.Plasma
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Usage History
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("📜 Usage History")
            st.dataframe(
                student_data[['Semester', 'AI_Tools', 'Trust_Level', 'Usage_Intensity_Score', 'Kategori_Intensitas']],
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif menu == "🤖 Predict My Usage":
        st.markdown('<h1 class="header-title">🤖 Predict My AI Usage Category</h1>', unsafe_allow_html=True)
        
        if st.session_state.model is None:
            st.warning("⚠️ Model is not available yet. Please contact administrator.")
        
        # Prediction Form
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("📝 Input Your Data")
            
            jurusan = st.selectbox("Study Program", [
                "Teknologi Informasi", "Teknik Informatika", "Farmasi", 
                "Keperawatan", "Mesin Otomotif"
            ])
            
            semester = st.slider("Current Semester", 1, 8, 5)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("⚙️ AI Preferences")
            
            ai_tool = st.selectbox("Primary AI Tool", [
                "ChatGPT", "Gemini", "Copilot", "Multiple"
            ])
            
            trust_level = st.slider("Trust Level in AI", 1, 5, 4,
                                  help="How much do you trust AI tools?")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Prediction Button
        if st.button("🔮 Predict My Category", use_container_width=True, type="primary"):
            if hasattr(st.session_state, 'preprocessing_results') and st.session_state.model is not None:
                preprocessing_results = st.session_state.preprocessing_results
                model = st.session_state.model
                
                try:
                    # Encode input
                    jurusan_encoded = preprocessing_results['label_encoders']['Studi_Jurusan'].transform([jurusan])[0]
                    ai_tool_encoded = preprocessing_results['label_encoders']['AI_Tools'].transform([ai_tool])[0]
                    
                    # Prepare input
                    input_array = np.array([[semester, trust_level, jurusan_encoded, ai_tool_encoded]])
                    input_scaled = preprocessing_results['scaler'].transform(input_array)
                    
                    # Predict
                    prediction = model.predict(input_scaled)
                    probability = model.predict_proba(input_scaled)[0]
                    
                    # Decode prediction
                    label_decoder = preprocessing_results['label_encoders']['Target']
                    predicted_label = label_decoder.inverse_transform(prediction)[0]
                    
                    # Display Results
                    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                    
                    # Result Header
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1.5rem; 
                                background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
                                border-radius: 10px; margin-bottom: 1rem;">
                        <h2 style="margin: 0; color: #1e293b;">Predicted Category</h2>
                        <h1 style="margin: 0; font-size: 3rem; 
                                  background: linear-gradient(45deg, #667eea, #764ba2);
                                  -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                            {predicted_label}
                        </h1>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability Chart
                    prob_df = pd.DataFrame({
                        'Category': label_decoder.classes_,
                        'Probability (%)': (probability * 100)
                    }).sort_values('Probability (%)', ascending=True)
                    
                    fig = px.bar(
                        prob_df,
                        x='Probability (%)',
                        y='Category',
                        orientation='h',
                        color='Probability (%)',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendations
                    st.subheader("💡 Personalized Recommendations")
                    
                    recommendations = {
                        'Rendah': [
                            "Explore basic AI tools for academic assistance",
                            "Start with simple AI tasks like summarizing articles",
                            "Join AI workshops on campus",
                            "Try using AI for daily task planning"
                        ],
                        'Sedang': [
                            "Experiment with advanced AI features",
                            "Combine multiple AI tools for complex tasks",
                            "Document your AI usage patterns",
                            "Share your experiences with peers"
                        ],
                        'Tinggi': [
                            "Consider becoming an AI mentor",
                            "Explore AI integration in your projects",
                            "Stay updated with latest AI developments",
                            "Contribute to AI community discussions"
                        ]
                    }
                    
                    for rec in recommendations.get(predicted_label, []):
                        st.markdown(f"✓ {rec}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
    
    elif menu == "📚 Learning Center":
        st.markdown('<h1 class="header-title">📚 AI Learning Center</h1>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("🎓 Learning Resources")
            
            resources = [
                {"title": "AI Basics 101", "level": "Beginner", "duration": "2 hours"},
                {"title": "Advanced Prompt Engineering", "level": "Intermediate", "duration": "4 hours"},
                {"title": "AI Ethics & Best Practices", "level": "Advanced", "duration": "3 hours"},
                {"title": "AI in Academic Research", "level": "Intermediate", "duration": "3 hours"}
            ]
            
            for res in resources:
                st.markdown(f"""
                <div style="padding: 1rem; margin: 0.5rem 0; background: #f8fafc; border-radius: 8px;">
                    <strong>{res['title']}</strong><br>
                    <span style="color: #64748b; font-size: 0.9rem;">
                    Level: {res['level']} • Duration: {res['duration']}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("Explore All Resources", use_container_width=True):
                st.info("Feature coming soon!")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("📊 Community Stats")
            
            stats = [
                {"label": "Active Learners", "value": "1,234"},
                {"label": "Courses Completed", "value": "456"},
                {"label": "Avg Learning Hours", "value": "12.5"},
                {"label": "Success Rate", "value": "92%"}
            ]
            
            for stat in stats:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(stat['label'])
                with col2:
                    st.write(f"**{stat['value']}**")
            
            st.markdown("---")
            st.subheader("🏆 Your Progress")
            
            progress_data = {
                "AI Fundamentals": 85,
                "Prompt Engineering": 60,
                "Research Tools": 40,
                "Ethics Course": 25
            }
            
            for course, progress in progress_data.items():
                st.write(f"{course}")
                st.progress(progress / 100)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif menu == "👤 My Profile":
        st.markdown('<h1 class="header-title">👤 My Profile</h1>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("🖼️ Profile Picture")
            
            # Placeholder for profile picture
            st.markdown("""
            <div style="width: 150px; height: 150px; margin: 0 auto; 
                        background: linear-gradient(45deg, #667eea, #764ba2);
                        border-radius: 50%; display: flex; align-items: center;
                        justify-content: center; color: white; font-size: 3rem;">
                👤
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Upload Photo", use_container_width=True):
                st.info("Upload feature coming soon!")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("📝 Profile Information")
            
            # Profile form
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("First Name", value="John")
                st.text_input("Email", value="john.student@ummgl.ac.id")
                st.selectbox("Study Program", ["Teknologi Informasi", "Teknik Informatika", "Farmasi"])
            
            with col2:
                st.text_input("Last Name", value="Doe")
                st.text_input("Student ID", value="20230001")
                st.selectbox("Semester", list(range(1, 9)), index=4)
            
            st.text_area("Bio", "AI enthusiast and data science student. Passionate about machine learning and its applications in education.")
            
            if st.button("💾 Update Profile", use_container_width=True, type="primary"):
                st.success("Profile updated successfully!")
            
            st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main function"""
    
    # Check if logged in
    if not st.session_state.logged_in:
        login_page()
    else:
        # Display appropriate dashboard
        if st.session_state.user_role == 'admin':
            admin_dashboard()
        else:
            student_dashboard()
        
        # Footer
        st.markdown("""
        <div class="footer">
            <p>🤖 AI Analytics Platform v1.0 • UMMgl Data Science Initiative • © 2024</p>
            <p style="font-size: 0.8rem; color: #94a3b8;">
            Powered by Streamlit • Random Forest Algorithm • Machine Learning
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
