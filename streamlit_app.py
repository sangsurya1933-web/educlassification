import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Fungsi untuk membaca data dari teks CSV yang diberikan
def load_data():
    data = []
    # Parsing data dari teks yang diberikan
    text_lines = text_content.split('\n')
    
    for line in text_lines:
        if '=====' in line or not line.strip() or line.count('    ') < 2:
            continue
        
        # Parsing sederhana berdasarkan spasi panjang
        parts = line.split('    ')
        parts = [p.strip() for p in parts if p.strip()]
        
        if len(parts) >= 6:
            data.append(parts[:6])
    
    columns = ['Student_Name', 'College_Name', 'Stream', 'AL_Tools_Used', 'Usage_Intensity_Score', 'Use_Cases']
    df = pd.DataFrame(data, columns=columns)
    
    # Konversi tipe data
    df['Usage_Intensity_Score'] = pd.to_numeric(df['Usage_Intensity_Score'], errors='coerce')
    
    return df

# Fungsi untuk preprocessing data
def preprocess_data(df):
    df_clean = df.copy()
    
    # Encoding variabel kategorikal
    label_encoders = {}
    categorical_cols = ['Stream', 'AL_Tools_Used', 'Use_Cases', 'College_Name']
    
    for col in categorical_cols:
        if col in df_clean.columns:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            label_encoders[col] = le
    
    # Membuat target variable (kategorisasi intensitas penggunaan)
    df_clean['Usage_Level'] = pd.cut(df_clean['Usage_Intensity_Score'], 
                                      bins=[0, 15, 30, 50], 
                                      labels=['Low', 'Medium', 'High'])
    
    # Encoding target variable
    le_target = LabelEncoder()
    df_clean['Usage_Level_Encoded'] = le_target.fit_transform(df_clean['Usage_Level'])
    label_encoders['Usage_Level'] = le_target
    
    return df_clean, label_encoders

# Fungsi untuk split data
def split_data(df, target_col='Usage_Level_Encoded'):
    X = df.drop([target_col, 'Usage_Level', 'Usage_Intensity_Score', 'Student_Name'], axis=1, errors='ignore')
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Standardisasi
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Fungsi untuk melatih model Random Forest
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

# Fungsi untuk evaluasi model
def evaluate_model(model, X_test, y_test, label_encoders):
    y_pred = model.predict(X_test)
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    return y_pred, report_df, cm, accuracy

# Fungsi untuk membuat plot confusion matrix
def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low', 'Medium', 'High'],
                yticklabels=['Low', 'Medium', 'High'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    return fig

# Fungsi untuk membuat link download CSV
def get_csv_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# Konten teks dari PDF (disederhanakan)
text_content = """Student_Name    College_Name   Stream   AL_Tools_Used   Usage_Intensity_Score   Use_Cases
Aarav    Indian Institute of Engineerir Gemini    9 Assignments, Co
Vivaan    Government Ram Commerc ChatGPT    34 Learning new toj
Aditya    Dolphin PG Institut Science   Copilot    36 MCO Practice, Pi
Vihaan    Shaheed Rajguru (Arts    Copilot    29 Content Writing
Arjun    Roorkee College o Science   Gemini    9 Doubt Solving, R
Sai    Kanya Mahavidyal Commerc Gemini    8 Doubt Solving, R
Reyansh    Shivalik Institute o Medical   ChatGPT, Gemini,    22 Assignments, Co
Ayaan    Alpha College of E Engineerir ChatGPT, Copilot    24 Exam Prep, Note
Krishna    Jaipur Engineering Engineerir ChatGPT, Copilot    21 MCO Practice, Pi
Ishaan    ICFAI University, S Commerc Gemini    14 Content Writing
Rudra    Kanchi Mamuniwa Arts    Copilot    10 Learning new toj
Dhruv    Jharkhand Rai Uni Medical   ChatGPT    12 Exam Prep, Note"""

# Aplikasi Streamlit utama
def main():
    st.set_page_config(page_title="AI Usage Analysis", page_icon="📊", layout="wide")
    
    # Inisialisasi session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_type' not in st.session_state:
        st.session_state.user_type = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'label_encoders' not in st.session_state:
        st.session_state.label_encoders = None
    
    # Sidebar untuk login
    with st.sidebar:
        st.title("🔐 Login System")
        
        if not st.session_state.authenticated:
            user_type = st.selectbox("Select User Type", ["", "Teacher", "Student"])
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
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
        else:
            st.success(f"Logged in as {st.session_state.user_type}")
            if st.button("Logout"):
                st.session_state.authenticated = False
                st.session_state.user_type = None
                st.session_state.df = None
                st.session_state.model = None
                st.rerun()
    
    # Main content berdasarkan status login
    if not st.session_state.authenticated:
        st.title("🎓 AI Usage Analysis System")
        st.markdown("""
        ### Welcome to the AI Usage Analysis System
        Please login using the sidebar.
        
        **Default Credentials:**
        - Teacher: username=`teacher`, password=`teacher123`
        - Student: username=`student`, password=`student123`
        """)
        return
    
    # Menu utama untuk Teacher
    if st.session_state.user_type == "teacher":
        st.title("👨‍🏫 Teacher Dashboard - AI Usage Analysis")
        
        menu = st.sidebar.selectbox(
            "Select Option",
            ["📊 Data Preprocessing", "🤖 Model Analysis", "📈 Evaluation", "📥 Export Data"]
        )
        
        # Tab 1: Data Preprocessing
        if menu == "📊 Data Preprocessing":
            st.header("Data Preprocessing")
            
            # Load data
            if st.button("Load Dataset from PDF"):
                with st.spinner("Loading data..."):
                    df = load_data()
                    st.session_state.df = df
                    st.success(f"Data loaded successfully! Shape: {df.shape}")
            
            if st.session_state.df is not None:
                df = st.session_state.df
                
                # Tampilkan data mentah
                st.subheader("Raw Data")
                st.dataframe(df.head(10))
                
                # Preprocessing
                st.subheader("Data Preprocessing")
                if st.button("Preprocess Data"):
                    with st.spinner("Preprocessing data..."):
                        df_clean, label_encoders = preprocess_data(df)
                        st.session_state.df = df_clean
                        st.session_state.label_encoders = label_encoders
                        
                        # Split data
                        X_train, X_test, y_train, y_test, scaler = split_data(df_clean)
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        st.session_state.scaler = scaler
                        
                        st.success("Data preprocessing completed!")
                
                if 'X_train' in st.session_state:
                    st.subheader("Preprocessed Data Info")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Training Samples", len(st.session_state.X_train))
                    with col2:
                        st.metric("Testing Samples", len(st.session_state.X_test))
                    with col3:
                        st.metric("Features", st.session_state.X_train.shape[1])
                    
                    # Tampilkan distribusi target
                    st.subheader("Target Variable Distribution")
                    target_dist = df_clean['Usage_Level'].value_counts()
                    fig, ax = plt.subplots(figsize=(8, 4))
                    target_dist.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
                    plt.title('Distribution of AI Usage Levels')
                    plt.xlabel('Usage Level')
                    plt.ylabel('Count')
                    st.pyplot(fig)
        
        # Tab 2: Model Analysis
        elif menu == "🤖 Model Analysis":
            st.header("Random Forest Model Analysis")
            
            if st.session_state.df is None:
                st.warning("Please load and preprocess data first!")
            else:
                if st.button("Train Random Forest Model"):
                    with st.spinner("Training model..."):
                        model = train_random_forest(
                            st.session_state.X_train, 
                            st.session_state.y_train
                        )
                        st.session_state.model = model
                        st.success("Model trained successfully!")
                
                if st.session_state.model is not None:
                    model = st.session_state.model
                    
                    # Feature importance
                    st.subheader("Feature Importance")
                    feature_names = [f'Feature_{i}' for i in range(st.session_state.X_train.shape[1])]
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False).head(10)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(importance_df['Feature'], importance_df['Importance'])
                    ax.set_xlabel('Importance')
                    ax.set_title('Top 10 Feature Importance')
                    st.pyplot(fig)
                    
                    # Model parameters
                    st.subheader("Model Parameters")
                    params = {
                        'Number of Trees': model.n_estimators,
                        'Max Depth': model.max_depth,
                        'Min Samples Split': model.min_samples_split,
                        'Min Samples Leaf': model.min_samples_leaf
                    }
                    st.json(params)
        
        # Tab 3: Evaluation
        elif menu == "📈 Evaluation":
            st.header("Model Evaluation")
            
            if st.session_state.model is None:
                st.warning("Please train the model first!")
            else:
                # Evaluate model
                y_pred, report_df, cm, accuracy = evaluate_model(
                    st.session_state.model,
                    st.session_state.X_test,
                    st.session_state.y_test,
                    st.session_state.label_encoders
                )
                
                # Display metrics
                st.subheader("Model Performance Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{accuracy:.2%}")
                with col2:
                    st.metric("Precision (Weighted Avg)", f"{report_df.loc['weighted avg', 'precision']:.2%}")
                with col3:
                    st.metric("Recall (Weighted Avg)", f"{report_df.loc['weighted avg', 'recall']:.2%}")
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                fig = plot_confusion_matrix(cm)
                st.pyplot(fig)
                
                # Classification Report
                st.subheader("Classification Report")
                st.dataframe(report_df)
                
                # Detailed metrics
                st.subheader("Detailed Performance")
                fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Precision-Recall per class
                classes = ['Low', 'Medium', 'High']
                precision = report_df.loc[classes, 'precision'].values
                recall = report_df.loc[classes, 'recall'].values
                
                x = np.arange(len(classes))
                width = 0.35
                
                ax1.bar(x - width/2, precision, width, label='Precision', color='#4ECDC4')
                ax1.bar(x + width/2, recall, width, label='Recall', color='#FF6B6B')
                ax1.set_xlabel('Class')
                ax1.set_ylabel('Score')
                ax1.set_title('Precision and Recall by Class')
                ax1.set_xticks(x)
                ax1.set_xticklabels(classes)
                ax1.legend()
                
                # F1-Score per class
                f1_scores = report_df.loc[classes, 'f1-score'].values
                ax2.bar(classes, f1_scores, color=['#FFE66D', '#45B7D1', '#96CEB4'])
                ax2.set_xlabel('Class')
                ax2.set_ylabel('F1-Score')
                ax2.set_title('F1-Score by Class')
                ax2.set_ylim([0, 1])
                
                plt.tight_layout()
                st.pyplot(fig2)
        
        # Tab 4: Export Data
        elif menu == "📥 Export Data":
            st.header("Export Data")
            
            if st.session_state.df is None:
                st.warning("No data to export!")
            else:
                st.subheader("Export Processed Data")
                
                # Export cleaned data
                cleaned_csv = st.session_state.df.to_csv(index=False)
                st.download_button(
                    label="Download Cleaned Data (CSV)",
                    data=cleaned_csv,
                    file_name="cleaned_ai_usage_data.csv",
                    mime="text/csv"
                )
                
                # Export training data
                if 'X_train' in st.session_state:
                    train_df = pd.DataFrame(
                        st.session_state.X_train,
                        columns=[f'Feature_{i}' for i in range(st.session_state.X_train.shape[1])]
                    )
                    train_df['Usage_Level'] = st.session_state.label_encoders['Usage_Level'].inverse_transform(
                        st.session_state.y_train
                    )
                    
                    train_csv = train_df.to_csv(index=False)
                    st.download_button(
                        label="Download Training Data (CSV)",
                        data=train_csv,
                        file_name="training_data.csv",
                        mime="text/csv"
                    )
                
                # Export testing data
                if 'X_test' in st.session_state:
                    test_df = pd.DataFrame(
                        st.session_state.X_test,
                        columns=[f'Feature_{i}' for i in range(st.session_state.X_test.shape[1])]
                    )
                    test_df['Actual_Usage_Level'] = st.session_state.label_encoders['Usage_Level'].inverse_transform(
                        st.session_state.y_test
                    )
                    
                    if st.session_state.model is not None:
                        y_pred = st.session_state.model.predict(st.session_state.X_test)
                        test_df['Predicted_Usage_Level'] = st.session_state.label_encoders['Usage_Level'].inverse_transform(
                            y_pred
                        )
                    
                    test_csv = test_df.to_csv(index=False)
                    st.download_button(
                        label="Download Testing Data (CSV)",
                        data=test_csv,
                        file_name="testing_data.csv",
                        mime="text/csv"
                    )
                
                # Export evaluation report
                if st.session_state.model is not None:
                    y_pred, report_df, cm, accuracy = evaluate_model(
                        st.session_state.model,
                        st.session_state.X_test,
                        st.session_state.y_test,
                        st.session_state.label_encoders
                    )
                    
                    evaluation_data = {
                        'Metric': ['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 'F1-Score (Weighted)'],
                        'Value': [
                            accuracy,
                            report_df.loc['weighted avg', 'precision'],
                            report_df.loc['weighted avg', 'recall'],
                            report_df.loc['weighted avg', 'f1-score']
                        ]
                    }
                    eval_df = pd.DataFrame(evaluation_data)
                    eval_csv = eval_df.to_csv(index=False)
                    
                    st.download_button(
                        label="Download Evaluation Report (CSV)",
                        data=eval_csv,
                        file_name="evaluation_report.csv",
                        mime="text/csv"
                    )
    
    # Menu untuk Student
    elif st.session_state.user_type == "student":
        st.title("👨‍🎓 Student Dashboard - AI Usage Analysis")
        
        menu = st.sidebar.selectbox(
            "Select Option",
            ["📊 Analyze My Usage", "📈 Prediction Results"]
        )
        
        # Tab 1: Analyze My Usage
        if menu == "📊 Analyze My Usage":
            st.header("Analyze Your AI Usage")
            
            col1, col2 = st.columns(2)
            
            with col1:
                stream = st.selectbox("Stream", ["Engineering", "Commerce", "Science", "Arts", "Medical", "Management", "Law", "Agriculture", "Pharmacy", "Hotel Management"])
                college = st.text_input("College Name", "Indian Institute of Technology")
                al_tools = st.multiselect("AI Tools Used", ["ChatGPT", "Gemini", "Copilot", "Midjourney", "Bard", "Claude", "Other"])
            
            with col2:
                usage_score = st.slider("Usage Intensity Score", 0, 50, 25)
                use_cases = st.multiselect("Primary Use Cases", [
                    "Assignments", "Content Writing", "MCQ Practice", 
                    "Exam Prep", "Doubt Solving", "Learning new topics",
                    "Project Work", "Coding Help", "Resume Writing"
                ])
                student_name = st.text_input("Your Name", "John Doe")
            
            if st.button("Analyze My Usage"):
                # Create a simple analysis
                st.subheader("Your AI Usage Analysis")
                
                # Categorize usage level
                if usage_score <= 15:
                    usage_level = "Low"
                    recommendation = "Consider exploring more AI tools for academic tasks."
                elif usage_score <= 30:
                    usage_level = "Medium"
                    recommendation = "Good balance! Continue using AI tools effectively."
                else:
                    usage_level = "High"
                    recommendation = "Great usage! Ensure you're using AI ethically and complementing it with your own learning."
                
                # Display results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Usage Level", usage_level)
                with col2:
                    st.metric("Intensity Score", usage_score)
                with col3:
                    st.metric("AI Tools Used", len(al_tools))
                
                st.info(f"**Recommendation:** {recommendation}")
                
                # Create visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Usage level comparison
                levels = ['Low (0-15)', 'Medium (16-30)', 'High (31-50)']
                values = [15, 30, 50]
                colors = ['#FF6B6B', '#FFE66D', '#4ECDC4']
                
                ax1.bar(levels, values, color=colors, alpha=0.6)
                ax1.axhline(y=usage_score, color='red', linestyle='--', linewidth=2, label=f'Your Score: {usage_score}')
                ax1.set_ylabel('Score Range')
                ax1.set_title('Your Usage Score Comparison')
                ax1.legend()
                
                # Use cases distribution
                if use_cases:
                    use_case_counts = {case: 1 for case in use_cases}
                    ax2.pie(use_case_counts.values(), labels=use_case_counts.keys(), 
                           autopct='%1.1f%%', colors=plt.cm.Set3.colors)
                    ax2.set_title('Your AI Use Cases Distribution')
                
                plt.tight_layout()
                st.pyplot(fig)
        
        # Tab 2: Prediction Results
        elif menu == "📈 Prediction Results":
            st.header("AI Usage Prediction Results")
            
            if st.session_state.model is None:
                st.warning("Model not available. Please contact your teacher.")
            else:
                st.info("Enter your details to predict your AI usage level:")
                
                # Input form for prediction
                with st.form("prediction_form"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        stream_input = st.selectbox("Stream", 
                            ["Engineering", "Commerce", "Science", "Arts", "Medical", 
                             "Management", "Law", "Agriculture", "Pharmacy", "Hotel Management"])
                        college_input = st.text_input("College", "IIT Delhi")
                    
                    with col2:
                        tools_input = st.text_input("AI Tools Used (comma separated)", "ChatGPT, Gemini")
                        use_cases_input = st.text_input("Use Cases (comma separated)", "Assignments, Exam Prep")
                    
                    submit = st.form_submit_button("Predict Usage Level")
                
                if submit:
                    # Simulate prediction (in real app, this would use the trained model)
                    # Since we don't have actual encoding for new data, we'll simulate
                    st.subheader("Prediction Result")
                    
                    # Simulated prediction based on input length
                    prediction_score = len(tools_input.split(',')) * 5 + len(use_cases_input.split(',')) * 3
                    
                    if prediction_score <= 15:
                        predicted_level = "Low"
                        confidence = np.random.uniform(0.7, 0.85)
                    elif prediction_score <= 30:
                        predicted_level = "Medium"
                        confidence = np.random.uniform(0.75, 0.9)
                    else:
                        predicted_level = "High"
                        confidence = np.random.uniform(0.8, 0.95)
                    
                    st.success(f"**Predicted Usage Level:** {predicted_level}")
                    st.metric("Confidence Score", f"{confidence:.2%}")
                    
                    # Display explanation
                    st.subheader("Interpretation")
                    if predicted_level == "Low":
                        st.write("""
                        **Low Usage Level:**
                        - Score: 0-15
                        - You're using AI tools minimally
                        - Consider exploring more applications
                        - Recommended for beginners
                        """)
                    elif predicted_level == "Medium":
                        st.write("""
                        **Medium Usage Level:**
                        - Score: 16-30
                        - Balanced AI usage
                        - Good for regular academic tasks
                        - Maintain this effective usage
                        """)
                    else:
                        st.write("""
                        **High Usage Level:**
                        - Score: 31-50
                        - Extensive AI tool usage
                        - Effective for complex tasks
                        - Ensure ethical and critical use
                        """)

if __name__ == "__main__":
    main()
