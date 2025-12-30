def validate_dataset(df):
    """Validasi dataset yang diupload"""
    required_columns = ['Nama', 'Studi_Jurusan', 'Semester', 'AI_Tools', 'Trust_Level', 'Usage_Intensity_Score']
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Kolom yang hilang: {', '.join(missing_columns)}"
    
    # Check data types
    try:
        df['Semester'] = pd.to_numeric(df['Semester'])
        df['Trust_Level'] = pd.to_numeric(df['Trust_Level'])
        df['Usage_Intensity_Score'] = df['Usage_Intensity_Score'].astype(str).apply(
            lambda x: 10 if x.strip() == '10+' else float(x)
        )
    except Exception as e:
        return False, f"Error konversi tipe data: {str(e)}"
    
    # Check for empty values
    if df[required_columns].isnull().any().any():
        return False, "Dataset mengandung nilai kosong (null)"
    
    # Validate score range
    invalid_scores = df[~df['Usage_Intensity_Score'].between(1, 10)]
    if not invalid_scores.empty:
        return False, f"Terdapat {len(invalid_scores)} data dengan skor di luar range 1-10"
    
    return True, "Dataset valid"
