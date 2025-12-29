import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Set halaman
st.set_page_config(
    page_title="Analisis AI Mahasiswa",
    page_icon="🤖",
    layout="centered"
)

# ============================================
# KNOWLEDGE BASE SEDERHANA
# ============================================
class SimpleKnowledgeBase:
    def __init__(self):
        self.rules = {
            'RENDAH': {
                'level': 'AMAN',
                'warna': '🟢',
                'rekomendasi': 'Tidak perlu tindakan khusus',
                'deskripsi': 'Penggunaan AI dalam batas wajar'
            },
            'SEDANG': {
                'level': 'WASPADA',
                'warna': '🟡',
                'rekomendasi': 'Perlu konsultasi dengan dosen',
                'deskripsi': 'Mulai bergantung pada AI'
            },
            'TINGGI': {
                'level': 'BAHAYA',
                'warna': '🔴',
                'rekomendasi': 'Butuh program pembatasan AI',
                'deskripsi': 'Ketergantungan tinggi pada AI'
            }
        }
    
    def get_info(self, kategori):
        return self.rules.get(kategori, {})

# ============================================
# FUNGSI UTAMA
# ============================================
def show_login():
    """Halaman login sederhana"""
    st.title("🤖 Analisis AI Mahasiswa")
    st.write("Sistem analisis penggunaan Artificial Intelligence terhadap performa akademik")
    st.write("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Login sebagai Guru", use_container_width=True):
            st.session_state.page = "guru"
            st.rerun()
    
    with col2:
        if st.button("Login sebagai Mahasiswa", use_container_width=True):
            st.session_state.page = "mahasiswa"
            st.rerun()

def show_guru_page():
    """Halaman guru"""
    st.title("👨‍🏫 Dashboard Guru")
    st.write("---")
    
    # Menu sidebar sederhana
    menu = st.radio("Pilih menu:", 
                   ["1. Upload Data", "2. Preprocessing", "3. Analisis AI", "4. Hasil"])
    
    if menu == "1. Upload Data":
        st.subheader("📁 Upload Data Mahasiswa")
        
        # Upload file
        uploaded_file = st.file_uploader("Pilih file CSV", type=['csv'])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success(f"Data berhasil diupload: {len(df)} mahasiswa")
            
            with st.expander("Lihat data"):
                st.dataframe(df)
            
            st.session_state.df = df
        else:
            # Buat data contoh
            st.info("Menggunakan data contoh")
            data = {
                'Nama': ['Andi', 'Budi', 'Cici', 'Dodi', 'Eka'],
                'IPK': [3.5, 3.2, 3.8, 2.9, 3.1],
                'Jam_Belajar': [25, 20, 30, 15, 18],
                'Frekuensi_AI': ['Rendah', 'Sedang', 'Rendah', 'Tinggi', 'Sedang'],
                'Nilai': [85, 78, 92, 65, 75]
            }
            df = pd.DataFrame(data)
            st.dataframe(df)
            st.session_state.df = df
    
    elif menu == "2. Preprocessing":
        st.subheader("🔧 Preprocessing Data")
        
        if 'df' not in st.session_state:
            st.warning("Silakan upload data terlebih dahulu")
            return
        
        df = st.session_state.df
        
        # Data cleaning sederhana
        st.write("**1. Data Cleaning**")
        if st.checkbox("Hapus data kosong"):
            df = df.dropna()
            st.write(f"Sisa data: {len(df)} baris")
        
        # Encoding
        st.write("**2. Encoding Data**")
        if 'Frekuensi_AI' in df.columns:
            le = LabelEncoder()
            df['AI_Encoded'] = le.fit_transform(df['Frekuensi_AI'])
            st.write("Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
        
        # Split data
        st.write("**3. Split Data**")
        
        if 'AI_Encoded' in df.columns:
            X = df[['IPK', 'Jam_Belajar']]
            y = df['AI_Encoded']
            
            test_size = st.slider("Persentase testing:", 10, 50, 30)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=42
            )
            
            st.write(f"Training: {len(X_train)} data")
            st.write(f"Testing: {len(X_test)} data")
            
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            
            st.success("Preprocessing selesai!")
    
    elif menu == "3. Analisis AI":
        st.subheader("🤖 Analisis dengan Random Forest")
        
        if 'X_train' not in st.session_state:
            st.warning("Silakan lakukan preprocessing terlebih dahulu")
            return
        
        if st.button("Jalankan Analisis", type="primary"):
            with st.spinner("Menganalisis..."):
                # Training model
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(st.session_state.X_train, st.session_state.y_train)
                
                # Prediksi
                y_pred = model.predict(st.session_state.X_test)
                
                # Hitung akurasi
                from sklearn.metrics import accuracy_score
                accuracy = accuracy_score(st.session_state.y_test, y_pred)
                
                st.session_state.model = model
                st.session_state.y_pred = y_pred
                st.session_state.accuracy = accuracy
                
                st.success(f"Analisis selesai! Akurasi: {accuracy:.0%}")
                
                # Tampilkan hasil prediksi
                st.write("**Hasil Prediksi:**")
                hasil_df = pd.DataFrame({
                    'IPK': st.session_state.X_test['IPK'],
                    'Jam_Belajar': st.session_state.X_test['Jam_Belajar'],
                    'Prediksi': y_pred
                })
                st.dataframe(hasil_df)
    
    elif menu == "4. Hasil":
        st.subheader("📊 Hasil & Rekomendasi")
        
        if 'model' not in st.session_state:
            st.warning("Silakan jalankan analisis terlebih dahulu")
            return
        
        # Tampilkan akurasi
        st.metric("Akurasi Model", f"{st.session_state.accuracy:.0%}")
        
        # Knowledge base
        kb = SimpleKnowledgeBase()
        
        # Analisis hasil
        st.write("**Analisis Berdasarkan Prediksi:**")
        
        pred_counts = pd.Series(st.session_state.y_pred).value_counts()
        
        for pred, count in pred_counts.items():
            # Map prediksi ke kategori
            kategori_map = {0: 'RENDAH', 1: 'SEDANG', 2: 'TINGGI'}
            kategori = kategori_map.get(pred, 'TIDAK_TAHU')
            
            info = kb.get_info(kategori)
            
            col1, col2 = st.columns([1, 3])
            with col1:
                st.write(f"**{info.get('warna', '⚪')} {kategori}**")
                st.write(f"{count} mahasiswa")
            with col2:
                st.write(f"Level: {info.get('level', '')}")
                st.write(f"Rekomendasi: {info.get('rekomendasi', '')}")
            st.write("---")
        
        # Rekomendasi umum
        st.info("""
        **Rekomendasi Sistem:**
        1. **Level AMAN (Rendah)**: Pertahankan penggunaan AI yang sehat
        2. **Level WASPADA (Sedang)**: Monitor penggunaan AI, berikan bimbingan
        3. **Level BAHAYA (Tinggi)**: Intervensi dan pembatasan penggunaan AI
        """)
    
    st.write("---")
    if st.button("Kembali ke Login"):
        st.session_state.page = "login"
        st.rerun()

def show_mahasiswa_page():
    """Halaman mahasiswa"""
    st.title("👨‍🎓 Dashboard Mahasiswa")
    st.write("---")
    
    # Input data mahasiswa
    st.subheader("Profil Saya")
    
    col1, col2 = st.columns(2)
    
    with col1:
        nama = st.text_input("Nama")
        ipk = st.number_input("IPK", min_value=0.0, max_value=4.0, value=3.0)
    
    with col2:
        jam_belajar = st.number_input("Jam Belajar/Minggu", min_value=0, max_value=60, value=20)
        frekuensi_ai = st.selectbox("Frekuensi Penggunaan AI", 
                                  ["Rendah", "Sedang", "Tinggi"])
    
    if st.button("Analisis Profil Saya", type="primary"):
        # Knowledge base
        kb = SimpleKnowledgeBase()
        info = kb.get_info(frekuensi_ai.upper())
        
        # Tampilkan hasil
        st.success("Analisis selesai!")
        st.write("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Nama", nama)
        
        with col2:
            st.metric("IPK", f"{ipk:.2f}")
        
        with col3:
            st.metric("Level AI", info.get('warna', '') + " " + info.get('level', ''))
        
        # Rekomendasi
        st.subheader("📝 Rekomendasi untuk Anda")
        st.write(info.get('deskripsi', ''))
        st.write(f"**Tindakan:** {info.get('rekomendasi', '')}")
        
        # Tips
        st.subheader("💡 Tips")
        tips = [
            "✓ Gunakan AI sebagai alat bantu belajar",
            "✓ Jangan gantikan pemikiran kritis dengan AI",
            "✓ Diskusikan tugas dengan dosen, bukan hanya dengan AI",
            "✓ Verifikasi informasi dari AI dengan sumber lain",
            "✓ Atur waktu penggunaan AI"
        ]
        
        for tip in tips:
            st.write(tip)
    
    st.write("---")
    if st.button("Kembali ke Login"):
        st.session_state.page = "login"
        st.rerun()

# ============================================
# APLIKASI UTAMA
# ============================================
def main():
    # Inisialisasi session state
    if 'page' not in st.session_state:
        st.session_state.page = "login"
    
    # Routing sederhana
    if st.session_state.page == "login":
        show_login()
    elif st.session_state.page == "guru":
        show_guru_page()
    elif st.session_state.page == "mahasiswa":
        show_mahasiswa_page()

if __name__ == "__main__":
    main()
