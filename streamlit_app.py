import streamlit as st
import pandas as pd

# Setup halaman
st.set_page_config(page_title="AI Analyzer", layout="centered")

# Knowledge Base
KNOWLEDGE = {
    'RENDAH': {'warna': 'hijau', 'level': 'AMAN', 'rekom': 'Penggunaan wajar'},
    'SEDANG': {'warna': 'kuning', 'level': 'WASPADA', 'rekom': 'Batasi penggunaan'},
    'TINGGI': {'warna': 'merah', 'level': 'BAHAYA', 'rekom': 'Butuh pengawasan'}
}

# Halaman Utama
st.title("🤖 Analisis AI Mahasiswa")
st.write("Sistem sederhana analisis penggunaan AI")

# Pilih peran
st.subheader("Pilih Peran:")
col1, col2 = st.columns(2)

with col1:
    if st.button("👨‍🏫 Guru"):
        st.session_state.page = "guru"

with col2:
    if st.button("👨‍🎓 Mahasiswa"):
        st.session_state.page = "mahasiswa"

# Inisialisasi
if 'page' not in st.session_state:
    st.session_state.page = None

# Halaman Guru
if st.session_state.page == "guru":
    st.header("Dashboard Guru")
    
    # Upload data
    uploaded_file = st.file_uploader("Upload data CSV", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("**Data Mahasiswa:**")
        st.dataframe(df)
    else:
        # Data contoh
        df = pd.DataFrame({
            'Nama': ['Andi', 'Budi', 'Cici', 'Dedi'],
            'AI_Jam': [5, 15, 3, 25],
            'IPK': [3.5, 3.2, 3.8, 2.9]
        })
        st.write("**Data Contoh:**")
        st.dataframe(df)
    
    # Analisis sederhana
    st.divider()
    st.subheader("Hasil Analisis")
    
    # Kategorisasi manual
    def kategorisasi(jam):
        if jam < 10:
            return 'RENDAH'
        elif jam < 20:
            return 'SEDANG'
        else:
            return 'TINGGI'
    
    if 'AI_Jam' in df.columns:
        df['Kategori'] = df['AI_Jam'].apply(kategorisasi)
        
        # Hitung jumlah per kategori
        hasil = df['Kategori'].value_counts()
        
        for kategori in ['RENDAH', 'SEDANG', 'TINGGI']:
            if kategori in hasil:
                jumlah = hasil[kategori]
                info = KNOWLEDGE[kategori]
                st.write(f"**{info['level']}** ({jumlah} siswa): {info['rekom']}")

# Halaman Mahasiswa
elif st.session_state.page == "mahasiswa":
    st.header("Dashboard Mahasiswa")
    
    # Input data
    nama = st.text_input("Nama Mahasiswa")
    jam_ai = st.slider("Jam penggunaan AI per minggu", 0, 40, 15)
    
    if st.button("Analisis"):
        # Tentukan kategori
        if jam_ai < 10:
            kategori = 'RENDAH'
        elif jam_ai < 20:
            kategori = 'SEDANG'
        else:
            kategori = 'TINGGI'
        
        info = KNOWLEDGE[kategori]
        
        # Tampilkan hasil
        st.divider()
        st.subheader(f"Hasil untuk {nama}")
        
        # Kartu hasil
        if info['warna'] == 'hijau':
            st.success(f"**{info['level']}** - {info['rekom']}")
        elif info['warna'] == 'kuning':
            st.warning(f"**{info['level']}** - {info['rekom']}")
        else:
            st.error(f"**{info['level']}** - {info['rekom']}")
        
        # Tips
        st.divider()
        st.write("**Tips:**")
        if kategori == 'RENDAH':
            st.write("- Pertahankan penggunaan AI yang sehat")
        elif kategori == 'SEDANG':
            st.write("- Batasi penggunaan AI untuk tugas tertentu")
            st.write("- Tingkatkan kemampuan analisis mandiri")
        else:
            st.write("- Kurangi penggunaan AI secara bertahap")
            st.write("- Konsultasi dengan dosen pembimbing")

# Footer
st.divider()
st.write("© 2024 - Sistem Analisis AI Mahasiswa")

# Tombol reset
if st.session_state.page:
    if st.button("Kembali ke Menu Utama"):
        st.session_state.page = None
        st.rerun()
