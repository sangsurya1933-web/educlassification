import streamlit as st
import pandas as pd
import numpy as np

# Konfigurasi halaman
st.set_page_config(page_title="Monitor AI Mahasiswa", layout="centered")

# CSS minimal
st.markdown("""
<style>
    .card { padding: 20px; border-radius: 10px; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 10px 0; }
    .aman { background: #d4edda; border-left: 5px solid #28a745; padding: 10px; }
    .teguran { background: #fff3cd; border-left: 5px solid #ffc107; padding: 10px; }
    .pengawasan { background: #f8d7da; border-left: 5px solid #dc3545; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# Dataset sederhana
def buat_data():
    np.random.seed(42)
    data = {
        'Nama': [f'Siswa {i}' for i in range(1, 51)],
        'Jam_AI': np.random.uniform(5, 35, 50).round(1),
        'IPK': np.random.uniform(2.0, 4.0, 50).round(2),
    }
    df = pd.DataFrame(data)
    
    # Tentukan level
    def tentukan_level(jam):
        if jam <= 10: return 'AMAN'
        elif jam <= 20: return 'TEGURAN'
        else: return 'PENGAWASAN'
    
    df['Level'] = df['Jam_AI'].apply(tentukan_level)
    return df

# ===== DASHBOARD =====
st.title("🎓 Monitor Penggunaan AI")

pilihan = st.sidebar.selectbox("Pilih Dashboard:", ["Utama", "Guru", "Siswa"])

if pilihan == "Utama":
    st.subheader("Selamat Datang")
    st.write("Pilih dashboard di sidebar:")
    st.write("- **Guru**: Analisis data dan rekomendasi")
    st.write("- **Siswa**: Cek status dan rekomendasi")

elif pilihan == "Guru":
    st.subheader("📊 Dashboard Guru")
    
    df = buat_data()
    
    tab1, tab2, tab3 = st.tabs(["Data", "Analisis", "Rekomendasi"])
    
    with tab1:
        st.write("Data Mahasiswa:")
        st.dataframe(df)
        
        # Statistik
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Siswa", len(df))
        with col2:
            st.metric("Rata Jam AI", f"{df['Jam_AI'].mean():.1f}")
    
    with tab2:
        level_count = df['Level'].value_counts()
        st.bar_chart(level_count)
        
        # Distribusi
        st.write("Distribusi Level:")
        for level in ['AMAN', 'TEGURAN', 'PENGAWASAN']:
            count = (df['Level'] == level).sum()
            st.write(f"- {level}: {count} siswa ({count/len(df)*100:.0f}%)")
    
    with tab3:
        st.write("Rekomendasi:")
        
        if st.button("Hasilkan Rekomendasi"):
            for level in ['AMAN', 'TEGURAN', 'PENGAWASAN']:
                count = (df['Level'] == level).sum()
                with st.expander(f"Level {level} ({count} siswa)"):
                    if level == 'AMAN':
                        st.write("✅ Pertahankan, monitoring rutin")
                    elif level == 'TEGURAN':
                        st.write("⚠️ Beri peringatan, pantau mingguan")
                    else:
                        st.write("🚨 Intervensi langsung, konseling")

else:  # Siswa
    st.subheader("👨‍🎓 Dashboard Siswa")
    
    df = buat_data()
    
    nama = st.selectbox("Pilih Nama:", df['Nama'])
    data = df[df['Nama'] == nama].iloc[0]
    
    st.write(f"**Nama:** {data['Nama']}")
    st.write(f"**Jam AI/Minggu:** {data['Jam_AI']}")
    st.write(f"**IPK:** {data['IPK']}")
    
    # Tampilkan level
    level = data['Level']
    st.write(f"**Level:** {level}")
    
    if level == 'AMAN':
        st.markdown('<div class="aman">✅ Penggunaan wajar. Pertahankan!</div>', unsafe_allow_html=True)
        st.write("Rekomendasi: Lanjutkan pola sehat")
    elif level == 'TEGURAN':
        st.markdown('<div class="teguran">⚠️ Penggunaan berlebihan. Kurangi!</div>', unsafe_allow_html=True)
        st.write("Rekomendasi: Maksimal 15 jam/minggu")
    else:
        st.markdown('<div class="pengawasan">🚨 Penggunaan sangat berlebihan!</div>', unsafe_allow_html=True)
        st.write("Rekomendasi: Konsultasi dengan dosen")
