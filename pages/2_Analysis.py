import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
from sklearn.decomposition import FactorAnalysis
import numpy as np



st.title("📊 Analisis Lanjutan (Korelasi & Regresi Logistik)")

# =====================================================
# PILIH SUMBER DATA (SIDEBAR)
# =====================================================
st.sidebar.header("⚙ Pengaturan Analisis")

data_source = st.sidebar.radio(
    "Pilih data yang digunakan:",
    ("Gunakan Data Binary", "Upload Data Baru")
)
df = None

# =====================================================
# OPSI 1 — DATA BINARY DARI SESSION
# =====================================================
if data_source == "Gunakan Data Binary":

    if "binary_output" not in st.session_state:
        st.warning("⚠️ Data binary belum diproses. Silakan jalankan halaman Binary Processor.")
        st.stop()

    df = st.session_state["binary_output"]
    st.success("Menggunakan data binary dari halaman sebelumnya.")
    st.dataframe(df.head())

# =====================================================
# OPSI 2 — UPLOAD DATA
# =====================================================
else:
    uploaded = st.sidebar.file_uploader("Upload CSV/Excel:", type=["csv", "xlsx"])

    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)

            st.success("Data berhasil diupload.")
            st.dataframe(df.head())

        except Exception as e:
            st.error(f"Gagal membaca data: {e}")
            st.stop()
    else:
        st.info("Silakan upload file untuk analisis.")
        st.stop()

# =====================================================
# DAPATKAN KOLOM NUMERIK
# =====================================================
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

# =====================================================
# PILIH METODE ANALISIS DI SIDEBAR
# =====================================================
st.sidebar.subheader("📌 Pilih Metode Analisis")

metode = st.sidebar.multiselect(
    "Metode yang digunakan:",
    ["Korelasi", "Regresi Logistik", "Analisis Faktor"]
)
st.divider()

# ------------------------------ KORELASI ------------------------------
if "Korelasi" in metode:

    st.sidebar.subheader("🔗 Pengaturan Korelasi")
    kolom_korelasi = st.sidebar.multiselect(
        "Pilih kolom untuk korelasi:",
        numeric_cols,
        key="korelasi_cols"
    )

    if st.sidebar.button("📊 Jalankan Korelasi"):

        st.subheader("🔗 Hasil Korelasi Variabel")

        if len(kolom_korelasi) < 2:
            st.error("Pilih minimal 2 kolom untuk korelasi.")
        else:
            corr = df[kolom_korelasi].corr()
            st.dataframe(corr.style.background_gradient(cmap="Blues"))


# ------------------------------ REGRESI LOGISTIK ------------------------------
if "Regresi Logistik" in metode:

    st.sidebar.subheader("📈 Pengaturan Regresi Logistik")
    
    target = st.sidebar.selectbox(
        "Pilih kolom Target (Y):",
        numeric_cols,
        key="reg_target"
    )

    fitur = st.sidebar.multiselect(
        "Pilih kolom Fitur (X):",
        numeric_cols,
        key="reg_fitur"
    )

    if st.sidebar.button("🚀 Jalankan Regresi Logistik"):

        st.subheader("📈 Hasil Regresi Logistik & Random Forest")

        if len(fitur) == 0:
            st.error("Pilih minimal satu fitur.")
            st.stop()

        try:
            # =========================================
            # 1. DATASET
            # =========================================
            X = df[fitur]
            y = df[target]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.20, random_state=42
            )
            
            # =========================================
            # 2. REGRESI LOGISTIK
            # =========================================
            log_model = LogisticRegression()
            log_model.fit(X_train, y_train)
            y_pred_log = log_model.predict(X_test)

            acc = accuracy_score(y_test, y_pred_log)

            # =========================================
            # 3. RANDOM FOREST
            # =========================================
            rf_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                random_state=42
            )
            rf_model.fit(X_train, y_train)
            y_pred_rf = rf_model.predict(X_test)

            r2 = r2_score(y_test, y_pred_rf)

            # =========================================
            # 4. TABEL HASIL TANPA CROSS-VALIDATION
            # =========================================
            rows = [
                {
                    "Keterangan": "Akurasi / R²",
                    "Logistic": acc,
                    "RandomForest": r2
                }
            ]

            for i, f in enumerate(fitur):
                rows.append({
                    "Keterangan": f,
                    "Logistic": log_model.coef_[0][i],
                    "RandomForest": rf_model.feature_importances_[i]
                })

            result_df = pd.DataFrame(rows)

            st.write("### 📘 Tabel Hasil Regresi")
            st.dataframe(result_df, use_container_width=True)

        except Exception as e:
            st.error(f"Kesalahan dalam regresi: {e}")


# --------------- ANALISIS FAKTOR -----------------

def factanal(X, n_components=1):
    transformer = FactorAnalysis(n_components=n_components, random_state=0)
    X_transformed = transformer.fit_transform(X)
    eigenvalues = transformer.noise_variance_  # proxy untuk eigenvalues
    loadings = transformer.components_
    return X_transformed, eigenvalues, loadings

if "Analisis Faktor" in metode:

    st.sidebar.subheader("🧮 Pengaturan Analisis Faktor Multi-Group")

    # Masukkan nama grup
    grup_names = st.sidebar.text_area(
        "Masukkan nama grup (pisahkan dengan koma):",
        "aware,usage,current,relevance,image,disposition,future"
    ).split(',')

    # Dictionary untuk menampung kolom tiap grup
    grup_dict = {}

    st.sidebar.markdown("### Pilih kolom untuk tiap grup")
    for grup in grup_names:
        grup = grup.strip()
        if grup:
            kolom = st.sidebar.multiselect(
                f"Kolom untuk grup '{grup}':",
                numeric_cols,
                key=f"grup_{grup}"
            )
            if kolom:
                grup_dict[grup] = kolom

    n_components = st.sidebar.number_input(
        "Jumlah faktor per grup (default 1):",
        min_value=1, max_value=5,
        value=1
    )

    if st.sidebar.button("🔎 Jalankan Analisis Faktor"):

        hasil_gabungan = df.copy()

        for grup, kolom in grup_dict.items():
            if len(kolom) < 2:
                st.warning(f"Grup '{grup}' memiliki kurang dari 2 kolom, dilewati.")
                continue

            data_faktor = df[kolom].dropna()
            try:
                scores, eigen, loadings = factanal(data_faktor, n_components=n_components)

                # Factor loadings
                loading_df = pd.DataFrame(
                    loadings.T,
                    index=kolom,
                    columns=[f"{grup}_Faktor{i+1}" for i in range(n_components)]
                )
                st.write(f"### 📘 Factor Loadings - Grup {grup}")
                st.dataframe(loading_df.style.background_gradient(cmap="Blues"))

                # Factor scores
                factor_df = pd.DataFrame(
                    scores,
                    columns=[f"{grup}_Faktor{i+1}" for i in range(n_components)],
                    index=data_faktor.index
                )
                
                # Tambahkan ke dataframe utama
                hasil_gabungan = pd.concat([hasil_gabungan, factor_df], axis=1)

            except Exception as e:
                st.error(f"Grup '{grup}' gagal dianalisis: {e}")

        st.write("### 🟩 Factor Scores per Observasi (Semua Grup)")
        st.dataframe(hasil_gabungan)

