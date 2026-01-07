import streamlit as st
import pandas as pd
from io import BytesIO

st.title("🧮 Binary Processor")

# ============================
# FUNCTION PROSES PRODUK
# ============================

def proses_produk(data, input_produk, kolom_target):
    output = []

    for idx, row in data.iterrows():
        for merk in input_produk:

            hasil = {"index_asal": idx, "produk": merk}

            for kol in kolom_target:
                hasil[kol] = row.get(kol, None)

            for kol in kolom_target:
                try:
                    produk_list = [x.strip().lower() for x in str(row[kol]).split(";")]
                except:
                    produk_list = []

                count = sum(1 for p in produk_list if p.lower() == merk.lower())
                hasil[f"jumlah_{kol}"] = 1 if count > 0 else 0

            output.append(hasil)

    return pd.DataFrame(output)


# ============================
# STREAMLIT INPUT
# ============================

uploaded_file = st.sidebar.file_uploader("Upload Excel", type=["xlsx"])

if uploaded_file:

    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names

    selected_sheet = st.sidebar.selectbox("Pilih Sheet:", sheet_names)
    data = pd.read_excel(uploaded_file, sheet_name=selected_sheet)

    st.success(f"Sheet '{selected_sheet}' berhasil dibaca.")
    st.dataframe(data.head())

    semua_kolom = list(data.columns)

    # ✅ PILIH KOLOM PATOKAN
    kolom_patokan = st.sidebar.selectbox(
        "penentu baris proses (isi SERIAL / SBJNUM):",
        semua_kolom
    )

    jumlah_input = st.sidebar.number_input(
        "Jumlah Block Input:", min_value=1, max_value=20, value=1
    )

    input_kolom_list = []
    target_kolom_list = []

    for i in range(jumlah_input):
        st.sidebar.markdown(f"### Inputan {i+1}")

        kol_input = st.sidebar.selectbox(
            f"Kolom List Produk {i+1}:", semua_kolom, key=f"input_{i}"
        )
        input_kolom_list.append(kol_input)

        kol_target = st.sidebar.multiselect(
            f"Kolom Target {i+1}:", semua_kolom, key=f"target_{i}"
        )
        target_kolom_list.append(kol_target)

    if st.sidebar.button("🔄 Proses Data"):

        # ============================
        # FILTER DATA BERDASARKAN KOLOM PATOKAN
        # ============================
        data_patokan = data[
            data[kolom_patokan].notna()
            & (data[kolom_patokan].astype(str).str.strip() != "")
        ].copy()

        st.info(f"Jumlah baris diproses: {len(data_patokan)} (berdasarkan {kolom_patokan})")

        list_df_output = []

        for idx in range(jumlah_input):
            kol_input = input_kolom_list[idx]
            kol_target = target_kolom_list[idx]

            if len(kol_target) == 0:
                st.error(f"Kolom target untuk inputan {idx+1} belum dipilih!")
                st.stop()

            input_produk = (
                data[kol_input]
                .dropna()
                .astype(str)
                .str.lower()
                .tolist()
            )

            df = proses_produk(data_patokan, input_produk, kol_target)

            df = df.rename(columns={
                "produk": f"produk_{idx+1}",
                "index_asal": f"index_asal_{idx+1}"
            })

            list_df_output.append(df)

        hasil_final = pd.concat(list_df_output, axis=1)

        # ============================
        # OUTPUT
        # ============================
        st.session_state["binary_output"] = hasil_final
        st.dataframe(hasil_final)

        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            hasil_final.to_excel(writer, index=False, sheet_name="hasil")

        st.download_button(
            label="⬇️ Download Hasil Excel",
            data=buffer.getvalue(),
            file_name="hasil_binary.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:
    st.info("Upload file Excel terlebih dahulu.")
