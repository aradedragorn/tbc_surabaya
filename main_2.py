import os
import json
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import folium
from folium import Choropleth, TileLayer, GeoJsonTooltip
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ================================ #
# Konfigurasi Awal
# ================================ #
st.set_page_config(layout="wide")
st.title("🧭 Dashboard Prediksi TBC Kota Surabaya 2024")

# ================================ #
# Load GeoJSON
# ================================ #
with open("Choroplet/Batas_ADM_Kecamatan_Surabaya.geojson") as f:
    choro_geojson = json.load(f)

with open("Heatmap/Centroid_Kecamatan_SBY.geojson") as f:
    heat_geojson = json.load(f)

# ================================ #
# Load Excel Data
# ================================ #
df = pd.read_excel("Statistik/Hasil_Prediksi_TBC_Lengkap.xlsx")

# ================================ #
# Pusat Peta (hardcode untuk Surabaya)
# ================================ #
center = [-7.2756, 112.6426]

# ================================ #
# Pilih Model Choropleth
# ================================ #
choropleth_fields = [c for c in df.columns if "_Pred" in c]
choropleth_model = st.selectbox("Model Prediksi untuk Choropleth:", choropleth_fields)

# Masukkan nilai prediksi ke properti GeoJSON
pred_dict = dict(zip(df["Kecamatan"], df[choropleth_model]))
for feat in choro_geojson["features"]:
    nama = feat["properties"]["NAMOBJ"]
    val  = pred_dict.get(nama)
    feat["properties"]["Prediksi_Display"] = f"{val:.2f}" if val is not None else "N/A"

# ================================ #
# Build koordinat tiap kecamatan utk heatmap
# ================================ #
coord_dict = {
    feat["properties"]["NAMOBJ"]: (
        feat["geometry"]["coordinates"][1],
        feat["geometry"]["coordinates"][0]
    )
    for feat in heat_geojson["features"]
}

# ================================ #
# Tab Navigasi
# ================================ #
tabs = st.tabs(["🗺️ Peta Interaktif", "📊 Statistik Model", "📋 Data Lengkap"])

# --- Tab 1: Peta Interaktif --- #
with tabs[0]:
    # Choropleth Map
    m1 = folium.Map(location=center, zoom_start=11, tiles=None)
    TileLayer("CartoDB positron", name="Base").add_to(m1)
    TileLayer("Esri.WorldImagery", name="Citra Satelit").add_to(m1)

    Choropleth(
        geo_data=choro_geojson,
        data=df,
        columns=["Kecamatan", choropleth_model],
        key_on="feature.properties.NAMOBJ",
        fill_color="RdYlGn_r",
        fill_opacity=0.7,
        line_opacity=0.3,
        legend_name=f"Prediksi TBC ({choropleth_model})"
    ).add_to(m1)

    tooltip = GeoJsonTooltip(
        fields=["NAMOBJ", "Prediksi_Display"],
        aliases=["Kecamatan:", "Prediksi TBC:"],
        localize=True,
        sticky=False,
        labels=True,
        style="background-color: white; color: #333333; font-size: 12px; padding: 4px;"
    )

    folium.GeoJson(
        data=choro_geojson,
        tooltip=tooltip,
        name="Tooltip Prediksi",
        style_function=lambda _: {
            "fillOpacity": 0,
            "color": "black",
            "weight": 0.3
        }
    ).add_to(m1)

    # Heatmap Map
    m2 = folium.Map(location=center, zoom_start=11, tiles="Esri.WorldImagery")
    heatmap_fields = [c for c in df.columns if "_Pred" in c or c == "Actual"]

    for field in heatmap_fields:
        heat_data = []
        for _, row in df.iterrows():
            kec = row["Kecamatan"]
            val = row.get(field, None)
            if pd.notna(val) and kec in coord_dict:
                lat, lon = coord_dict[kec]
                heat_data.append([lat, lon, float(val)])
        fg = folium.FeatureGroup(name=f"Heatmap: {field}", show=False)
        HeatMap(heat_data, radius=25, blur=15, max_zoom=13).add_to(fg)
        fg.add_to(m2)

    folium.LayerControl(collapsed=False).add_to(m1)
    folium.LayerControl(collapsed=False).add_to(m2)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🟡 Peta Choropleth TBC per Kecamatan")
        st_folium(m1, width=750, height=500)
    with col2:
        st.subheader("🟡 Peta Heatmap TBC per Kecamatan")
        st_folium(m2, width=750, height=500)

# --- Tab 2: Statistik Model --- #
with tabs[1]:
    def model_metrics(y_true, y_pred):
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2 Score": r2_score(y_true, y_pred)
        }

    models = ["NB", "RF", "XGB"]
    metrics = {m: model_metrics(df["Actual"], df[f"{m}_Pred"]) for m in models}

    st.markdown("### 📈 Akurasi Model")
    metrics_df = pd.DataFrame(metrics).T.rename(index={
        "NB":"Negative Binomial",
        "RF":"Random Forest",
        "XGB":"XGBoost"
    })
    c1, c2, c3 = st.columns(3)
    c1.metric("📉 MAE Terendah", metrics_df["MAE"].idxmin(), f"{metrics_df['MAE'].min():.2f}")
    c2.metric("🔁 RMSE Terendah", metrics_df["RMSE"].idxmin(), f"{metrics_df['RMSE'].min():.2f}")
    c3.metric("📈 R² Tertinggi", metrics_df["R2 Score"].idxmax(), f"{metrics_df['R2 Score'].max():.2f}")

    with st.expander("🔍 Lihat Tabel Evaluasi Lengkap"):
        st.dataframe(metrics_df.style.format("{:.2f}"))

    # Visualisasi per kecamatan
    st.markdown("### 📊 Visualisasi Nilai Aktual vs Prediksi per Kecamatan")

    col1, col2, col3 = st.columns(3)
    model_names = {"NB": "Negative Binomial", "RF": "Random Forest", "XGB": "XGBoost"}
    colors = {"Actual": "#6A5ACD", "Pred": "#00BFFF"}

    for model, col in zip(model_names.keys(), [col1, col2, col3]):
        fig, ax = plt.subplots(figsize=(4.5, 4))

        # Siapkan DataFrame panjang (long format) untuk seaborn
        df_long = pd.DataFrame({
            "Kecamatan": list(df["Kecamatan"]) * 2,
            "Tipe": ["Aktual"] * len(df) + ["Prediksi"] * len(df),
            "Jumlah Kasus": list(df["Actual"]) + list(df[f"{model}_Pred"])
        })

        sns.barplot(
            data=df_long,
            x="Kecamatan",
            y="Jumlah Kasus",
            hue="Tipe",
            palette={"Aktual": colors["Actual"], "Prediksi": colors["Pred"]},
            ax=ax
        )

        ax.set_title(f"{model_names[model]}", fontsize=12)
        ax.tick_params(axis='x', rotation=90, labelsize=7)
        ax.set_xlabel("")
        ax.set_ylabel("Jumlah Kasus TBC")
        ax.legend(title="")
        col.pyplot(fig)

# --- Tab 3: Data Lengkap --- #
with tabs[2]:
    st.markdown("### 📋 Data Lengkap Prediksi dan Error")
    with st.expander("📁 Klik untuk menampilkan seluruh data"):
        st.dataframe(
            df.drop(columns=["GWR_Pred", "GWR_Error", "GWR_LocalR2"], errors="ignore")
        )
