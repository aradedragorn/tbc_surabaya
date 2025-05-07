import os
os.environ['GDAL_DATA'] = r'C:/Users/ASUS/anaconda3/envs/geo_env/Library/share/gdal'

import streamlit as st
import numpy as np
import geopandas as gpd
import folium
from folium import Choropleth, TileLayer, GeoJsonTooltip, FeatureGroup
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ================================
# Konfigurasi Streamlit
# ================================
st.set_page_config(layout="wide")
st.title("🧭 Dashboard Prediksi TBC Kota Surabaya 2024")

# ================================
# Load Data
# ================================
gdf_choro = gpd.read_file("Choroplet/Batas_ADM_Kecamatan_Surabaya.shp")
if gdf_choro.crs is None:
    gdf_choro.set_crs(epsg=4326, inplace=True)

gdf_heat = gpd.read_file("Heatmap/Centroid_Kecamatan_SBY.shp")
if gdf_heat.crs is None:
    gdf_heat.set_crs(epsg=4326, inplace=True)

df = pd.read_excel("Statistik/Hasil_Prediksi_TBC_Lengkap.xlsx")

# ================================
# Hitung Pusat Peta
# ================================
gdf_proj = gdf_choro.to_crs(epsg=32748)
centroid = gdf_proj.geometry.centroid.to_crs(epsg=4326)
center = [centroid.y.mean(), centroid.x.mean()]

# ================================
# Dropdown Choropleth
# ================================
choropleth_fields = [col for col in gdf_choro.columns if "_Pred" in col or col == "Aktual"]
heatmap_fields = [col for col in gdf_heat.columns if "_Pred" in col or col == "Aktual"]

# ================================
# Tab Navigasi
# ================================
tabs = st.tabs(["🗺️ Peta Interaktif", "📊 Statistik Model", "📋 Data Lengkap"])

# ================================
# Tab 1: Peta Interaktif
# ================================
with tabs[0]:
    choropleth_model = st.selectbox("Model Prediksi untuk Choropleth:", choropleth_fields)

    gdf_choro['Pred_display'] = gdf_choro[choropleth_model].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

    # Peta Choropleth
    m1 = folium.Map(location=center, zoom_start=11, tiles=None)
    TileLayer('CartoDB positron', name='Base').add_to(m1)
    TileLayer('Esri.WorldImagery', name='Citra Satelit').add_to(m1)

    Choropleth(
        geo_data=gdf_choro,
        data=gdf_choro,
        columns=["NAMOBJ", choropleth_model],
        key_on="feature.properties.NAMOBJ",
        fill_color="RdYlGn_r",
        fill_opacity=0.7,
        line_opacity=0.3,
        legend_name=f"Prediksi TBC ({choropleth_model})"
    ).add_to(m1)

    tooltip = GeoJsonTooltip(
        fields=["NAMOBJ",  'Pred_display'],
        aliases=["Kecamatan:", f"Prediksi TBC:"],
        localize=True,
        sticky=False,
        labels=True,
        style="background-color: white; color: #333333; font-size: 12px; padding: 4px;"
    )

    folium.GeoJson(
        data=gdf_choro,
        tooltip=tooltip,
        name="Tooltip Prediksi",
        style_function=lambda x: {
            'fillOpacity': 0,
            'color': 'black',
            'weight': 0.3
        }
    ).add_to(m1)

    folium.LayerControl(collapsed=False).add_to(m1)

    # Peta Heatmap
    m2 = folium.Map(location=center, zoom_start=11, tiles="Esri.WorldImagery")
    for field in heatmap_fields:
        heat_data = [
            [point.y, point.x, weight]
            for point, weight in zip(gdf_heat.geometry, gdf_heat[field])
            if not pd.isna(weight)
        ]
        fg = folium.FeatureGroup(name=f"Heatmap: {field}", show=False)
        HeatMap(heat_data, radius=25, blur=15, max_zoom=13).add_to(fg)
        fg.add_to(m2)

    folium.LayerControl(collapsed=False).add_to(m2)

    col1, col2 = st.columns(2)

    with st.container():
        with col1:
            st.subheader("🟡 Peta Choropleth TBC per Kecamatan")
            st_folium(m1, width=750, height=500)

        with col2:
            st.subheader("🟡 Peta Heatmap TBC per Kecamatan")
            st_folium(m2, width=750, height=500)

# ================================
# Tab 2: Statistik
# ================================
with tabs[1]:
    def model_metrics(y_true, y_pred):
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2 Score": r2_score(y_true, y_pred)
        }

    models = ["NB", "RF", "XGB"]
    metrics = {model: model_metrics(df["Actual"], df[f"{model}_Pred"]) for model in models}

    st.markdown("### 📈 Akurasi Model")
    metrics_df = pd.DataFrame(metrics).T.rename(index={"NB": "Negative Binomial", "RF": "Random Forest", "XGB": "XGBoost"})
    col1, col2, col3 = st.columns(3)
    col1.metric("📉 MAE Terendah", metrics_df["MAE"].idxmin(), f"{metrics_df['MAE'].min():.2f}")
    col2.metric("🔁 RMSE Terendah", metrics_df["RMSE"].idxmin(), f"{metrics_df['RMSE'].min():.2f}")
    col3.metric("📈 R² Tertinggi", metrics_df["R2 Score"].idxmax(), f"{metrics_df['R2 Score'].max():.2f}")

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

# ================================
# Tab 3: Data Lengkap
# ================================
with tabs[2]:
    st.markdown("### 📋 Data Lengkap Prediksi dan Error")
    with st.expander("📁 Klik untuk menampilkan seluruh data"):
        st.dataframe(df.drop(columns=["GWR_Pred", "GWR_Error", "GWR_LocalR2"]))
