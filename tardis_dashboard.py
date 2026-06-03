#!/usr/bin/env python3
##
## EPITECH PROJECT, 2026
## Tardis
## File description:
## tardis_dashboard.py
##

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import predicter
import numpy as np
import joblib
from folium import Map
from folium.plugins import HeatMap
from sklearn.pipeline import Pipeline

# Month to prediction
months = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def get_month_num_from_str(month: str) -> int:
    for i in range(len(months)):
        if months[i] == month:
            return i + 1
    return -1


# Load data
df = pd.read_csv("cleaned_dataset.csv")
delay_col = "retard_moyen_tous_trains_arrivee"
global_average_delay = df[delay_col].mean()
df["date"] = pd.to_datetime(df["date"])
model_pipeline = joblib.load("model.pkl")
model = model_pipeline.named_steps["model"]

# Make a menu
st.sidebar.image("assets/logo.png")

st.sidebar.title("Menu")

if "menu" not in st.session_state:
    st.session_state.menu = "HOME"

sidebar_columns = st.sidebar.columns(1)
menu_col = sidebar_columns[0]

if menu_col.button("HOME", use_container_width=True):
    st.session_state.menu = "HOME"

if menu_col.button("Delay distribution", use_container_width=True):
    st.session_state.menu = "Delay distribution"

if menu_col.button("Summary statistics", use_container_width=True):
    st.session_state.menu = "Summary statistics"

if menu_col.button("Delay prediction", use_container_width=True):
    st.session_state.menu = "Delay prediction"

if menu_col.button("Correlation heatmap", use_container_width=True):
    st.session_state.menu = "Correlation heatmap"

if menu_col.button("Model explanation", use_container_width=True):
    st.session_state.menu = "Model explanation"

menu = st.session_state.menu

# Filters
st.sidebar.header("Filters")

departure_filter = st.sidebar.multiselect(
    "Departure station", sorted(df["gare_depart"].dropna().unique())
)

if departure_filter:
    available_arrivals = sorted(
        df[df["gare_depart"].isin(departure_filter)]["gare_arrivee"].dropna().unique()
    )
else:
    available_arrivals = sorted(df["gare_arrivee"].dropna().unique())

arrival_filter = st.sidebar.multiselect("Arrival station", available_arrivals)

min_date = df["date"].min().date()
max_date = df["date"].max().date()

date_range = st.sidebar.date_input(
    "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)

filtered_df = df.copy()

if departure_filter:
    filtered_df = filtered_df[filtered_df["gare_depart"].isin(departure_filter)]

if arrival_filter:
    filtered_df = filtered_df[filtered_df["gare_arrivee"].isin(arrival_filter)]

if len(date_range) == 2:
    start_date, end_date = date_range

    after_start = filtered_df["date"].dt.date >= start_date
    before_end = filtered_df["date"].dt.date <= end_date

    filtered_df = filtered_df[after_start & before_end]

if filtered_df.empty:
    st.warning("No data available with selected filters.")

# Title
st.image("assets/title.png")

# Home
if menu == "HOME":
    st.subheader("Le projet est accessible ici : https://github.com/Hebijaaj/Tardis")
    try:
        readme_content = open("README.md", "r", encoding="utf-8").read()
        st.markdown(readme_content)
    except FileNotFoundError:
        st.error("Le fichier README.md n'a pas été trouvé à la racine du projet.")

# Delay distribution visualizations
if menu == "Delay distribution":
    st.header("Delay distribution")

    # Distribution of train delays
    st.subheader("Distribution of train delays")

    filtered_delay = filtered_df[
        (filtered_df[delay_col] >= 0) & (filtered_df[delay_col] <= 60)
    ]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    sns.histplot(filtered_delay[delay_col], bins=30, kde=True, ax=ax)

    ax.set_title("Distribution of train delays")
    ax.set_xlabel("Delay (minutes)")
    ax.set_ylabel("Number of trips")

    st.pyplot(fig)

    # Top 30 gare_departs
    st.subheader("Top 30 departure stations")

    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(111)

    counts = filtered_df["gare_depart"].value_counts().head(30)

    ax.bar(counts.index, counts.values)
    ax.set_title("Top 30 departure stations")
    ax.set_xlabel("Station")
    ax.set_ylabel("Number of departures")

    plt.xticks(rotation=90)

    st.pyplot(fig)

    # Top 30 gare_arrivees
    st.subheader("Top 30 arrival stations")

    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(111)

    counts = filtered_df["gare_arrivee"].value_counts().head(30)

    ax.bar(counts.index, counts.values)
    ax.set_title("Top 30 arrival stations")
    ax.set_xlabel("Station")
    ax.set_ylabel("Number of arrivals")

    plt.xticks(rotation=90)

    st.pyplot(fig)

    # Top 5 gare_departs with highest delays
    st.subheader("Top 5 departure stations with highest delays")

    delays = (
        filtered_df.groupby("gare_depart")["retard_moyen_tous_trains_arrivee"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.pie(delays.values, labels=delays.index, autopct="%1.1f%%")

    st.pyplot(fig)

    # Top 5 gare_arrivees with highest delays
    st.subheader("Top 5 arrival stations with highest delays")

    delays = (
        filtered_df.groupby("gare_arrivee")["retard_moyen_tous_trains_arrivee"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.pie(delays.values, labels=delays.index, autopct="%1.1f%%")

    st.pyplot(fig)

    # Number of delays per year
    st.subheader("Number of delays per year")

    date_df = filtered_df.copy()

    date_df["date"] = pd.to_datetime(date_df["date"])
    date_df["Year"] = date_df["date"].dt.year
    delays_per_year = date_df["Year"].value_counts().sort_index()

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.bar(delays_per_year.index, delays_per_year.values)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of delays")
    ax.set_title("Number of delays per year")
    ax.set_xticks(delays_per_year.index)

    st.pyplot(fig)

# Display key metrics
if menu == "Summary statistics":
    st.header("Summary statistics")

    total_trips = len(filtered_df)
    punctuality_rate = (filtered_df[delay_col] <= 5).mean() * 100

    columns = st.columns(3)
    col1 = columns[0]
    col2 = columns[1]
    col3 = columns[2]

    col1.metric("Total trips", total_trips)
    col2.metric("Punctuality rate", f"{punctuality_rate:.1f}%")
    col3.metric("Average delay", f"{global_average_delay:.2f} min")
    if filtered_df.empty:
        st.warning("No data available with current filters.")
    else:
        st.subheader("Detailed filtered data")
        st.dataframe(filtered_df)
    # Data quality indicators
    missing_values = df.isnull().sum().sum()
    completeness = (1 - missing_values / (df.shape[0] * df.shape[1])) * 100

    st.metric("Data completeness", f"{completeness:.2f}%")
    st.metric("Missing values", int(missing_values))

# Prediction interface
if menu == "Delay prediction":
    st.header("Delay prediction")

    departure_station = st.selectbox(
        "Departure station", sorted(filtered_df["gare_depart"].dropna().unique())
    )
    arrival_station = st.selectbox(
        "Arrival station", sorted(filtered_df["gare_arrivee"].dropna().unique())
    )
    month = st.selectbox("Month of trip", months)
    if st.button("Predict delay"):
        liste = predicter.predict_data(
            departure_station, arrival_station, get_month_num_from_str(month)
        )
        if not isinstance(liste, list):
            st.error("Invalid data found. Did you put The same station twice ?")
        else:
            st.success(
                f"Predicted delay for trip from {departure_station} to {arrival_station} in {month}: {float(liste[0]):.2f} minutes"
            )

# Correlation heatmap
if menu == "Correlation heatmap":
    st.subheader("Heatmap")
    coords = pd.read_csv("train_stations_coordonates.csv")

    filtered_df.columns = filtered_df.columns.str.strip()
    coords.columns = coords.columns.str.strip()

    coords["gare_depart"] = coords["station"]

    merged_df = filtered_df.merge(coords, on="gare_depart", how="left")

    merged_df = merged_df.dropna(subset=["latitude", "longitude"])

    merged_df["ratio_annulation"] = (
        (merged_df["nb_annulation"] / merged_df["nb_train_prevu"])
        .replace([float("inf"), -float("inf")], 0)
        .fillna(0)
    )

    merged_df["ratio_annulation"] = np.log1p(merged_df["ratio_annulation"])

    merged_df = merged_df.groupby(
        ["gare_depart", "latitude", "longitude"], as_index=False
    )["ratio_annulation"].mean()

    heat_data = list(
        zip(
            merged_df["latitude"],
            merged_df["longitude"],
            merged_df["ratio_annulation"],
        )
    )

    m = Map(location=[46.6, 2.5], zoom_start=6)

    HeatMap(heat_data, radius=20, blur=15, min_opacity=0.3).add_to(m)

    m.save("heatmap.html")
    st.iframe("heatmap.html", height=600)

    st.subheader("Statistical correlation matrix")
    num_cols = [
        "retard_moyen_tous_trains_depart",
        "retard_moyen_tous_trains_arrivee",
        "nb_annulation",
        "nb_train_retard_sup_15",
        "retard_moyen_trains_retard_sup15",
        "nb_train_retard_sup_30",
        "nb_train_retard_sup_60",
        "prct_cause_externe",
        "prct_cause_infra",
        "prct_cause_gestion_trafic",
        "prct_cause_materiel_roulant",
    ]

    corr = filtered_df[num_cols].corr()
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation between delay variables")
    plt.tight_layout()

    st.pyplot(fig)

if menu == "Model explanation":
    st.header("Model explanation")

    if isinstance(model, Pipeline):
        final_model = model.named_steps["model"]
        preprocessor = model.named_steps["preprocessor"]
    else:
        final_model = model_pipeline.named_steps["model"]
        preprocessor = model_pipeline.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()

    importances = np.abs(final_model.coef_)
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})

    importance_df = importance_df.sort_values(by="Importance", ascending=False).head(20)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.barh(importance_df["Feature"], importance_df["Importance"])
    ax.set_title("Top 20 most important features")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    plt.gca().invert_yaxis()
    st.pyplot(fig)
    st.subheader("Feature importance table")
    st.dataframe(importance_df)

st.sidebar.header("Export")

# Download filtered data
csv = filtered_df.to_csv(index=False).encode("utf-8")

st.sidebar.download_button(
    label="Download filtered data",
    data=csv,
    file_name="filtered_trains_data.csv",
    mime="text/csv",
)
