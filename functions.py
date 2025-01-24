import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from openai import OpenAI
import os
import folium
import json


data_files = {
    "Brazil": "data/Brazil.xlsx",
    "Congo": "data/Congo.xlsx",
    "Cameroon": "data/Cameroon.xlsx",
    "Germany": "data/Germany.xlsx",
    "Canada": "data/Canada.xlsx",
    "France": "data/France.xlsx",
    "Indonesia": "data/Indonesia.xlsx",
    "Australia": "data/Australia.xlsx",
    "USA": "data/USA.xlsx",
}
geojson_files = {
    "Brazil": "data/geojson/brazil.geojson",
    "Congo": "data/geojson/democratic republic of the congo.geojson",
    "Cameroon": "data/geojson/cameroon.geojson",
    "Indonesia": "data/geojson/indonesia.geojson",
    "Australia": "data/geojson/australia.geojson",
    "Germany": "data/geojson/germany.geojson",
    "France": "data/geojson/france.geojson",
    "Canada": "data/geojson/canada.geojson",
    "USA": "data/geojson/united states of america.geojson",
}
species_file = "data/filtered_species_data.csv"

map_centers = {
    "Australia": [-25.2744, 133.7751],
    "Brazil": [-14.235, -51.9253],
    "Cameroon": [7.369722, 12.354722],
    "Congo": [-0.228, 15.8277],
    "Indonesia": [-2.4834, 117.8903],
    "USA": [40.7821, -99.5501],
    "Germany": [51.1657, 10.4515],
    "Canada": [56.1304, -106.3468],
    "France": [46.6034, 1.8883],
    "All Countries": [0, 0],
}

global_file_path = "data/global.xlsx"
global_extinction_file_path = "data/filtered_species_data_all_countries.csv"

os.environ["OPENAI_API_KEY"] = "USE_YOUR_API_KEY"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@st.cache_data
def load_deforestation_data(country):
    """Loads country-specific deforestation data from Excel."""
    return pd.read_excel(data_files[country], sheet_name="Country tree cover loss")

@st.cache_data
def load_carbon_data(country):
    """Loads country-specific carbon emissions data from Excel."""
    return pd.read_excel(data_files[country], sheet_name="Country carbon data")

@st.cache_data
def load_geojson(country):
    """Loads GeoJSON data for a specific country."""
    with open(geojson_files[country], 'r') as f:
        return json.load(f)

@st.cache_data
def load_species_data():
    """Loads species data from a CSV file."""
    return pd.read_csv(species_file)

@st.cache_data
def initialize_map(center, zoom):
    """Initializes a Folium map centered at a given location."""
    return folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri"
    )

def calculate_forest_percentage(data, year):
    """
    Calculates remaining forest percentage based on cumulative forest loss up to a specific year.
    """
    loss_columns = [f"tc_loss_ha_{y}" for y in range(2001, year + 1)]
    cumulative_loss = data[loss_columns].sum(axis=1).values[0]
    initial_extent = data["extent_2000_ha"].values[0]
    remaining_forest = initial_extent - cumulative_loss
    return remaining_forest, initial_extent, cumulative_loss

def calculate_static_forest_loss(data):
    """
    Calculates the number of extinct and endangered species up to a given year.
    """
    loss_columns = [f"tc_loss_ha_{y}" for y in range(2001, 2021)]
    cumulative_loss = data[loss_columns].sum(axis=1).values[0]
    initial_extent = data["extent_2000_ha"].values[0]
    remaining_forest = initial_extent - cumulative_loss
    return remaining_forest, initial_extent, cumulative_loss

def calculate_species_stats(species_data, selected_year, country=None):
    """
    Calculates cumulative carbon emissions up to a specific year.
    """
    species_latest = (
        species_data[species_data["Year"] <= selected_year]
        .groupby("Binomial", as_index=False)
        .last()
    )

    if country:
        species_latest = species_latest[species_latest["Country"] == country]
        print(species_latest)

    extinct_count = species_latest[species_latest["Population"] == 0].shape[0]
    endangered_count = species_latest[
        (species_latest["Population"] > 0) & (species_latest["Population"] < 50)
    ].shape[0]  # Example threshold for endangered

    return extinct_count, endangered_count

def calculate_carbon_emissions(data, selected_year):
    """
    Returns a color code based on forest percentage.
    Green: > 95%, Yellow: 90-95%, Red: < 90%
    """
    initial_carbon_stock = data["gfw_aboveground_carbon_stocks_2000__Mg_C"].values[0]

    cumulative_emissions = 0
    for year in range(2001, selected_year + 1):
        column_name = f"gfw_forest_carbon_gross_emissions_{year}__Mg_CO2e"
        cumulative_emissions += data[column_name].fillna(0).values[0]

    return initial_carbon_stock, cumulative_emissions

def calculate_static_carbon_emissions(data):
    """
    Updates map layers with forest percentage and species markers for a given country and year.
    """
    gain_columns = [f"gfw_forest_carbon_gross_emissions_{y}__Mg_CO2e" for y in range(2001, 2024)]
    cumulative_gain = data[gain_columns].sum(axis=1).values[0]
    initial_extent = data["gfw_aboveground_carbon_stocks_2000__Mg_C"].values[0]
    total_emissions = initial_extent + cumulative_gain
    return total_emissions

def get_polygon_color(forest_percentage):
    """
    Updates countries color based on forest percentage
    """
    if forest_percentage >= 0.95:
        return "#00FF00"  # Green
    elif 0.90 <= forest_percentage < 0.95:
        return "#FFFF00"  # Yellow
    else:
        return "#FF0000"  # Red

def update_map_layers(map_object, selected_country, selected_year):
    """
    Updates map layers with forest percentage and species markers for a given country
    """
    for country, file in data_files.items():
        if selected_country == "All Countries" or selected_country == country:
            data = load_deforestation_data(country)
            geojson = load_geojson(country)
            data_75_threshold = data[data["threshold"] == 75]
            remaining_forest, initial_extent, cumulative_loss = calculate_forest_percentage(data_75_threshold, selected_year)
            forest_percentage = remaining_forest / initial_extent
            folium.GeoJson(
                geojson,
                style_function=lambda feature: {
                    'fillColor': get_polygon_color(forest_percentage),
                    'color': get_polygon_color(forest_percentage),
                    'weight': 0.5,
                    'fillOpacity': 0.5,
                },
                tooltip=f"{country} - Forest: {forest_percentage:.2%}"
            ).add_to(map_object)

    species_data = load_species_data()
    species_data_sorted = species_data.sort_values(by=["Binomial", "Year"])
    species_latest = (
        species_data_sorted[species_data_sorted["Year"] <= selected_year]
        .groupby("Binomial", as_index=False)
        .last()
    )

    if selected_country != "All Countries":
        species_latest = species_latest[species_latest["Country"] == selected_country]

    for _, row in species_latest.iterrows():
        color = "red" if row["Population"] == 0 else "blue"
        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            popup=folium.Popup(
                f"<b>{row['Common_name']}</b> ({row['Binomial']})<br>"
                f"Population: {row['Population']}<br>"
                f"Status: {'Extinct' if row['Population'] == 0 else 'Endangered' if row['Population'] < 1000 else 'Stable'}",
                max_width=300,
            ),
            icon=folium.Icon(icon="info-sign", color=color),
        ).add_to(map_object)

    return map_object

def load_global_data_with_threshold(file_path, deforestation_sheet, carbon_sheet):
    """
    Loads global data from excel sheet with deforestation data and carbon
    """
    global_deforestation_data = pd.read_excel(file_path, sheet_name=deforestation_sheet)
    global_carbon_data = pd.read_excel(file_path, sheet_name=carbon_sheet)

    global_deforestation_data = global_deforestation_data[global_deforestation_data["threshold"] == 75]
    global_carbon_data = global_carbon_data[global_carbon_data["umd_tree_cover_density_2000__threshold"] == 75]

    return global_deforestation_data, global_carbon_data

def calculate_species_trend(species_data, selected_country=None):
    """
    Calculate annual extinct and endangered species trends globally or for a specific country.
    """
    if selected_country:
        species_data = species_data[species_data["Country"] == selected_country]

    trend_data = []
    for year in range(2001, 2021):
        yearly_data = species_data[species_data["Year"] <= year]
        latest_species_data = yearly_data.groupby("Binomial", as_index=False).last()

        extinct_count = latest_species_data[latest_species_data["Population"] == 0].shape[0]
        endangered_count = latest_species_data[
            (latest_species_data["Population"] > 0) & (latest_species_data["Population"] < 1000)
        ].shape[0]

        trend_data.append({"Year": year, "Extinct Species": extinct_count, "Endangered Species": endangered_count})

    return pd.DataFrame(trend_data)

def plot_trend_chart_filtered(selected_country, deforestation_data, carbon_data, species_trend_data):
    """
    Plots the trend chart filtered by selected country.
    """
    years = range(2001, 2021)

    missing_deforestation_cols = [
        col for col in [f"tc_loss_ha_{year}" for year in range(2001, 2021)]
        if col not in global_deforestation_data.columns
    ]
    missing_carbon_cols = [
        col for col in [f"gfw_forest_carbon_gross_emissions_{year}__Mg_CO2e" for year in range(2001, 2021)]
        if col not in global_carbon_data.columns
    ]

    print("Missing Deforestation Columns:", missing_deforestation_cols)
    print("Missing Carbon Columns:", missing_carbon_cols)

    deforestation_data = global_deforestation_data.fillna(0)
    carbon_data = global_carbon_data.fillna(0)

    forest_loss = [
        deforestation_data.loc[
            (deforestation_data["country"] == selected_country), f"tc_loss_ha_{year}"
        ].values[0] if f"tc_loss_ha_{year}" in deforestation_data.columns else 0
        for year in range(2001, 2021)
    ]

    carbon_emissions = [
        carbon_data.loc[
            (carbon_data["country"] == selected_country), f"gfw_forest_carbon_gross_emissions_{year}__Mg_CO2e"
        ].values[0] if f"gfw_forest_carbon_gross_emissions_{year}__Mg_CO2e" in carbon_data.columns else 0
        for year in range(2001, 2021)
    ]


    # Extract annual extinction trends from pre-calculated data
    annual_extinction_trends = species_trend_data.set_index("Year")["Extinct Species"].reindex(years, fill_value=0)

    # Scale metrics
    forest_loss_mha = [loss / 1e6 for loss in forest_loss]  # Scale to Mha
    carbon_emissions_gt = [emission / 1e9 for emission in carbon_emissions]  # Scale to Gt

    # Cumulative sums for trends
    cumulative_forest_loss_mha = np.cumsum(forest_loss_mha)
    cumulative_carbon_emissions_gt = np.cumsum(carbon_emissions_gt)

    # Plot with Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(years),
        y=cumulative_forest_loss_mha,
        mode='lines+markers',
        name='Cumulative Forest Loss (Mha)',
        line=dict(width=2),
    ))
    fig.add_trace(go.Scatter(
        x=list(years),
        y=cumulative_carbon_emissions_gt,
        mode='lines+markers',
        name='Cumulative Carbon Emissions (Gt CO2e)',
        line=dict(width=2),
    ))

    fig.update_layout(
        title=f"Trend of Forest Loss and Carbon Emissions ({selected_country}: 2001-2020)",
        xaxis_title="Year",
        yaxis_title="Cumulative Values",
        legend=dict(
            orientation="h",  # Horizontal orientation
            yanchor="bottom",
            y=1.02,  # Position above the plot
            xanchor="center",
            x=0.5
        ),
        template="plotly_white",
        margin=dict(b=10),  # Reduce bottom margin of the first plot
    )

    # Plot Biodiversity Loss
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=list(years),
        y=annual_extinction_trends.values,
        mode='lines+markers',
        name='Extinct Species (Annual)',
        line=dict(width=2, dash='dot', color='green'),
    ))
    fig2.update_layout(
        title=f"Biodiversity Loss ({selected_country}: 2001-2020)",
        xaxis_title="Year",
        yaxis_title="Number of Extinct Species",
        legend=dict(
            orientation="h",  # Horizontal orientation
            yanchor="bottom",
            y=1.02,  # Position above the plot
            xanchor="center",
            x=0.5
        ),
        template="plotly_white",
        showlegend=True,
        margin=dict(t=10),  # Reduce bottom margin of the first plot
    )

    return fig, fig2

def calculate_global_insights(deforestation_data, carbon_data, extinction_data):
    """
    Calculate global insights based on deforestation data and carbon data
    """

    total_forest_loss_mha = deforestation_data[
        [col for col in deforestation_data.columns if col.startswith("tc_loss_ha_")]
    ].sum().sum() / 1e6  # Convert to Mha

    total_carbon_emissions_gt = carbon_data[
        [col for col in carbon_data.columns if col.startswith("gfw_forest_carbon_gross_emissions_")]
    ].sum().sum() / 1e9  # Convert to Gt CO2e

    extinction_data_distinct_species = extinction_data.groupby("Binomial", as_index=False).last()
    total_extinct_species = extinction_data_distinct_species[
        extinction_data_distinct_species["Population"] == 0
    ].shape[0]

    insights = (
        f"From 2001 to 2020, the planet has lost approximately "
        f"**{total_forest_loss_mha:,.2f}** million hectares of forest, resulting in "
        f"**{total_carbon_emissions_gt:,.2f}** gigatonnes of carbon emissions. "
        f"During this period, an alarming **{total_extinct_species}** species have gone extinct. "
        f"These figures highlight the urgent need for global action to protect biodiversity and combat climate change."
    )
    return insights

def generate_insights(country, forest_loss, carbon_emissions, biodiversity_loss):
    """
    Generates actionable insights for lewmakers for a given country.
    """
    prompt = f"""
    Based on the environmental trends in {country}:
    - Forest loss: {forest_loss} Mha
    - Carbon emissions: {carbon_emissions} Gt CO2e
    - Biodiversity loss: {biodiversity_loss} species

    Suggest actionable insights that lawmakers in {country} can implement to mitigate these issues.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates actionable insights."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.7
    )
    return response.choices[0].message.content

def get_or_generate_insight(country, forest_loss, carbon_emissions, biodiversity_loss):
    """
    Loads or generates an insight for a country if it doesn't exist.
    """

    existing_insight = insights_cache.loc[insights_cache["country"] == country, "insight"]
    if not existing_insight.empty:
        return existing_insight.values[0]

    new_insight = generate_insights(country, forest_loss, carbon_emissions, biodiversity_loss)

    # Save the new insight to the cache
    new_row = pd.DataFrame({"country": [country], "insight": [new_insight]})
    updated_cache = pd.concat([insights_cache, new_row], ignore_index=True)
    updated_cache.to_csv(INSIGHTS_CACHE_FILE, index=False)

    return new_insight

def get_or_generate_forest_loss_insight(country, predictions):
    """
    Retrieve or generate an insight paragraph for forest loss predictions.
    """
    # Load existing insights cache
    if os.path.exists(PREDICTION_INSIGHT_CACHE_FILE):
        with open(PREDICTION_INSIGHT_CACHE_FILE, "r") as f:
            insights_cache = json.load(f)
    else:
        insights_cache = {}

    # Check if insight already exists for the country
    if country in insights_cache:
        return insights_cache[country]

    # Generate new insight
    total_loss = predictions["predicted_value"].sum() / 1e6  # Convert to Mha
    peak_loss = predictions["predicted_value"].max() / 1e6
    peak_year = predictions.loc[predictions["predicted_value"].idxmax(), "year"]

    prompt = f"""
    In {country}, the predicted forest loss between 2026 and 2046 is staggering. 
    A total of approximately {total_loss:.2f} million hectares of forest is expected to vanish during this period. 
    The year {peak_year} is projected to witness the highest annual loss, with an estimated {peak_loss:.2f} million hectares disappearing.

    Write a deep, impactful paragraph addressing lawmakers and stakeholders, urging them to take immediate action.
    Use an eloquent tone that highlights the gravity of the situation and the consequences of inaction.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an eloquent and persuasive environmental advocate."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=500,
        temperature=0.8
    )

    insight = response.choices[0].message.content

    # Cache the generated insight
    insights_cache[country] = insight
    with open(PREDICTION_INSIGHT_CACHE_FILE, "w") as f:
        json.dump(insights_cache, f)

    return insight

def reshape_forest_loss_data(data):
    """
    Reshape the forest loss data into a numpy array.
    """
    year_columns = [col for col in data.columns if col.startswith("tc_loss_ha_")]
    reshaped_data = data.melt(
        id_vars=["country"],
        value_vars=year_columns,
        var_name="Year",
        value_name="Forest Loss (ha)"
    )

    reshaped_data["Year"] = reshaped_data["Year"].str.extract("(\d+)").astype(int)

    return reshaped_data

def reshape_carbon_data(data):
    """
    Reshape the carbon data into a numpy array.
    """
    year_columns = [col for col in data.columns if col.startswith("gfw_forest_carbon_gross_emissions_")]
    reshaped_data = data.melt(
        id_vars=["country"],
        value_vars=year_columns,
        var_name="Year",
        value_name="Carbon Emissions (Mg CO₂)"
    )

    # Extract the year
    reshaped_data["Year"] = reshaped_data["Year"].str.extract("(\d+)").astype(int)

    return reshaped_data

global_deforestation_data, global_carbon_data = load_global_data_with_threshold(
    global_file_path, deforestation_sheet="Country tree cover loss", carbon_sheet="Country carbon data"
)

global_extinction_data = pd.read_csv(global_extinction_file_path)

INSIGHTS_CACHE_FILE = "data/insights_cache.csv"
PREDICTION_INSIGHT_CACHE_FILE = "data/predicted_insights_cache.json"

if not os.path.exists(INSIGHTS_CACHE_FILE):
    pd.DataFrame(columns=["country", "insight"]).to_csv(INSIGHTS_CACHE_FILE, index=False)

insights_cache = pd.read_csv(INSIGHTS_CACHE_FILE)

predicted_forest_loss = pd.read_csv("data/predicted_trends_2026_2046.csv")

forest_loss_predictions = predicted_forest_loss[predicted_forest_loss["metric"] == "forest_loss"]