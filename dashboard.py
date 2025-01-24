import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
import plotly.graph_objects as go
import functions


st.set_page_config(
    page_title="EcoGuard Insights Dashboard",
    page_icon="🌍",
    layout="wide"
)

col1, col2 = st.columns([3, 7])
col3, col4 = st.columns([3, 7])
col5, col6 = st.columns([5, 5])

st.markdown(
    """
    <style>
        /* Remove padding around the main content */
        .stMainBlockContainer {
            padding: 0rem 2rem 1rem 2rem;
        }

        /* Adjust padding of sidebar */
        .css-1d391kg {
            padding: 0.5rem 1rem 0.5rem 1rem;
        }

        /* General body adjustments */
        .css-18e3th9 {
            padding: 0.5rem;
        }
        
        .st-emotion-cache-0.eiemyj5 p {
            margin-bottom: 0px;
            font-size: 15px;
            color: #ccc;
        }
        
        h3#top-5-countries {
            font-size: 21px;
        }
        
        h4 {
            font-size: 15px !important;
            padding-bottom: 5px !important;
            margin-top: 20px !important;
        }
        
        .stElementContainer.element-container.st-emotion-cache-77fcpe.eiemyj1 {
        }
        
        .stColumn.st-emotion-cache-1b50p5p.eiemyj2 {
            padding: 20px;
            background: #202736;
        }
        
        #root > div:nth-child(1) > div.withScreencast > div > div > section > div.stMainBlockContainer.block-container.st-emotion-cache-1ibsh2c.ekr3hml4 > div > div > div > div:nth-child(3) > div.stColumn.st-emotion-cache-1b50p5p.eiemyj2 > div > div > div > div:nth-child(22) > div .st-emotion-cache-1cvow4s.e121c1cl0 strong {
            color: #ef553b;
        }
        
        #root > div:nth-child(1) > div.withScreencast > div > div > section > div.stMainBlockContainer.block-container.st-emotion-cache-1ibsh2c.ekr3hml4 > div > div > div > div:nth-child(3) > div.stColumn.st-emotion-cache-1b50p5p.eiemyj2 > div > div > div > div:nth-child(2) h3 {
            padding-bottom: 0px;
            margin-bottom: -25px;
        }
        
        #root > div:nth-child(1) > div.withScreencast > div > div > section > div.stMainBlockContainer.block-container.st-emotion-cache-1ibsh2c.ekr3hml4 > div > div > div > div:nth-child(1) > div.stColumn.st-emotion-cache-1b50p5p.eiemyj2 > div > div > div > div:nth-child(3) > div > div p {
            margin-bottom: 20px;
        }
        #root > div:nth-child(1) > div.withScreencast > div > div > section > div.stMainBlockContainer.block-container.st-emotion-cache-1ibsh2c.ekr3hml4 > div > div > div > div:nth-child(3) > div:nth-child(1){
            padding: 20px 20px 40px !important;
            background: #202736;
            border: 1px #798aad solid;
        }
        
        #root > div:nth-child(1) > div.withScreencast > div > div > section > div.stMainBlockContainer.block-container.st-emotion-cache-1ibsh2c.ekr3hml4 > div > div > div > div:nth-child(3) {
            margin-top: 30px;
        }
        
        #root > div:nth-child(1) > div.withScreencast > div > div > section > div.stMainBlockContainer.block-container.st-emotion-cache-1ibsh2c.ekr3hml4 > div > div > div > div:nth-child(1) > div.stColumn.st-emotion-cache-1b50p5p.eiemyj2 > div > div > div > div:nth-child(3) > div > div {
            padding-left: 6px;
            margin-top: 20px;
        }
        
        .st-emotion-cache-1dj3ksd {
            background-color: #6ee66e;
        }
        
        .st-emotion-cache-1373cj4 {
            color: #6ee66e;
        }
        #root > div:nth-child(1) > div.withScreencast > div > div > section > div.stMainBlockContainer.block-container.st-emotion-cache-1ibsh2c.ekr3hml4 > div > div > div > div:nth-child(1) > div.stColumn.st-emotion-cache-1b50p5p.eiemyj2 > div > div > div > div:nth-child(1) {
            margin-top: 50px;
        }
        
        #root > div:nth-child(1) > div.withScreencast > div > div > section > div.stMainBlockContainer.block-container.st-emotion-cache-1ibsh2c.ekr3hml4 > div > div > div > div:nth-child(1) > div.stColumn.st-emotion-cache-1b50p5p.eiemyj2 > div > div > div > div:nth-child(3) > div > label > div p, #root > div:nth-child(1) > div.withScreencast > div > div > section > div.stMainBlockContainer.block-container.st-emotion-cache-1ibsh2c.ekr3hml4 > div > div > div > div:nth-child(1) > div.stColumn.st-emotion-cache-1b50p5p.eiemyj2 > div > div > div > div:nth-child(2) > div > label > div p {
            font-weight: bold;
            font-size: 17px !important;
        }
        
        #root > div:nth-child(1) > div.withScreencast > div > div > section > div.stMainBlockContainer.block-container.st-emotion-cache-1ibsh2c.ekr3hml4 > div > div > div > div:nth-child(1) > div.stColumn.st-emotion-cache-1yrnl19.eiemyj2 > div > div {
            margin-top: 60px;
            margin-bottom: -25px !important;
        }
        
        h3#summary-for-brazil {
            font-size: 20px;
        }
        
        #root > div:nth-child(1) > div.withScreencast > div > div > section > div.stMainBlockContainer.block-container.st-emotion-cache-1ibsh2c.ekr3hml4 > div > div > div > div:nth-child(3) > div:nth-child(1) > div > div > div > div:nth-child(2) > div > div strong {
            color: #6ee66e;
        }
        
        .st-emotion-cache-1cvow4s h3 {
            font-size: 1.5rem;
            padding: 0.5rem 0px 1rem;
            margin-bottom: 15px;
        }
        #root > div:nth-child(1) > div.withScreencast > div > div > section > div.stMainBlockContainer.block-container.st-emotion-cache-1ibsh2c.ekr3hml4 > div > div > div > div:nth-child(2) > div.stColumn.st-emotion-cache-1b50p5p.eiemyj2 > div > div > div > div:nth-child(21) > div > div > p > strong,
        #root > div:nth-child(1) > div.withScreencast > div > div > section > div.stMainBlockContainer.block-container.st-emotion-cache-1ibsh2c.ekr3hml4 > div > div > div > div:nth-child(1) > div.stColumn.st-emotion-cache-1b50p5p.eiemyj2 > div > div > div > div:nth-child(3) > div > div > p strong {
            color: #ff4b4b;
        }
        
        .st-bb {
            background-color: #000 !important;
        }
        
        .st-emotion-cache-1cvow4s h3 {
            font-size: 1.5rem;
            padding: 0.5rem 0px 1rem;
            margin-bottom: 15px;
        }

    </style>
    """,
    unsafe_allow_html=True,
)

with col1:
    st.subheader("🌍 EcoGuard Insights Dashboard")

    selected_country = st.selectbox("Select Country", ["All Countries"] + list(functions.data_files.keys()))
    selected_year = st.slider("Select a Year", 2001, 2023, 2001)

    if selected_country != "All Countries":

        deforestation_data = functions.load_deforestation_data(selected_country)
        carbon_data = functions.load_carbon_data(selected_country)

        data_75_threshold = deforestation_data[deforestation_data["threshold"] == 75]
        remaining_forest, initial_extent, cumulative_loss = functions.calculate_forest_percentage(data_75_threshold,
                                                                                        selected_year)

        remaining_forest_static, initial_extent_static, total_loss_static = functions.calculate_static_forest_loss(
            data_75_threshold)
        forest_loss_percentage_static = (total_loss_static / initial_extent_static) * 100

        filtered_data = carbon_data[carbon_data["umd_tree_cover_density_2000__threshold"] == 75]
        total_emissions_static = functions.calculate_static_carbon_emissions(filtered_data)

        species_data = functions.load_species_data()
        total_extinct_count, total_endangered_count = functions.calculate_species_stats(
            species_data, 2021, selected_country if selected_country != "All Countries" else None
        )

        st.markdown(f"""
            ### Summary for {selected_country}
            From 2001 to 2023, **{selected_country}** lost approximately **{total_loss_static / 1e6:.2f} million hectares** of tree cover, 
            which is equivalent to a decrease of **{forest_loss_percentage_static:.2f}%** since 2000. 
            Additionally, this resulted in **{total_emissions_static / 1e9:.2f} Gt** of carbon emissions.
            During this period, **{total_extinct_count} species** went extinct, and **{total_endangered_count} species** became endangered.
            """)

        initial_carbon_stock, cumulative_emissions = functions.calculate_carbon_emissions(filtered_data, selected_year)

        extinct_count, endangered_count = functions.calculate_species_stats(
            species_data, selected_year, selected_country if selected_country != "All Countries" else None
        )

        st.markdown(f"""
        **Forest Cover Loss Statistics for {selected_country} ({selected_year}):**
        - **Remaining Forest Area:** {remaining_forest:,.2f} hectares
        - **Initial Forest Area (2000):** {initial_extent:,.2f} hectares
        - **Cumulative Loss:** {cumulative_loss:,.2f} hectares
        - **Initial Carbon Stock (2000):** {initial_carbon_stock:,.2f} Mg C
        - **Cumulative Carbon Emissions:** {cumulative_emissions:,.2f} Mg CO₂
        """)

        st.markdown(f"""
        **Biodiversity Statistics for {selected_country} ({selected_year}):**
        - **Number of Endangered Species:** {endangered_count}
        - **Number of Extinct Species:** {extinct_count}
        """)

    else:
        st.markdown(
            """
            This dashboard provides an in-depth analysis of global deforestation, carbon emissions, and biodiversity loss.
            Use the controls above to explore data trends for specific countries and years.

            ⚠️ **Action Needed:** Understand the alarming trends and take action to mitigate the environmental impact. 

            🌲 Select a country to dive deeper into its environmental trends.
            """
        )

map_center = functions.map_centers[selected_country]
zoom_start = 3 if selected_country == "All Countries" else 5
m = functions.initialize_map(map_center, zoom_start)

m = functions.update_map_layers(m, selected_country, selected_year)

with col2:
    st_folium(m, width=1150, height=790, key="map")

with col3:
    st.subheader("Top 5 Countries Statistics")

    functions.global_deforestation_data["cumulative_forest_loss"] = functions.global_deforestation_data[
        [f"tc_loss_ha_{year}" for year in range(2001, 2021)]
    ].sum(axis=1)
    top_forest_loss = functions.global_deforestation_data.nlargest(5, "cumulative_forest_loss")[["country", "cumulative_forest_loss"]]

    functions.global_carbon_data["cumulative_carbon_emissions"] = functions.global_carbon_data[
        [f"gfw_forest_carbon_gross_emissions_{year}__Mg_CO2e" for year in range(2001, 2021)]
    ].sum(axis=1)
    top_carbon_emissions = functions.global_carbon_data.nlargest(5, "cumulative_carbon_emissions")[["country", "cumulative_carbon_emissions"]]

    biodiversity_loss = (
        functions.global_extinction_data[functions.global_extinction_data["Population"] == 0]
        .groupby("Country", as_index=False)["Binomial"]
        .nunique()
        .rename(columns={"Binomial": "extinct_species"})
    )
    top_biodiversity_loss = biodiversity_loss.nlargest(5, "extinct_species")

    st.markdown("#### **🌲 Forest Loss (Mha)**")
    for _, row in top_forest_loss.iterrows():
        st.write(f"**{row['country']}**: {row['cumulative_forest_loss'] / 1e6:.2f} Mha")

    st.markdown("#### **💨 Carbon Emissions (Gt CO₂)**")
    for _, row in top_carbon_emissions.iterrows():
        st.write(f"**{row['country']}**: {row['cumulative_carbon_emissions'] / 1e9:.2f} Gt CO₂")

    st.markdown("#### **🦋 Biodiversity Loss (Species)**")
    for _, row in top_biodiversity_loss.iterrows():
        st.write(f"**{row['Country']}**: {row['extinct_species']} species")

    global_insights = functions.calculate_global_insights(functions.global_deforestation_data, functions.global_carbon_data, functions.global_extinction_data)
    st.markdown("#### 🌍 Global Insights")
    st.markdown(global_insights)

with col4:
    if selected_country == "All Countries":
        species_trend_data = functions.calculate_species_trend(functions.global_extinction_data, None)

        reshaped_forest_loss = functions.reshape_forest_loss_data(functions.global_deforestation_data)
        reshaped_carbon_data = functions.reshape_carbon_data(functions.global_carbon_data)

        global_deforestation_trend = reshaped_forest_loss.groupby("Year")["Forest Loss (ha)"].sum() / 1e6  # Mha
        global_carbon_trend = reshaped_carbon_data.groupby("Year")["Carbon Emissions (Mg CO₂)"].sum() / 1e9  # Gt CO₂

        common_years = sorted(set(global_deforestation_trend.index) & set(global_carbon_trend.index))

        global_deforestation_trend = global_deforestation_trend.reindex(common_years, fill_value=0)
        global_carbon_trend = global_carbon_trend.reindex(common_years, fill_value=0)

        global_trend_df = pd.DataFrame({
            "Year": common_years,
            "Forest Loss (Mha)": global_deforestation_trend.values,
            "Carbon Emissions (Gt CO₂)": global_carbon_trend.values,
        })

        trend_chart_filtered1 = go.Figure()
        trend_chart_filtered1.add_trace(go.Scatter(
            x=global_trend_df["Year"],
            y=global_trend_df["Forest Loss (Mha)"],
            mode="lines+markers",
            name="Global Forest Loss (Mha)"
        ))
        trend_chart_filtered1.add_trace(go.Scatter(
            x=global_trend_df["Year"],
            y=global_trend_df["Carbon Emissions (Gt CO₂)"],
            mode="lines+markers",
            name="Global Carbon Emissions (Gt CO₂)"
        ))
        trend_chart_filtered1.update_layout(
            title="Global Trends: Forest Loss and Carbon Emissions (2001-2020)",
            xaxis_title="Year",
            yaxis_title="Cumulative Values",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            template="plotly_white"
        )

        trend_chart_filtered2 = go.Figure()
        trend_chart_filtered2.add_trace(go.Scatter(
            x=species_trend_data["Year"],
            y=species_trend_data["Extinct Species"],
            mode="lines+markers",
            name="Extinct Species (Global)"
        ))
        trend_chart_filtered2 = go.Figure()
        trend_chart_filtered2.add_trace(go.Scatter(
            x=species_trend_data["Year"],
            y=species_trend_data["Extinct Species"].fillna(0),
            mode="lines+markers",
            name="Extinct Species (Global)",
            line=dict(color="green"),
            marker=dict(color="green")
        ))


    else:
        species_trend_data = functions.calculate_species_trend(functions.global_extinction_data, selected_country)

        congo_species_data = functions.global_extinction_data[functions.global_extinction_data["Country"] == "Congo"]

        trend_chart_filtered1, trend_chart_filtered2 = functions.plot_trend_chart_filtered(
            selected_country,
            functions.global_deforestation_data,
            functions.global_carbon_data,
            species_trend_data
        )

    st.plotly_chart(trend_chart_filtered1, use_container_width=True)
    st.plotly_chart(trend_chart_filtered2, use_container_width=True)

with col5:
    if selected_country.lower() == "all countries":

        final_forest_loss = (
            functions.global_deforestation_data.loc[
                :,
                [f"tc_loss_ha_{year}" for year in range(2001, 2021)]
            ].sum().sum() / 1e6  # Convert to Mha
        )

        final_carbon_emissions = (
            functions.global_carbon_data.loc[
                :,
                [f"gfw_forest_carbon_gross_emissions_{year}__Mg_CO2e" for year in range(2001, 2021)]
            ].sum().sum() / 1e9  # Convert to Gt CO2e
        )

        species_trend = functions.calculate_species_trend(functions.global_extinction_data, "all_countries")
        final_biodiversity_loss = species_trend["Extinct Species"].iloc[-1]
    else:
        final_forest_loss = (
            functions.global_deforestation_data.loc[
                functions.global_deforestation_data["country"] == selected_country,
                [f"tc_loss_ha_{year}" for year in range(2001, 2021)]
            ].sum(axis=1).values[0] / 1e6  # Convert to Mha
        )

        final_carbon_emissions = (
            functions.global_carbon_data.loc[
                functions.global_carbon_data["country"] == selected_country,
                [f"gfw_forest_carbon_gross_emissions_{year}__Mg_CO2e" for year in range(2001, 2021)]
            ].sum(axis=1).values[0] / 1e9  # Convert to Gt CO2e
        )

        species_trend = functions.calculate_species_trend(functions.global_extinction_data, selected_country)
        final_biodiversity_loss = species_trend["Extinct Species"].iloc[-1]

    insight = functions.get_or_generate_insight(
        "Global" if selected_country.lower() == "all countries" else selected_country,
        round(final_forest_loss, 2),
        round(final_carbon_emissions, 2),
        int(final_biodiversity_loss)
    )

    st.subheader("Actionable Insights for Lawmakers")
    st.markdown(insight)

with col6:
    if selected_country.lower() == "all countries":

        global_predictions = functions.forest_loss_predictions.groupby("year")["predicted_value"].sum().reset_index()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=global_predictions["year"],
            y=global_predictions["predicted_value"] / 1e6,  # Convert to Mha
            mode='lines+markers',
            name="Global Predicted Forest Loss",
            line=dict(dash="dot"),
        ))

        fig.update_layout(
            title="Global Predicted Forest Loss (2026-2046)",
            xaxis_title="Year",
            yaxis_title="Predicted Forest Loss (Mha)",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )

        base_insight = functions.get_or_generate_forest_loss_insight("Global", global_predictions)
    else:
        selected_country_predictions = functions.forest_loss_predictions[functions.forest_loss_predictions["country"] == selected_country]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=selected_country_predictions["year"],
            y=selected_country_predictions["predicted_value"] / 1e6,  # Convert to Mha
            mode='lines+markers',
            name=f"Predicted Forest Loss ({selected_country})",
            line=dict(dash="dot"),
        ))

        fig.update_layout(
            title=f"Predicted Forest Loss (2026-2046) - {selected_country}",
            xaxis_title="Year",
            yaxis_title="Predicted Forest Loss (Mha)",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )

        base_insight = functions.get_or_generate_forest_loss_insight(selected_country, selected_country_predictions)

    st.plotly_chart(fig, use_container_width=True)

    extended_insight = (
        f"{base_insight} Furthermore, the loss of forest ecosystems disrupts animal migration patterns, which play a "
        f"critical role in maintaining ecological balance. Migratory species rely on forests for shelter and food during "
        f"their journeys, and their displacement can lead to cascading effects on agricultural pollination, pest control, "
        f"and soil fertility. This disruption not only threatens biodiversity but also jeopardizes global food security, "
        f"emphasizing the need for immediate, coordinated action to mitigate these losses."
    )

    st.markdown(
        f"""
                <style>
                .warning-box {{
                    background: #8b3636;
                    padding: 20px !important;
                    border: 1px solid #e92a2a;
                    color: #ffff;
                }}
                .warning-box h4 {{
                    margin: 0;
                    padding: 0;
                    font-size: 20px;
                    font-weight: bold;
                    margin-top: 0px !important;
                }}
                .warning-box p {{
                    margin: 10px 0 0;
                    font-size: 16px;
                    line-height: 1.5;
                    color: #fff !important;
                }}
                </style>
                <div class="warning-box">
                    <h4>⚠️ Warning!</h4>
                    <p>{extended_insight}</p>
                </div>
                """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <style>
            .sources {{
                padding: 20px;
                background: #202736;
                margin-top: 20px;
                border: solid 1px #798aad;
            }}
            
            .sources h4 {{
                margin-top: 0px;
                padding-top: 0px;
            }}
            
            .sources a {{
                color: #ef553b;
                font-weight: bold;
            }}
        </style>
        <div class="sources">
        <h4>Data Sources for Verification</h4>
        <ol>
            <li><a href="https://www.globalforestwatch.org/](https://www.globalforestwatch.org/">Global Forest Watch</a>: 
           A comprehensive platform for forest monitoring, deforestation, and biodiversity data.</li>
           <li><a href="https://mol.org/](https://mol.org/">Map of Life (MOL)</a>: Provides global biodiversity and species distribution data.</li>
        </ol>
        </div>
           """,
        unsafe_allow_html=True)