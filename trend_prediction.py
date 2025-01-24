import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load your data with standardized column names
def load_data():
    deforestation_data = pd.read_excel("data/global.xlsx", sheet_name="Country tree cover loss")
    carbon_data = pd.read_excel("data/global.xlsx", sheet_name="Country carbon data")
    biodiversity_data = pd.read_csv("data/filtered_species_data_all_countries.csv")

    # Standardize column names
    deforestation_data.rename(columns={"Country": "country"}, inplace=True)
    carbon_data.rename(columns={"Country": "country"}, inplace=True)
    biodiversity_data.rename(columns={"Country": "country"}, inplace=True)

    print(deforestation_data.columns)
    print(carbon_data.columns)
    print(biodiversity_data.columns)

    return deforestation_data, carbon_data, biodiversity_data


# Updated predict_trends function remains the same
def predict_trends(data, column_prefix, start_year, end_year, predict_years, group_by="country"):
    predictions = []
    for group, group_data in data.groupby(group_by):
        years = np.arange(start_year, end_year + 1).reshape(-1, 1)

        # Safely handle missing columns
        values = []
        for year in range(start_year, end_year + 1):
            column_name = f"{column_prefix}_{year}"
            if column_name in group_data:
                values.append(group_data[column_name].sum())
            else:
                values.append(0)

        print(f"Processing group: {group}")
        print(group_data.head())

        values = np.cumsum(values)

        # Train linear regression
        model = LinearRegression()
        model.fit(years, values)

        # Predict for future years
        future_years = np.arange(end_year + 1, end_year + 1 + predict_years).reshape(-1, 1)
        future_values = model.predict(future_years)

        # Save results
        for year, value in zip(future_years.flatten(), future_values):
            predictions.append({"country": group, "year": year, "predicted_value": value})

    return pd.DataFrame(predictions)


def predict_biodiversity_trends(biodiversity_data, start_year, end_year, predict_years, group_by="country"):
    predictions = []
    for group, group_data in biodiversity_data.groupby(group_by):
        years = np.arange(start_year, end_year + 1).reshape(-1, 1)

        # Count extinct species up to each year
        extinction_counts = []
        for year in range(start_year, end_year + 1):
            extinct_count = group_data[(group_data["Year"] <= year) & (group_data["Population"] == 0)].shape[0]
            extinction_counts.append(extinct_count)

        extinction_cumsum = np.cumsum(extinction_counts)

        # Train linear regression
        model = LinearRegression()
        model.fit(years, extinction_cumsum)

        # Predict for future years
        future_years = np.arange(end_year + 1, end_year + 1 + predict_years).reshape(-1, 1)
        future_values = model.predict(future_years)

        # Save results
        for year, value in zip(future_years.flatten(), future_values):
            predictions.append({"country": group, "year": year, "predicted_value": value})

    return pd.DataFrame(predictions)


# Main script
def main():
    deforestation_data, carbon_data, biodiversity_data = load_data()

    # Predict forest loss
    forest_loss_predictions = predict_trends(
        deforestation_data, column_prefix="tc_loss_ha", start_year=2001, end_year=2025, predict_years=20
    )
    forest_loss_predictions["metric"] = "forest_loss"

    # Predict carbon emissions
    carbon_emissions_predictions = predict_trends(
        carbon_data, column_prefix="gfw_forest_carbon_gross_emissions", start_year=2001, end_year=2025, predict_years=20
    )
    carbon_emissions_predictions["metric"] = "carbon_emissions"

    # Predict biodiversity loss
    biodiversity_predictions = predict_biodiversity_trends(
        biodiversity_data, start_year=2001, end_year=2025, predict_years=20
    )
    biodiversity_predictions["metric"] = "biodiversity_loss"

    # Combine predictions
    all_predictions = pd.concat([forest_loss_predictions, carbon_emissions_predictions, biodiversity_predictions])

    # Save to CSV
    all_predictions.to_csv("predicted_trends_2026_2046.csv", index=False)
    print("Predictions saved to predicted_trends_2026_2046.csv")

    # Predict biodiversity loss
    biodiversity_predictions = predict_trends(
        biodiversity_data, column_prefix="population", start_year=2001, end_year=2025, predict_years=20
    )
    biodiversity_predictions["metric"] = "biodiversity_loss"

    # Combine predictions
    all_predictions = pd.concat([forest_loss_predictions, carbon_emissions_predictions, biodiversity_predictions])

    # Save to CSV
    all_predictions.to_csv("predicted_trends_2026_2046.csv", index=False)
    print("Predictions saved to predicted_trends_2026_2046.csv")


if __name__ == "__main__":
    main()
