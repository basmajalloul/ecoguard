import pandas as pd

# File path to the CSV
file_path = "data/LPD_2024_public.csv"

data = pd.read_csv(file_path, encoding="ISO-8859-1")

# Step 2: Filter for Brazil, Congo, and Indonesia
countries_of_interest = ["Brazil", "Congo, The Democratic Republic Of The", "Indonesia", "Australia", "Cameroon", "United States", "France", "Germany"]
filtered_data = data[data["Country"].isin(countries_of_interest)]

# Step 3: Reshape the data (melt years into a single column)
year_columns = [str(year) for year in range(2001, 2020)]  # Only keep 2001-2023
melted_data = pd.melt(
    filtered_data,
    id_vars=[
        "Binomial", "Common_name", "Location", "Country", "Latitude", "Longitude"
    ],
    value_vars=year_columns,
    var_name="Year",
    value_name="Population"
)

# Step 4: Drop rows with missing population values
melted_data = melted_data.dropna(subset=["Population"])

# Step 5: Export or display the cleaned dataset
output_path = "data/filtered_species_data.csv"
melted_data.to_csv(output_path, index=False)

print(f"Filtered data saved to: {output_path}")
