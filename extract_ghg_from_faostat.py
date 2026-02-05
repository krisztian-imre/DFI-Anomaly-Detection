"""
Extract GHG emissions data from FAOSTAT Emissions_Totals dataset.
This creates the ghg_emissions.csv file needed for the analysis.
"""

import zipfile
import pandas as pd

print("Extracting GHG emissions data from FAOSTAT...")

# Load reference European countries
df_ref = pd.read_csv('FAOSTAT/reference_Europe.csv', encoding='latin1')
countries = df_ref['Area'].unique()
print(f"Reference countries: {len(countries)}")

# Extract and load FAOSTAT emissions data
zip_file_path = 'FAOSTAT/Emissions_Totals_E_All_Data_(Normalized).zip'
file_path_inside_zip = 'Emissions_Totals_E_All_Data_(Normalized).csv'

print(f"\nExtracting {file_path_inside_zip}...")
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extract(file_path_inside_zip, 'FAOSTAT_extract')

print("Loading emissions data...")
df = pd.read_csv(f'FAOSTAT_extract/{file_path_inside_zip}', encoding='latin1')

print(f"Total rows: {len(df)}")
print(f"\nAvailable elements:")
print(df['Element'].unique())

# Filter for total emissions
# Use 'Emissions (CO2eq)' or similar
filtered_df = df[df['Element'].str.contains('Emissions', case=False, na=False)]

# Filter for European countries
filtered_df = filtered_df[filtered_df['Area'].isin(countries)]

# Keep only necessary columns
result_df = filtered_df[['Area', 'Year', 'Value']].copy()
result_df.columns = ['Area', 'Year', 'ghg_emissions']

# Group by Area and Year (in case there are multiple emission types, sum them)
result_df = result_df.groupby(['Area', 'Year'], as_index=False)['ghg_emissions'].sum()

# Sort by Area and Year
result_df = result_df.sort_values(['Area', 'Year'])

print(f"\nFiltered data:")
print(f"  Countries: {len(result_df['Area'].unique())}")
print(f"  Year range: {result_df['Year'].min()} - {result_df['Year'].max()}")
print(f"  Total rows: {len(result_df)}")

# Save to CSV
result_df.to_csv('ghg_emissions.csv', index=False)
print(f"\n✓ Created 'ghg_emissions.csv'")

print("\nSample data:")
print(result_df.head(10))
