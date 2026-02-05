"""
Helper script to download missing CSV files for anomaly detection analysis.
Downloads data for European countries from 1990-2022.
"""

import pandas as pd
import numpy as np

# Load the reference European countries
print("Loading reference countries...")
df_ref = pd.read_csv('FAOSTAT/reference_Europe.csv', encoding='latin1')
countries = sorted(df_ref['Area'].unique())
print(f"Found {len(countries)} European countries")

# Create year range
years = list(range(1990, 2023))  # 1990-2022

print("\nPreparing datasets...")

# 1. Land use data
# This data is available from Our World in Data
# URL: https://github.com/owid/owid-datasets/tree/master/datasets/Long-term%20land%20use
print("\n1. Land use data:")
print("   Please download from: https://ourworldindata.org/land-use")
print("   Look for 'land-use-over-the-long-term.csv'")
print("   Or use the dataset from: https://github.com/owid/owid-datasets")
print("   Required columns: Entity (Area), Year, 'Land use: Built-up area', 'Land use: Grazingland', 'Land use: Cropland'")

# 2. Precipitation data
# Can use WorldBank Climate API or create template
print("\n2. Precipitation data:")
print("   Creating template file 'precip_template.csv'...")
print("   You can fill this with data from:")
print("   - WorldBank Climate Portal: https://climateknowledgeportal.worldbank.org/")
print("   - CRU dataset: https://crudata.uea.ac.uk/")
print("   - FAOSTAT Climate data")

# Create template with structure
precip_data = []
for country in countries:
    for year in years:
        precip_data.append({
            'Area': country,
            'Year': year,
            'precip': np.nan  # Fill with actual precipitation data (mm/year)
        })

df_precip = pd.DataFrame(precip_data)
df_precip.to_csv('precip_template.csv', index=False)
print("   ✓ Created 'precip_template.csv'")

# 3. Renewable Freshwater
print("\n3. Renewable freshwater data:")
print("   Creating template file 'renewable_freshwater_template.csv'...")
print("   You can fill this with data from:")
print("   - WorldBank: https://data.worldbank.org/indicator/ER.H2O.INTR.PC")
print("   - FAOSTAT Aquastat")
print("   - FAO Aquastat: https://www.fao.org/aquastat/")

rfw_data = []
for country in countries:
    for year in years:
        rfw_data.append({
            'Area': country,
            'Year': year,
            'rfw': np.nan  # Fill with renewable freshwater per capita (m³/year)
        })

df_rfw = pd.DataFrame(rfw_data)
df_rfw.to_csv('renewable_freshwater_template.csv', index=False)
print("   ✓ Created 'renewable_freshwater_template.csv'")

# 4. GHG Emissions
print("\n4. GHG emissions data:")
print("   Creating template file 'ghg_emissions_template.csv'...")
print("   You can fill this with data from:")
print("   - FAOSTAT Emissions database (already in your FAOSTAT folder)")
print("   - Our World in Data: https://ourworldindata.org/co2-and-greenhouse-gas-emissions")
print("   - WorldBank: https://data.worldbank.org/indicator/EN.ATM.GHGT.KT.CE")

# Check if FAOSTAT has emissions data
import os
emissions_files = [f for f in os.listdir('FAOSTAT') if 'Emission' in f or 'Climate_change' in f]
if emissions_files:
    print(f"\n   Note: Found emissions files in FAOSTAT folder:")
    for f in emissions_files[:5]:
        print(f"   - {f}")
    print("   You might be able to extract GHG data from these!")

ghg_data = []
for country in countries:
    for year in years:
        ghg_data.append({
            'Area': country,
            'Year': year,
            'ghg_emissions': np.nan  # Fill with GHG emissions (kt CO2 equivalent)
        })

df_ghg = pd.DataFrame(ghg_data)
df_ghg.to_csv('ghg_emissions_template.csv', index=False)
print("   ✓ Created 'ghg_emissions_template.csv'")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("\nTemplate files created:")
print("  ✓ precip_template.csv")
print("  ✓ renewable_freshwater_template.csv")
print("  ✓ ghg_emissions_template.csv")
print("\nStill needed:")
print("  ✗ land-use-over-the-long-term.csv (download from Our World in Data)")
print("\nNext steps:")
print("1. Fill the template files with actual data from the sources listed above")
print("2. Rename templates by removing '_template' from filenames")
print("3. Download land-use data from Our World in Data")
print("4. Run the main anomaly detection script")
print("\nAlternatively, you can extract some data from your FAOSTAT folder!")
