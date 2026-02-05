"""
Download precipitation and renewable freshwater data from WorldBank API.
This script fetches data for European countries from 1990-2022.
"""

import pandas as pd
import requests
import time
from io import StringIO

print("="*70)
print("Downloading data from WorldBank API")
print("="*70)

# Load reference European countries
print("\nLoading reference countries...")
df_ref = pd.read_csv('FAOSTAT/reference_Europe.csv', encoding='latin1')
countries = sorted(df_ref['Area'].unique())
print(f"Target: {len(countries)} European countries")

# WorldBank country codes mapping (partial - common ones)
# Note: WorldBank uses ISO 3-letter codes
country_code_map = {
    'Albania': 'ALB', 'Austria': 'AUT', 'Belarus': 'BLR', 'Belgium': 'BEL',
    'Bosnia and Herzegovina': 'BIH', 'Bulgaria': 'BGR', 'Croatia': 'HRV',
    'Czechia': 'CZE', 'Denmark': 'DNK', 'Estonia': 'EST', 'Finland': 'FIN',
    'France': 'FRA', 'Germany': 'DEU', 'Greece': 'GRC', 'Hungary': 'HUN',
    'Iceland': 'ISL', 'Ireland': 'IRL', 'Italy': 'ITA', 'Latvia': 'LVA',
    'Lithuania': 'LTU', 'Luxembourg': 'LUX', 'Moldova': 'MDA', 'Montenegro': 'MNE',
    'Netherlands': 'NLD', 'North Macedonia': 'MKD', 'Norway': 'NOR', 'Poland': 'POL',
    'Portugal': 'PRT', 'Romania': 'ROU', 'Russia': 'RUS', 'Serbia': 'SRB',
    'Slovakia': 'SVK', 'Slovenia': 'SVN', 'Spain': 'ESP', 'Sweden': 'SWE',
    'Switzerland': 'CHE', 'Ukraine': 'UKR', 'United Kingdom': 'GBR',
    'Republic of Moldova': 'MDA', 'Russian Federation': 'RUS',
    'Serbia and Montenegro': 'SRB', 'Czechia': 'CZE', 'Slovakia': 'SVK'
}

def download_worldbank_indicator(indicator_code, indicator_name):
    """Download data from WorldBank API for a specific indicator."""

    print(f"\n{indicator_name}")
    print("-" * 50)

    all_data = []
    successful_countries = []
    failed_countries = []

    for country in countries:
        if country not in country_code_map:
            print(f"  ⚠ Skipping {country} (no country code mapping)")
            failed_countries.append(country)
            continue

        code = country_code_map[country]

        # WorldBank API URL
        url = f"http://api.worldbank.org/v2/country/{code}/indicator/{indicator_code}"
        params = {
            'date': '1990:2022',
            'format': 'json',
            'per_page': 500
        }

        try:
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                if len(data) > 1 and data[1]:
                    for entry in data[1]:
                        if entry['value'] is not None:
                            all_data.append({
                                'Area': country,
                                'Year': entry['date'],
                                'value': entry['value']
                            })
                    successful_countries.append(country)
                    print(f"  ✓ {country}")
                else:
                    failed_countries.append(country)
                    print(f"  ✗ {country} (no data)")
            else:
                failed_countries.append(country)
                print(f"  ✗ {country} (HTTP {response.status_code})")

            time.sleep(0.1)  # Be nice to the API

        except Exception as e:
            failed_countries.append(country)
            print(f"  ✗ {country} ({str(e)})")

    print(f"\nResults: {len(successful_countries)} successful, {len(failed_countries)} failed")

    return pd.DataFrame(all_data), successful_countries, failed_countries


# 1. Precipitation data
# Indicator: AG.LND.PRCP.MM (Average precipitation in depth, mm per year)
print("\n" + "="*70)
print("1. PRECIPITATION DATA")
print("="*70)

df_precip, precip_success, precip_failed = download_worldbank_indicator(
    'AG.LND.PRCP.MM',
    'Average precipitation (mm/year)'
)

if not df_precip.empty:
    df_precip['Year'] = df_precip['Year'].astype(int)
    df_precip = df_precip.rename(columns={'value': 'precip'})
    df_precip = df_precip.sort_values(['Area', 'Year'])
    df_precip.to_csv('precip.csv', index=False)
    print(f"\n✓ Saved precip.csv ({len(df_precip)} rows)")
    print(f"  Year range: {df_precip['Year'].min()} - {df_precip['Year'].max()}")
    print(f"  Countries: {len(df_precip['Area'].unique())}")
else:
    print("\n✗ No precipitation data downloaded")


# 2. Renewable freshwater resources
# Indicator: ER.H2O.INTR.PC (Renewable internal freshwater resources per capita, m³)
print("\n" + "="*70)
print("2. RENEWABLE FRESHWATER DATA")
print("="*70)

df_rfw, rfw_success, rfw_failed = download_worldbank_indicator(
    'ER.H2O.INTR.PC',
    'Renewable freshwater per capita (m³/year)'
)

if not df_rfw.empty:
    df_rfw['Year'] = df_rfw['Year'].astype(int)
    df_rfw = df_rfw.rename(columns={'value': 'rfw'})
    df_rfw = df_rfw.sort_values(['Area', 'Year'])
    df_rfw.to_csv('renewable_freshwater.csv', index=False)
    print(f"\n✓ Saved renewable_freshwater.csv ({len(df_rfw)} rows)")
    print(f"  Year range: {df_rfw['Year'].min()} - {df_rfw['Year'].max()}")
    print(f"  Countries: {len(df_rfw['Area'].unique())}")
else:
    print("\n✗ No freshwater data downloaded")


# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print("\nFiles created:")
if not df_precip.empty:
    print(f"  ✓ precip.csv ({len(df_precip)} rows, {len(df_precip['Area'].unique())} countries)")
else:
    print("  ✗ precip.csv (failed)")

if not df_rfw.empty:
    print(f"  ✓ renewable_freshwater.csv ({len(df_rfw)} rows, {len(df_rfw['Area'].unique())} countries)")
else:
    print("  ✗ renewable_freshwater.csv (failed)")

if precip_failed or rfw_failed:
    print("\nCountries with missing data:")
    all_failed = set(precip_failed + rfw_failed)
    for country in sorted(all_failed):
        print(f"  - {country}")
    print(f"\nNote: {len(all_failed)} countries have missing or incomplete data")
    print("These will be handled by the script's imputation step.")

print("\nNext step: Run the main analysis script!")
print("  python ind_lu_ad_tut3_shap_6_fao+3_eustat_iforest_grid_search_no_training_inverse_anomaly_score_color_set_inverse_enrich_pca.py")
