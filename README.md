# Agricultural Anomaly Detection Project

This project performs anomaly detection on agricultural and environmental data for European countries using Isolation Forest and SHAP explainability.

## Setup

### 1. Virtual Environment

Activate the virtual environment:
```bash
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Data Requirements

### Completed ✓
- **FAOSTAT Data**: Pesticides, Fertilizers, Crops, Temperature, Population (in `FAOSTAT/` folder)
- **GHG Emissions**: `ghg_emissions.csv` (extracted from FAOSTAT)

### Still Needed ✗

1. **Precipitation Data** (`precip.csv`)
   - Columns: Area, Year, precip
   - Years: 1990-2022
   - 45 European countries
   - Sources: WorldBank, CRU, FAOSTAT Climate

2. **Renewable Freshwater** (`renewable_freshwater.csv`)
   - Columns: Area, Year, rfw
   - Sources: WorldBank, FAO Aquastat

3. **Land Use** (`land-use-over-the-long-term.csv`)
   - Download from: [Our World in Data](https://ourworldindata.org/land-use)
   - Or: [OWID GitHub](https://github.com/owid/owid-datasets)
   - Columns: Entity (Area), Year, Land use: Built-up area, Land use: Grazingland, Land use: Cropland

## Helper Scripts

### `download_missing_data.py`
Creates template CSV files with the correct structure for missing datasets.

```bash
python download_missing_data.py
```

### `extract_ghg_from_faostat.py`
Extracts GHG emissions data from FAOSTAT files (already run).

```bash
python extract_ghg_from_faostat.py
```

## Running the Analysis

Once all data files are in place:

```bash
python ind_lu_ad_tut3_shap_6_fao+3_eustat_iforest_grid_search_no_training_inverse_anomaly_score_color_set_inverse_enrich_pca.py
```

## Output

The script generates:
- **Visualizations**: 3D t-SNE plots (HTML), SHAP plots, anomaly distributions
- **Data files**: `input_dataset.xlsx`, `input_dataset.csv`, `year_area_joiner7.csv`
- **SHAP analysis**: Force plots, waterfall plots, feature importance

## Data Sources

- **FAOSTAT**: Agricultural data (pesticides, fertilizers, crops, etc.)
- **Our World in Data**: Land use, GHG emissions
- **WorldBank**: Climate data (precipitation, freshwater)

## Project Structure

```
.
├── venv/                           # Virtual environment
├── FAOSTAT/                        # FAOSTAT ZIP files
├── FAOSTAT_extract/                # Extracted CSV files
├── requirements.txt                # Python dependencies
├── download_missing_data.py        # Helper script
├── extract_ghg_from_faostat.py    # GHG extraction script
├── ghg_emissions.csv              # ✓ Generated
├── precip.csv                     # ✗ Needed
├── renewable_freshwater.csv       # ✗ Needed
├── land-use-over-the-long-term.csv # ✗ Needed
└── ind_lu_ad_tut3_shap_6_...py   # Main analysis script
```

## Notes

- Python 3.9+ required
- Uses NumPy <2.0 for compatibility
- Processes 45 European countries from 1990-2022
- Missing values are handled via iterative imputation
