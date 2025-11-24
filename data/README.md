# Chicago Crime Dataset Directory

## Directory Structure

- **`raw/`** - Original Chicago crime dataset (CSV format)
- **`processed/`** - Cleaned and feature-engineered datasets

## Expected Files

### raw/
- `chicago_crimes_500k.csv` - Sampled 500,000 crime records from full dataset

### processed/
- `crimes_cleaned.csv` - Cleaned crime data after preprocessing
- `crimes_featured.csv` - Data with all engineered features
- `clustering_results.csv` - Cluster assignments from all algorithms
- `pca_components.csv` - PCA-transformed features
- `tsne_embeddings.csv` - t-SNE 2D embeddings

## Dataset Specifications

### Chicago Crime Dataset (2001-2025)
- **Full Dataset Size**: 7.8 Million crime records
- **Sample Used**: 500,000 recent crime records
- **Data Source**: [Chicago Data Portal - Crimes 2001 to Present](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2)
- **Format**: CSV (Comma-separated values)
- **Update Frequency**: Daily updates from Chicago Police Department
- **File Size**: ~200-300 MB for 500k records

## Data Acquisition Instructions

### Method 1: Web Interface Download
1. Visit official dataset: https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2
2. Click **"Export"** button (top right corner)
3. Select **"CSV"** format
4. Apply filters if needed:
   - Date range: Last 2-3 years for recent data
   - Click "Export" and download
5. Rename file to `chicago_crimes_500k.csv`
6. Place in `data/raw/` directory

### Method 2: API Access (for automated downloads)
```python
import requests
import pandas as pd

# Chicago Data Portal API endpoint
url = "https://data.cityofchicago.org/resource/ijzp-q8t2.csv"
params = {
    "$limit": 500000,
    "$order": "date DESC"
}

# Download data
response = requests.get(url, params=params)
with open('data/raw/chicago_crimes_500k.csv', 'wb') as f:
    f.write(response.content)
```

## Input Features (22 Variables)

### Crime Identification
- **ID**: Unique crime record identifier (integer)
- **Case Number**: Official police case reference (string, format: HY123456)
- **IUCR**: Illinois Uniform Crime Reporting code (4-digit string)
- **FBI Code**: FBI crime classification code (2-digit string)

### Crime Classification
- **Primary Type**: Main crime category (33 types)
  - THEFT, BATTERY, CRIMINAL DAMAGE, NARCOTICS, ASSAULT, BURGLARY, MOTOR VEHICLE THEFT, ROBBERY, DECEPTIVE PRACTICE, CRIMINAL TRESPASS, WEAPONS VIOLATION, PUBLIC PEACE VIOLATION, OFFENSE INVOLVING CHILDREN, CRIM SEXUAL ASSAULT, SEX OFFENSE, GAMBLING, LIQUOR LAW VIOLATION, ARSON, INTERFERENCE WITH PUBLIC OFFICER, HOMICIDE, KIDNAPPING, INTIMIDATION, STALKING, OBSCENITY, etc.
- **Description**: Detailed crime subcategory description (string)
- **Location Description**: Specific location type (e.g., STREET, RESIDENCE, APARTMENT, SIDEWALK)

### Temporal Information
- **Date**: Complete datetime when crime occurred (MM/DD/YYYY HH:MM:SS AM/PM)
- **Year**: Extracted year of crime (integer, 2001-2025)
- **Updated On**: Last update timestamp for the record (datetime)

### Geographic Features
- **Block**: Anonymized street address block (e.g., "001XX N STATE ST")
- **Latitude**: Geographic latitude coordinate (float, range: 41.6 to 42.1)
- **Longitude**: Geographic longitude coordinate (float, range: -87.9 to -87.5)
- **X Coordinate**: Illinois State Plane coordinate system X (float)
- **Y Coordinate**: Illinois State Plane coordinate system Y (float)
- **Location**: Combined latitude and longitude string "(lat, lon)"

### Administrative Boundaries
- **Beat**: Police beat number - patrol area subdivision (integer, 4-digit)
- **District**: Police district number (integer, 1-25)
- **Ward**: City council ward number (integer, 1-50)
- **Community Area**: Chicago community area number (integer, 1-77)

### Crime Status Flags
- **Arrest**: Boolean flag indicating if arrest was made (True/False or Y/N)
- **Domestic**: Boolean flag for domestic violence incidents (True/False or Y/N)

## Engineered Features (Created During Analysis)

### Temporal Features
- **Hour**: Extracted hour of day (0-23) from datetime
- **Day_of_Week**: Day name (Monday-Sunday) or numeric (0-6)
- **Month**: Month number (1-12)
- **Season**: Season classification (Winter, Spring, Summer, Fall)
- **Is_Weekend**: Boolean flag for weekend crimes (Saturday/Sunday)
- **Is_Night**: Boolean flag for nighttime crimes (8 PM - 5 AM)
- **Time_Period**: Categorical (Morning, Afternoon, Evening, Night)

### Geographic Features
- **Lat_Bin**: Latitude coordinate bins for spatial clustering (20 bins)
- **Lon_Bin**: Longitude coordinate bins for spatial clustering (20 bins)
- **Grid_Cell**: Combined grid cell identifier for spatial analysis

### Crime Severity Features
- **Crime_Severity_Score**: Numerical score (1-5) based on crime type severity
  - 5: HOMICIDE, CRIMINAL SEXUAL ASSAULT
  - 4: ROBBERY, ASSAULT, BATTERY, WEAPONS VIOLATION
  - 3: BURGLARY, THEFT, MOTOR VEHICLE THEFT
  - 2: CRIMINAL DAMAGE, NARCOTICS, DECEPTIVE PRACTICE
  - 1: PUBLIC PEACE VIOLATION, OTHER OFFENSE

### Encoded Features
- **Crime_Type_Encoded**: One-hot encoded crime type (top 10 categories)
- **Location_Type_Encoded**: Encoded location description

## Data Quality Notes

### Expected Data Issues
- **Missing Values**: ~5-10% missing in Location Description, Ward, Community Area
- **Invalid Coordinates**: ~1-2% outside Chicago city bounds
- **Duplicates**: Minimal (~0.1%) duplicate records

### Preprocessing Steps Applied
1. Remove duplicates based on Case Number
2. Drop records with missing critical fields (Lat, Lon, Primary Type, Date)
3. Filter invalid coordinates (outside Chicago bounds)
4. Standardize text fields to uppercase
5. Convert date strings to datetime objects
6. Handle missing categorical values with 'UNKNOWN'

## Usage in Analysis Pipeline

```python
# Load raw data
import pandas as pd
df = pd.read_csv('data/raw/chicago_crimes_500k.csv')

# Load processed data
df_cleaned = pd.read_csv('data/processed/crimes_cleaned.csv')
df_featured = pd.read_csv('data/processed/crimes_featured.csv')
df_clusters = pd.read_csv('data/processed/clustering_results.csv')
```

## Data Privacy & Ethics

- All location data is anonymized to block-level (not exact addresses)
- No personally identifiable information (PII) is included
- Use responsibly for public safety research and urban planning
- Respect data usage policies from Chicago Data Portal

---

**Last Updated**: November 2025  
**Data Version**: 2001-2025 (Daily updates)
