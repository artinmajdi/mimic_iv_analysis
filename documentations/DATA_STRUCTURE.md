# MIMIC-IV Database Structure (v3.1)

## Directory Structure
```
mimic-iv-3.1/
├── hosp/                   # Hospital-wide data
│   ├── admissions.csv(.gz)         # Patient hospital admissions
│   ├── patients.csv(.gz)           # Patient demographics
│   ├── labevents.csv(.gz)         # Laboratory measurements (17GB)
│   ├── microbiologyevents.csv(.gz) # Microbiology results
│   ├── pharmacy.csv(.gz)           # Pharmacy orders
│   ├── prescriptions.csv(.gz)      # Medication prescriptions
│   ├── procedures_icd.csv(.gz)     # Patient procedures
│   ├── diagnoses_icd.csv(.gz)      # Patient diagnoses
│   ├── emar.csv(.gz)               # Medication administration
│   ├── emar_detail.csv(.gz)        # Detailed medication data
│   ├── poe.csv(.gz)                # Provider order entries
│   ├── poe_detail.csv(.gz)         # Detailed order information
│   ├── d_hcpcs.csv(.gz)            # HCPCS code definitions
│   ├── d_icd_diagnoses.csv(.gz)    # ICD diagnosis codes
│   ├── d_icd_procedures.csv(.gz)   # ICD procedure codes
│   ├── d_labitems.csv(.gz)         # Lab item definitions
│   ├── hcpcsevents.csv(.gz)        # HCPCS events
│   ├── drgcodes.csv(.gz)           # DRG codes
│   ├── services.csv(.gz)           # Hospital services
│   ├── transfers.csv(.gz)          # Patient transfers
│   ├── provider.csv(.gz)           # Provider information
│   └── omr.csv(.gz)                # Order monitoring
├── icu/                    # ICU-specific data
│   ├── chartevents.csv(.gz)        # Patient charting data
│   ├── datetimeevents.csv(.gz)     # Date/time events
│   ├── inputevents.csv(.gz)        # Patient intake data
│   ├── outputevents.csv(.gz)       # Patient output data
│   ├── procedureevents.csv(.gz)    # ICU procedures
│   ├── ingredientevents.csv(.gz)   # Medication ingredients
│   ├── d_items.csv(.gz)            # ICU item dictionary
│   ├── icustays.csv(.gz)           # ICU stay information
│   └── caregiver.csv(.gz)          # Caregiver information
├── CHANGELOG.txt           # Version changes
├── LICENSE.txt            # Usage license
└── SHA256SUMS.txt         # File checksums
```

## Overview

MIMIC-IV is organized into two main components:
1. Hospital (hosp)
2. Intensive Care Unit (icu)

## Hospital (hosp) Module

### Core Tables

#### admissions.csv
Patient hospital admissions information
- `subject_id`: Unique patient identifier
- `hadm_id`: Unique hospital admission identifier
- `admittime`: Admission time
- `dischtime`: Discharge time
- `deathtime`: Time of death
- `admission_type`: Type of admission
- `admit_provider_id`: Provider ID
- `admission_location`: Location patient was admitted from
- `discharge_location`: Location patient was discharged to
- `insurance`: Insurance type
- `language`: Primary language
- `marital_status`: Marital status
- `race`: Race
- `edregtime`: Emergency department registration time
- `edouttime`: Emergency department exit time
- `hospital_expire_flag`: Death flag

#### patients.csv
Patient demographic data
- `subject_id`: Unique patient identifier
- `gender`: Patient's gender
- `anchor_age`: Patient's age
- `anchor_year`: Year of patient record
- `anchor_year_group`: Grouped year
- `dod`: Date of death

### Clinical Event Tables

- `labevents.csv` (17GB): Laboratory measurements
- `microbiologyevents.csv` (867MB): Microbiology test results
- `pharmacy.csv` (3.7GB): Pharmacy orders
- `prescriptions.csv` (3.2GB): Medication prescriptions
- `procedures_icd.csv` (33MB): Patient procedures
- `diagnoses_icd.csv` (173MB): Patient diagnoses

### Medication-related Tables

- `emar.csv` (5.8GB): Electronic medication administration records
- `emar_detail.csv` (8.1GB): Detailed medication administration data
- `poe.csv` (4.8GB): Provider order entries
- `poe_detail.csv` (405MB): Detailed order information

### Dictionary Tables

- `d_hcpcs.csv`: HCPCS code definitions
- `d_icd_diagnoses.csv`: ICD diagnosis code definitions
- `d_icd_procedures.csv`: ICD procedure code definitions
- `d_labitems.csv`: Laboratory test definitions

### Other Tables

- `hcpcsevents.csv` (11MB): Healthcare Common Procedure Coding System events
- `drgcodes.csv` (52MB): Diagnosis-related group codes
- `services.csv` (25MB): Hospital services
- `transfers.csv` (196MB): Patient transfers
- `provider.csv` (289KB): Provider information
- `omr.csv` (306MB): Order monitoring results

## Intensive Care Unit (icu) Module

### Core Tables

#### chartevents.csv
Patient charting data (vital signs, etc.)
- `subject_id`: Unique patient identifier
- `hadm_id`: Hospital admission identifier
- `stay_id`: ICU stay identifier
- `caregiver_id`: Caregiver identifier
- `charttime`: Time of charting
- `storetime`: Time of record storage
- `itemid`: Identifier for the charted item
- `value`: Recorded value
- `valuenum`: Numeric value (if applicable)
- `valueuom`: Unit of measurement
- `warning`: Warning flag

### Event Tables

- `datetimeevents.csv.gz` (61MB): Date/time-based events
- `inputevents.csv.gz` (383MB): Patient intake data
- `outputevents.csv.gz` (47MB): Patient output data
- `procedureevents.csv.gz` (23MB): ICU procedures
- `ingredientevents.csv.gz` (297MB): Detailed medication ingredients

### Dictionary and Reference Tables

- `d_items.csv.gz` (57KB): Dictionary of ICU items
- `icustays.csv.gz` (3.2MB): ICU stay information
- `caregiver.csv.gz` (41KB): Caregiver information

## File Sizes and Formats

Most files are available in both CSV and compressed CSV.GZ format. The compressed versions are significantly smaller but require decompression before use. For example:
- `labevents.csv` (17GB) → `labevents.csv.gz` (2.4GB)
- `emar_detail.csv` (8.1GB) → `emar_detail.csv.gz` (713MB)
- `poe.csv` (4.8GB) → `poe.csv.gz` (636MB)

## Notes

1. File sizes indicate the scale of data available in each table
2. Some files are available in both compressed (.gz) and uncompressed formats
3. Dictionary tables (d_*) contain reference information and are relatively small
4. The largest tables are typically those containing patient events and measurements

## Recommended Project Structure for Analysis

For analyzing the MIMIC-IV dataset, we recommend organizing your project as follows:

```
mimic_iv_analysis/
├── src/                      # Source code directory
│   ├── __init__.py          # Package initialization
│   ├── analysis/            # Analysis modules
│   │   ├── __init__.py
│   │   ├── icu_analysis.py  # ICU-specific analysis
│   │   ├── hosp_analysis.py # Hospital-wide analysis
│   │   └── cohort.py        # Cohort selection logic
│   ├── data/                # Data handling modules
│   │   ├── __init__.py
│   │   ├── loaders.py       # Data loading utilities
│   │   ├── processors.py    # Data processing utilities
│   │   └── validators.py    # Data validation utilities
│   ├── core/                # Core functionality
│   │   ├── __init__.py
│   │   ├── constants.py     # Project constants
│   │   └── config.py        # Configuration management
│   ├── utils/               # Utility functions
│   │   ├── __init__.py
│   │   ├── db_utils.py      # Database utilities
│   │   └── preprocessing.py # Data preprocessing utilities
│   └── visualization/       # Visualization modules
│       ├── __init__.py
│       ├── plotters.py      # Plotting functions
│       └── dashboards.py    # Dashboard creation
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── unit/               # Unit tests
│   │   ├── __init__.py
│   │   ├── test_loaders.py
│   │   └── test_processors.py
│   └── integration/        # Integration tests
│       ├── __init__.py
│       └── test_pipeline.py
├── notebooks/              # Jupyter notebooks
│   ├── exploratory/       # Exploratory analysis
│   └── results/           # Results and figures
├── docs/                  # Documentation
│   ├── data_dictionary.md # Detailed data dictionary
│   └── analysis_guide.md  # Analysis documentation
├── scripts/               # Utility scripts
│   ├── setup_db.sh       # Database setup
│   └── extract_data.py   # Data extraction
├── configs/              # Configuration files
│   ├── db_config.yaml    # Database configuration
│   └── analysis_config.yaml # Analysis parameters
├── .env.example          # Example environment variables
├── .gitignore           # Git ignore rules
├── LICENSE              # Project license
├── pyproject.toml       # Project configuration
├── README.md           # Project documentation
└── setup.py            # Setup configuration
```

### Key Components

1. **src/analysis/**: Contains the main analysis logic
   - Separate modules for ICU and hospital-wide analysis
   - Cohort selection and management

2. **src/data/**: Data handling utilities
   - Standardized data loading from CSV/CSV.GZ files
   - Data processing and validation

3. **src/visualization/**: Visualization tools
   - Standard plotting functions
   - Interactive dashboard creation

4. **notebooks/**: Jupyter notebooks
   - Exploratory analysis
   - Results presentation

5. **configs/**: Configuration management
   - Database connection settings
   - Analysis parameters

6. **tests/**: Comprehensive test suite
   - Unit tests for individual components
   - Integration tests for full pipeline

### Best Practices

1. Use environment variables for sensitive information
2. Keep raw data separate from analysis code
3. Document analysis steps and assumptions
4. Version control all code and configurations
5. Maintain test coverage for critical components
6. Use consistent coding style throughout the project
