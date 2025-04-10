# MIMIC-IV Provider Order Pattern Analysis

This project analyzes provider order patterns in the MIMIC-IV 3.1 database to identify clusters associated with shorter length of stay for Type 2 Diabetes patients.

## Project Structure

```
.
├── src/
│   ├── data/           # Data loading and preprocessing
│   ├── models/         # Clustering and machine learning models
│   ├── utils/          # Utility functions
│   └── visualization/  # Visualization components
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up the dataset path:
The application expects the MIMIC-IV 3.1 dataset to be located at:
`/Users/artinmajdi/Documents/GitHubs/Career/mimic_iv/dataset/mimic-iv-3.1`

## Running the Application

To start the Streamlit application:
```bash
streamlit run src/app.py
```

## Features

- Memory-efficient data processing using Dask
- Provider order pattern clustering
- Length of stay analysis
- Interactive visualizations
- Statistical significance testing

## Data Processing Pipeline

1. Data Loading: Chunked reading with Dask DataFrames
2. Preprocessing: Patient cohort filtering
3. Feature Engineering: Order matrices and temporal sequences
4. Clustering: Multiple clustering techniques
5. Analysis: Length of stay comparison and statistical testing
6. Visualization: Interactive dashboards and reports
