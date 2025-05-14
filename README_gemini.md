# MIMIC-IV Analysis Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

A comprehensive analytical toolkit for exploring and modeling data from the MIMIC-IV clinical database. This project provides tools for data loading, preprocessing, feature engineering, clustering, and visualization, primarily focusing on provider order pattern analysis.

## Features

* **Data Loading & Preprocessing:** Utilities for loading and preparing MIMIC-IV data.
* **Feature Engineering:** Tools for creating features from clinical temporal data, including order frequency matrices, temporal order sequences, and order timing features.
* **Clustering Analysis:** Implementations for K-Means, Hierarchical, DBSCAN clustering, and LDA Topic Modeling to identify patterns in clinical data.
* **Predictive Modeling Support:** Designed to prepare data for various predictive tasks.
* **Interactive Dashboard:** A Streamlit application for visualizing data, cluster results, and analysis.
* **Configuration Management:** Easy-to-use YAML configuration for managing data paths and application settings.
* **MIMIC-IV Data Focus:** Specifically designed to work with the MIMIC-IV clinical database structure.

## About MIMIC-IV Data

This toolkit is designed to analyze data from the [MIMIC-IV (Medical Information Mart for Intensive Care IV)](https://mimic.mit.edu/docs/iv/) clinical database. MIMIC-IV is a large, freely-available database comprising de-identified health-related data associated with patients who stayed in critical care units at the Beth Israel Deaconess Medical Center.

For detailed information on the MIMIC-IV data structure used by this project, please refer to the documentation:
* [MIMIC-IV Data Structure Overview](documentations/mimic_iv_data_structure.md)
* [Detailed Table Structures](documentations/DATA_STRUCTURE.md)

## Project Structure

The repository is organized as follows:


```
mimic_iv_analysis/
├── mimic_iv_analysis/ # Main package source code
│ ├── init.py # Package initialization
│ ├── configurations/ # Configuration files (e.g., config.yaml)
│ ├── core/ # Core functionalities (data loading, clustering, feature engineering)
│ │ ├── init.py
│ │ ├── clustering.py
│ │ ├── data_loader.py
│ │ ├── feature_engineering.py
│ │ └── filtering.py
│ ├── examples/ # Example scripts and notebooks
│ └── visualization/ # Streamlit dashboard application and utilities
│ ├── init.py
│ ├── app.py
│ └── app_components/
├── documentations/ # Project documentation
├── scripts/ # Utility and helper scripts (install, run dashboard)
├── setup_config/ # Configuration for setup and testing (e.g., pytest.ini)
├── tests/ # Test suite for the project
├── .streamlit/ # Configuration for Streamlit Cloud deployment
├── README.md # This file
├── requirements.txt # Python package dependencies
└── setup.py # Package setup script
```


## Installation

### Prerequisites

* Python 3.10 or higher (Python 3.12+ recommended as per original README)
* pip or conda package manager

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/artinmajdi/mimic_iv_analysis.git](https://github.com/artinmajdi/mimic_iv_analysis.git)
    cd mimic_iv_analysis
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    The `requirements.txt` file lists all necessary Python packages.
    ```bash
    pip install -r requirements.txt
    ```
    To install the package in editable mode along with development dependencies:
    ```bash
    pip install -e ".[dev]"
    ```
    Alternatively, you can use the provided installation script which offers environment choices (venv, conda, docker):
    ```bash
    bash scripts/install.sh
    ```

## Configuration

The main configuration for the application is located in `mimic_iv_analysis/configurations/config.yaml`.

You **must** update the `mimic_data_path` in this file to point to the root directory of your local MIMIC-IV dataset (version 3.1 or compatible).

Example `config.yaml` structure:
```yaml
data:
  mimic_data_path: "/path/to/your/mimic-iv-data" # <-- IMPORTANT: Update this path

app:
  port: 8501
  theme: "light"
  debug: false

# ... other configurations
```

## **Usage**

The primary way to interact with this toolkit is through the Streamlit dashboard.

**To run the Streamlit dashboard:**

1. Ensure your virtual environment is activated (if you created one).  
2. Make sure you have configured the mimic\_data\_path in config.yaml.  
3. Run the application using:  
   streamlit run mimic\_iv\_analysis/visualization/app.py

   Alternatively, if you used scripts/install.sh and it set up an entry point (like mimic-iv), you might be able to use that, but the streamlit run command is standard. The original README.md also mentioned a potential mimic-iv command.

The dashboard should open in your web browser, typically at http://localhost:8501.

## **Core Modules Overview**

* **mimic\_iv\_analysis.core**: Contains the fundamental logic for data handling and analysis.  
  * data\_loader.py: Utilities for loading MIMIC-IV tables efficiently, supporting both CSV and Parquet formats, with options for Dask integration for large datasets.  
  * feature\_engineering.py: Tools to create meaningful features from raw clinical data, such as order frequencies and temporal sequences.  
  * clustering.py: Implements various clustering algorithms (K-Means, Hierarchical, DBSCAN) and LDA topic modeling.  
  * filtering.py: Enables applying inclusion and exclusion criteria to the dataset.  
* **mimic\_iv\_analysis.visualization**: Houses the Streamlit application.  
  * app.py: The main entry point for the interactive dashboard.  
  * app\_components/: Contains different tabs and UI elements of the dashboard.  
* **mimic\_iv\_analysis.configurations**: Manages application settings.

## **Development**

### **Code Style**

This project uses the following tools to maintain code quality:

* **Black:** For code formatting.  
* **isort:** For import sorting.  
* **Flake8:** For style guide enforcement (PEP 8).  
* **MyPy:** For static type checking.

To format your code:

```bash
black .  
isort .
```

To check your code:

```bash
flake8 .  
mypy .
```

### **Running Tests**

Tests are located in the tests/ directory. To run the test suite:

```bash
pytest tests/
```

To run tests with coverage:

```bash
pytest \--cov=mimic\_iv\_analysis tests/
```

Test configuration can be found in setup\_config/pytest.ini.

## **Documentation**

Further documentation can be found in the documentations/ directory:

* DATA\_STRUCTURE.md: Describes the expected structure of the MIMIC-IV data.  
* mimic\_iv\_data\_structure.md: Provides an overview of MIMIC-IV tables and identifiers.  
* .streamlit/README.md: Guide for deploying the Streamlit application to Streamlit Cloud.  
* The documentations/pyhealth/ directory contains documentation for the PyHealth library, which might be a dependency or a related project.

## **Streamlit Cloud Deployment**

For deploying the dashboard to Streamlit Cloud, refer to the guide in .streamlit/README.md. This includes steps for repository preparation, secret management, and dependency configuration.

## **Contributing**

Contributions are welcome\! Please follow these general steps:

1. Fork the repository.  
2. Create a new feature branch (git checkout \-b feature/your-feature-name).  
3. Make your changes.  
4. Ensure all tests pass (pytest tests/).  
5. Format your code (black . and isort .).  
6. Submit a pull request with a clear description of your changes.

## **License**

This project is licensed under the MIT License. See the LICENSE file (if included, or assume MIT as per README.md and setup.py) for details.

## **Author**

* Artin Majdi (\[msm2024@gmail.com\])
