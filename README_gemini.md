# MIMIC-IV Analysis Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

A comprehensive analytical toolkit for exploring and modeling data from the MIMIC-IV clinical database. This project provides tools for data loading, preprocessing, feature engineering, clustering, and visualization, primarily focusing on provider order pattern analysis.

## Features

* **Data Loading & Preprocessing:** Utilities for loading and preparing MIMIC-IV data.
* **Feature Engineering:** Tools for creating features from clinical temporal data, including order frequency matrices, temporal order sequences, and order timing features.
* **Clustering Analysis:** Implementations for K-Means, Hierarchical, DBSCAN clustering, and LDA Topic Modeling to identify patterns in clinical data.
* **Predictive Modeling Support:** Designed to prepare data for various predictive tasks. [cite: 5]
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
