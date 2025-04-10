# MIMIC-IV Data Pipeline

A comprehensive data pipeline for processing and analyzing MIMIC-IV data, with a focus on machine learning model development and fairness analysis.

## Features

- **Data Preprocessing**
  - Cohort extraction
  - Feature extraction (diagnoses, procedures, medications, lab tests)
  - Data cleaning and imputation
  - Summary statistics generation

- **Feature Selection**
  - Feature importance analysis
  - Feature subset selection
  - Feature visualization

- **Model Training**
  - Multiple model architectures (LSTM, GRU, Transformer)
  - Hyperparameter tuning
  - Training progress monitoring
  - Model checkpointing

- **Evaluation**
  - Performance metrics calculation
  - ROC curves
  - Precision-recall curves
  - Confusion matrices

- **Fairness Analysis**
  - Demographic parity
  - Equal opportunity
  - Equalized odds
  - Treatment equality

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MIMIC-IV-Data-Pipeline.git
cd MIMIC-IV-Data-Pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up MIMIC-IV data:
   - Download MIMIC-IV data from PhysioNet
   - Place the data in the `data/mimic` directory
   - Update the path in the Streamlit app if needed

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Navigate through the pipeline using the sidebar:
   - Data Preprocessing
   - Feature Selection
   - Model Training
   - Evaluation
   - Fairness Analysis

## Project Structure

```
MIMIC-IV-Data-Pipeline/
├── app.py                  # Streamlit application
├── preprocessing_module.py # Data preprocessing module
├── model_module.py        # Model training and evaluation module
├── preprocessing/         # Preprocessing scripts
│   ├── day_intervals_preproc/
│   └── hosp_module_preproc/
├── model/                 # Model scripts
│   ├── dl_train.py
│   ├── evaluation.py
│   └── fairness.py
├── data/                  # Data directory
│   ├── mimic/            # MIMIC-IV data
│   ├── cohort/           # Extracted cohorts
│   ├── features/         # Extracted features
│   ├── summary/          # Summary statistics
│   └── models/           # Trained models
└── requirements.txt       # Project dependencies
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MIMIC-IV database
- PhysioNet
- Contributors and maintainers
