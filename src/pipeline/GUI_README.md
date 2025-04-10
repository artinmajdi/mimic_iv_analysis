# MIMIC-IV Data Pipeline GUI

This is a modern graphical user interface for the MIMIC-IV Data Pipeline project. The GUI provides an intuitive way to interact with the pipeline's capabilities, including data preprocessing, feature selection, model training, evaluation, and fairness analysis.

## Features

- **Data Preprocessing**: Configure and run the data preprocessing pipeline
- **Feature Selection**: Select and extract features from the MIMIC-IV dataset
- **Model Training**: Train various machine learning models on the processed data
- **Evaluation**: Evaluate model performance with various metrics and visualizations
- **Fairness Analysis**: Analyze model fairness across different protected attributes

## Quick Start

### Using Setup Scripts (Recommended)

#### On macOS/Linux:
```bash
# Make the setup script executable
chmod +x setup.sh

# Run the setup script
./setup.sh

# Run the application
./run_gui.sh
```

#### On Windows:
```bash
# Run the setup script
setup.bat

# Run the application
run_gui.bat
```

### Manual Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MIMIC-IV-Data-Pipeline.git
cd MIMIC-IV-Data-Pipeline
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the Streamlit application:
```bash
streamlit run app.py
```

## Usage

1. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

2. Use the sidebar to navigate between different modules:
   - Data Preprocessing
   - Feature Selection
   - Model Training
   - Evaluation
   - Fairness Analysis

## Configuration

- The default MIMIC-IV data path is set to `/Users/artinmajdi/Documents/Datasets_Models/MIMIC_IV/mimic-iv-3.1`
- You can modify this path in the Data Preprocessing section of the GUI
- All configuration options are available through the intuitive interface

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the same license as the MIMIC-IV Data Pipeline project.

## Acknowledgments

- MIMIC-IV Data Pipeline team
- Streamlit team for the amazing framework
- PhysioNet for providing the MIMIC-IV dataset
