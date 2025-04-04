# MIMIC-IV Analysis

A comprehensive analytical toolkit for exploring and modeling data from the MIMIC-IV clinical database.

## Features

- Exploratory Data Analysis
- Patient Trajectory Visualization
- Order Pattern Analysis
- Predictive Modeling

## Installation

### Prerequisites

- Python 3.12 or higher
- pip or conda package manager

### Option 1: Using pip

```bash
pip install -e ".[dev]"  # Install with development dependencies
```

### Option 2: Using Docker

```bash
docker-compose up
```

### Option 3: Manual Installation

1. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -e ".[dev]"  # Install with development dependencies
```

## Development

### Code Style

This project uses:

- black for code formatting
- isort for import sorting
- flake8 for style guide enforcement
- mypy for static type checking

To format code:

```bash
black .
isort .
```

To check code:

```bash
flake8 .
mypy .
```

### Running Tests

```bash
pytest tests/
pytest --cov=mimic_iv_analysis tests/  # With coverage
```

## Project Structure

```
mimic_iv_analysis/
├── mimic_iv_analysis/        # Main package
│   ├── __init__.py          # Package initialization
│   ├── analysis/            # Analysis modules
│   ├── data/                # Data handling modules
│   ├── models/              # ML models
│   ├── utils/               # Utility functions
│   └── visualization/       # Visualization modules
├── tests/                   # Test suite
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── docs/                    # Documentation
│   └── README.md           # Documentation home
├── scripts/                 # Utility scripts
│   └── install.sh          # Installation script
├── config/                  # Configuration files
│   └── config.yaml         # Main config
├── .env.example            # Example environment variables
├── .gitignore              # Git ignore rules
├── LICENSE                 # Project license
├── pyproject.toml          # Project configuration
├── README.md               # Project documentation
└── setup.py                # Legacy setup file
```

## Usage

Run the Streamlit dashboard:

```bash
streamlit run mimic_iv_analysis/app.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

MIT
