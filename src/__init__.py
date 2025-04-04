"""MIMIC-IV Analysis Package.

A comprehensive toolkit for analyzing MIMIC-IV clinical database.
"""

from src.data import (
    load_patient_data,
    load_admissions,
    load_diagnoses,
    load_procedures,
    load_medications,
    load_lab_results,
)

from src.analysis import (
    perform_eda,
    analyze_demographics,
    analyze_temporal_patterns,
    analyze_order_patterns,
    identify_common_sequences,
    interpret_findings,
    analyze_clinical_outcomes,
)

from src.core import (
    train_model,
    evaluate_model,
    predict,
    cross_validate,
)

from src.visualization import (
    plot_patient_trajectory,
    visualize_timeline,
    create_sankey_diagram,
)

__version__ = "0.1.0"
__author__ = "Artin Majdi"

__all__ = [
    # Data loading functions
    'load_patient_data',
    'load_admissions',
    'load_diagnoses',
    'load_procedures',
    'load_medications',
    'load_lab_results',

    # Analysis functions
    'perform_eda',
    'analyze_demographics',
    'analyze_temporal_patterns',
    'analyze_order_patterns',
    'identify_common_sequences',
    'interpret_findings',
    'analyze_clinical_outcomes',

    # Model functions
    'train_model',
    'evaluate_model',
    'predict',
    'cross_validate',

    # Visualization functions
    'plot_patient_trajectory',
    'visualize_timeline',
    'create_sankey_diagram',
]
