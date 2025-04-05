"""MIMIC-IV Analysis Package.

A comprehensive toolkit for analyzing MIMIC-IV clinical database.
"""

from mimic_iv_analysis.data import (
    load_patient_data,
    load_admissions,
    load_diagnoses,
    load_procedures,
    load_medications,
    load_lab_results,
)

from mimic_iv_analysis.analysis import (
    perform_eda,
    analyze_demographics,
    analyze_temporal_patterns,
    analyze_order_patterns,
    identify_common_sequences,
    interpret_findings,
    analyze_clinical_outcomes,
)

from mimic_iv_analysis.core import (
    train_model,
    evaluate_model,
    predict,
    cross_validate,
)

from mimic_iv_analysis.visualization import (
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
