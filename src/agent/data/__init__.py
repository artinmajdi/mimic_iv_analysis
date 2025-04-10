"""Data loading and preprocessing module for MIMIC-IV Analysis."""

from .data_loader import MIMICDataLoader

# Convenience functions for common data operations
def load_patient_data(mimic_path, **kwargs):
    """Load patient data from MIMIC-IV dataset.

    Args:
        mimic_path (str): Path to MIMIC-IV dataset
        **kwargs: Additional parameters for data loader

    Returns:
        pd.DataFrame: Patient data
    """
    loader = MIMICDataLoader(mimic_path)
    return loader.load_patients()

def load_admissions(mimic_path, **kwargs):
    """Load admissions data from MIMIC-IV dataset.

    Args:
        mimic_path (str): Path to MIMIC-IV dataset
        **kwargs: Additional parameters for data loader

    Returns:
        pd.DataFrame: Admissions data
    """
    loader = MIMICDataLoader(mimic_path)
    return loader.load_admissions()

def load_transfers(mimic_path, **kwargs):
    """Load transfers data from MIMIC-IV dataset.

    Args:
        mimic_path (str): Path to MIMIC-IV dataset
        **kwargs: Additional parameters for data loader

    Returns:
        pd.DataFrame: Transfers data
    """
    loader = MIMICDataLoader(mimic_path)
    return loader.load_transfers()

__all__ = [
    'MIMICDataLoader',
    'load_patient_data',
    'load_admissions',
    'load_transfers',
]
