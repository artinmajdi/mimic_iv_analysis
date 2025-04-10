"""Machine learning models for MIMIC-IV data analysis."""

# Import core modules
from . import predictive_modeling_module
from . import patient_trajectory_module
from . import order_pattern_module
from . import clinical_interpretation_module
from . import eda_module

# Convenience functions for common modeling tasks
def train_model(data, target, model_type='random_forest', **kwargs):
    """Train a predictive model on MIMIC-IV data.

    Args:
        data (pd.DataFrame): Training data
        target (str): Target variable name
        model_type (str): Type of model to train
        **kwargs: Additional model parameters

    Returns:
        object: Trained model
    """
    return predictive_modeling_module.train_model(data, target, model_type, **kwargs)

def evaluate_model(model, test_data, test_target, **kwargs):
    """Evaluate a trained model on test data.

    Args:
        model: Trained model
        test_data (pd.DataFrame): Test data
        test_target (pd.Series): Test target
        **kwargs: Additional evaluation parameters

    Returns:
        dict: Performance metrics
    """
    return predictive_modeling_module.evaluate_model(model, test_data, test_target, **kwargs)

def predict(model, data, **kwargs):
    """Make predictions with a trained model.

    Args:
        model: Trained model
        data (pd.DataFrame): Input data
        **kwargs: Additional prediction parameters

    Returns:
        pd.Series: Predictions
    """
    return predictive_modeling_module.predict(model, data, **kwargs)

def cross_validate(data, target, model_type='random_forest', cv=5, **kwargs):
    """Perform cross-validation for a model.

    Args:
        data (pd.DataFrame): Training data
        target (str): Target variable name
        model_type (str): Type of model to train
        cv (int): Number of cross-validation folds
        **kwargs: Additional parameters

    Returns:
        dict: Cross-validation results
    """
    return predictive_modeling_module.cross_validate(data, target, model_type, cv, **kwargs)

__all__ = [
    # Modules
    'predictive_modeling_module',
    'patient_trajectory_module',
    'order_pattern_module',
    'clinical_interpretation_module',
    'eda_module',

    # Convenience functions
    'train_model',
    'evaluate_model',
    'predict',
    'cross_validate',
]
