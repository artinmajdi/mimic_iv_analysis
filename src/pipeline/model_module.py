import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
try:
    from model.dl_train import DL_models
    from model.evaluation import Loss
    from model.fairness import fairness_evaluation
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: Could not import model modules")

class ModelModule:
    """
    A wrapper class for the MIMIC-IV model training and evaluation functionality.
    This class provides a simplified interface to the model modules.
    """

    def __init__(self, output_dir="./data/models"):
        """
        Initialize the model module.

        Args:
            output_dir (str): Directory to save model files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Default parameters
        self.model_type = "lstm"
        self.hidden_size = 128
        self.num_layers = 2
        self.dropout = 0.2
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_epochs = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_data(self, feature_paths, cohort_path, target_column, sensitive_columns=None):
        """
        Prepare data for model training.

        Args:
            feature_paths (dict): Dictionary of paths to feature files
            cohort_path (str): Path to cohort file
            target_column (str): Name of the target column
            sensitive_columns (list): List of sensitive attribute columns

        Returns:
            tuple: (X_train, X_test, y_train, y_test, sensitive_train, sensitive_test)
        """
        # Load cohort data
        cohort_df = pd.read_csv(cohort_path)

        # Load feature data
        feature_dfs = {}
        for category, path in feature_paths.items():
            if path is not None:
                feature_dfs[category] = pd.read_csv(path)

        # Merge features with cohort data
        X = cohort_df.copy()
        for category, df in feature_dfs.items():
            X = X.merge(df, on="patient_id", how="left")

        # Split features and target
        y = X[target_column]
        X = X.drop([target_column], axis=1)

        # Split sensitive attributes if provided
        sensitive = None
        if sensitive_columns:
            sensitive = X[sensitive_columns]
            X = X.drop(sensitive_columns, axis=1)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if sensitive is not None:
            sensitive_train = sensitive.iloc[X_train.index]
            sensitive_test = sensitive.iloc[X_test.index]
        else:
            sensitive_train = None
            sensitive_test = None

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test, sensitive_train, sensitive_test

    def train_model(self, X_train, y_train, model_name="default_model"):
        """
        Train a model on the prepared data.

        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            model_name (str): Name of the model

        Returns:
            tuple: (model, model_path)
        """
        if not MODEL_AVAILABLE:
            raise ImportError("Model modules not available")

        try:
            # Initialize model
            model = DL_models(
                input_size=X_train.shape[1],
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                model_type=self.model_type
            )

            # Train model
            model.train(
                X_train, y_train,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                num_epochs=self.num_epochs,
                device=self.device
            )

            # Save model
            model_path = os.path.join(self.output_dir, f"{model_name}.pt")
            torch.save(model.state_dict(), model_path)

            return model, model_path
        except Exception as e:
            raise Exception(f"Error training model: {str(e)}")

    def evaluate_model(self, model, X_test, y_test, sensitive_test=None):
        """
        Evaluate a trained model.

        Args:
            model (DL_models): Trained model
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            sensitive_test (pd.DataFrame): Test sensitive attributes

        Returns:
            dict: Dictionary of evaluation metrics
        """
        if not MODEL_AVAILABLE:
            raise ImportError("Model modules not available")

        try:
            # Initialize evaluator
            evaluator = Loss()

            # Get predictions
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test).to(self.device)
                y_pred = model(X_test_tensor)
                y_pred = y_pred.cpu().numpy()

            # Calculate metrics
            metrics = evaluator(y_test, y_pred)

            # Calculate fairness metrics if sensitive attributes are provided
            if sensitive_test is not None:
                fairness_metrics = {}
                for column in sensitive_test.columns:
                    fairness_metrics[column] = fairness_evaluation(
                        y_test, y_pred, sensitive_test[column]
                    )
                metrics["fairness"] = fairness_metrics

            return metrics
        except Exception as e:
            raise Exception(f"Error evaluating model: {str(e)}")

    def run_model_pipeline(self, feature_paths, cohort_path, target_column, sensitive_columns=None, model_name="default_model"):
        """
        Run the complete model training and evaluation pipeline.

        Args:
            feature_paths (dict): Dictionary of paths to feature files
            cohort_path (str): Path to cohort file
            target_column (str): Name of the target column
            sensitive_columns (list): List of sensitive attribute columns
            model_name (str): Name of the model

        Returns:
            dict: Dictionary of model and evaluation results
        """
        # Prepare data
        X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = self.prepare_data(
            feature_paths, cohort_path, target_column, sensitive_columns
        )

        # Train model
        model, model_path = self.train_model(X_train, y_train, model_name)

        # Evaluate model
        metrics = self.evaluate_model(model, X_test, y_test, sensitive_test)

        return {
            "model": model,
            "model_path": model_path,
            "metrics": metrics
        }
