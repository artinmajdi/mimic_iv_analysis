import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
try:
    from preprocessing_module import PreprocessingModule
    from model_module import ModelModule
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    print("Warning: Could not import project modules")


class PipelineStep(ABC):
    """Abstract base class for pipeline steps."""

    @abstractmethod
    def render(self):
        """Render the UI for this pipeline step."""
        pass

    @abstractmethod
    def validate(self):
        """Validate that prerequisites are met for this step."""
        pass


class DataPreprocessingStep(PipelineStep):
    """Handles the data preprocessing step of the pipeline."""

    def render(self):
        st.header("Data Preprocessing")

        if not self.validate():
            return

        # MIMIC-IV path
        mimic_path = st.text_input("MIMIC-IV Dataset Path", value="./data/mimic")

        # Output directory
        output_dir = st.text_input("Output Directory", value="./data")

        # Cohort settings
        st.subheader("Cohort Settings")
        col1, col2 = st.columns(2)
        with col1:
            cohort_name = st.text_input("Cohort Name", value="default_cohort")
            use_icu = st.checkbox("Use ICU Data", value=False)
        with col2:
            time_interval = st.number_input("Time Interval (hours)", value=24, min_value=1)
            disease_label = st.text_input("Disease Label (optional)", value="")

        # Feature settings
        st.subheader("Feature Settings")
        col1, col2 = st.columns(2)
        with col1:
            diagnoses = st.checkbox("Diagnoses", value=True)
            procedures = st.checkbox("Procedures", value=True)
        with col2:
            medications = st.checkbox("Medications", value=True)
            lab_tests = st.checkbox("Lab Tests", value=True)

        # Advanced settings
        with st.expander("Advanced Settings"):
            col1, col2 = st.columns(2)
            with col1:
                remove_outliers = st.checkbox("Remove Outliers", value=True)
                impute_missing = st.checkbox("Impute Missing Values", value=True)

        # Run preprocessing
        if st.button("Run Preprocessing"):
            try:
                # Initialize preprocessing module
                st.session_state.preprocessing_module = PreprocessingModule(
                    mimic_path=mimic_path,
                    output_dir=output_dir
                )

                # Set parameters
                st.session_state.preprocessing_module.use_icu = use_icu
                st.session_state.preprocessing_module.remove_outliers = remove_outliers
                st.session_state.preprocessing_module.impute_missing = impute_missing
                st.session_state.preprocessing_module.time_interval = time_interval
                st.session_state.preprocessing_module.disease_label = disease_label

                # Run pipeline
                with st.spinner("Running preprocessing pipeline..."):
                    feature_categories = {
                        "diagnoses": diagnoses,
                        "procedures": procedures,
                        "medications": medications,
                        "lab_tests": lab_tests
                    }

                    results = st.session_state.preprocessing_module.run_preprocessing_pipeline(
                        cohort_name=cohort_name,
                        feature_categories=feature_categories
                    )

                    st.success("Preprocessing completed successfully!")

                    # Display results
                    st.subheader("Output Files")
                    for category, paths in results.items():
                        st.write(f"**{category.title()}:**")
                        for name, path in paths.items():
                            if path is not None:
                                st.write(f"- {name}: {path}")
            except Exception as e:
                st.error(f"Error during preprocessing: {str(e)}")

    def validate(self):
        if not MODULES_AVAILABLE:
            st.error("Required modules not available. Please check the installation.")
            return False
        return True


class FeatureSelectionStep(PipelineStep):
    """Handles the feature selection step of the pipeline."""

    def render(self):
        st.header("Feature Selection")

        if not self.validate():
            return

        st.info("Feature selection will be implemented in the next version.")
        st.write("This feature will allow you to:")
        st.write("- Select specific features from each category")
        st.write("- Apply feature importance methods")
        st.write("- Generate feature importance plots")
        st.write("- Save selected feature sets")

    def validate(self):
        if not MODULES_AVAILABLE:
            st.error("Required modules not available. Please check the installation.")
            return False
        elif st.session_state.preprocessing_module is None:
            st.warning("Please run the preprocessing step first.")
            return False
        return True


class ModelTrainingStep(PipelineStep):
    """Handles the model training step of the pipeline."""

    def render(self):
        st.header("Model Training")

        if not self.validate():
            return

        # Model settings
        st.subheader("Model Settings")
        col1, col2 = st.columns(2)
        with col1:
            model_type = st.selectbox("Model Type", ["lstm", "gru", "transformer"])
            hidden_size = st.number_input("Hidden Size", value=128, min_value=1)
            num_layers = st.number_input("Number of Layers", value=2, min_value=1)
        with col2:
            dropout = st.slider("Dropout", value=0.2, min_value=0.0, max_value=1.0)
            batch_size = st.number_input("Batch Size", value=32, min_value=1)
            learning_rate = st.number_input("Learning Rate", value=0.001, format="%.3f")

        # Training settings
        st.subheader("Training Settings")
        col1, col2 = st.columns(2)
        with col1:
            num_epochs = st.number_input("Number of Epochs", value=10, min_value=1)
            target_column = st.text_input("Target Column", value="mortality")
        with col2:
            sensitive_columns = st.multiselect(
                "Sensitive Columns",
                options=["ethnicity", "gender", "age"],
                default=["ethnicity", "gender"]
            )

        # Run training
        if st.button("Train Model"):
            try:
                # Initialize model module
                st.session_state.model_module = ModelModule(
                    output_dir="./data/models"
                )

                # Set parameters
                st.session_state.model_module.model_type = model_type
                st.session_state.model_module.hidden_size = hidden_size
                st.session_state.model_module.num_layers = num_layers
                st.session_state.model_module.dropout = dropout
                st.session_state.model_module.batch_size = batch_size
                st.session_state.model_module.learning_rate = learning_rate
                st.session_state.model_module.num_epochs = num_epochs

                # Run pipeline
                with st.spinner("Training model..."):
                    results = st.session_state.model_module.run_model_pipeline(
                        feature_paths=st.session_state.preprocessing_module.feature_paths,
                        cohort_path=st.session_state.preprocessing_module.cohort_path,
                        target_column=target_column,
                        sensitive_columns=sensitive_columns,
                        model_name=f"{model_type}_{target_column}"
                    )

                    st.success("Model training completed successfully!")

                    # Display results
                    st.subheader("Model Metrics")
                    for metric, value in results["metrics"].items():
                        if metric != "fairness":
                            st.write(f"**{metric}:** {value:.4f}")

                    if "fairness" in results["metrics"]:
                        st.subheader("Fairness Metrics")
                        for attribute, metrics in results["metrics"]["fairness"].items():
                            st.write(f"**{attribute}:**")
                            for metric, value in metrics.items():
                                st.write(f"- {metric}: {value:.4f}")
            except Exception as e:
                st.error(f"Error during model training: {str(e)}")

    def validate(self):
        if not MODULES_AVAILABLE:
            st.error("Required modules not available. Please check the installation.")
            return False
        elif st.session_state.preprocessing_module is None:
            st.warning("Please run the preprocessing step first.")
            return False
        return True


class EvaluationStep(PipelineStep):
    """Handles the model evaluation step of the pipeline."""

    def render(self):
        st.header("Model Evaluation")

        if not self.validate():
            return

        st.info("Evaluation metrics will be displayed here.")
        st.write("This section will show:")
        st.write("- Model performance metrics")
        st.write("- ROC curves")
        st.write("- Precision-recall curves")
        st.write("- Confusion matrices")

    def validate(self):
        if not MODULES_AVAILABLE:
            st.error("Required modules not available. Please check the installation.")
            return False
        elif st.session_state.model_module is None:
            st.warning("Please train a model first.")
            return False
        return True


class FairnessAnalysisStep(PipelineStep):
    """Handles the fairness analysis step of the pipeline."""

    def render(self):
        st.header("Fairness Analysis")

        if not self.validate():
            return

        st.info("Fairness analysis will be displayed here.")
        st.write("This section will show:")
        st.write("- Demographic parity")
        st.write("- Equal opportunity")
        st.write("- Equalized odds")
        st.write("- Treatment equality")

    def validate(self):
        if not MODULES_AVAILABLE:
            st.error("Required modules not available. Please check the installation.")
            return False
        elif st.session_state.model_module is None:
            st.warning("Please train a model first.")
            return False
        return True


class StreamlitApp:
    """Main application class for the MIMIC-IV Data Pipeline."""

    def __init__(self):
        """Initialize the application."""
        self.steps = {
            "Data Preprocessing": DataPreprocessingStep(),
            "Feature Selection": FeatureSelectionStep(),
            "Model Training": ModelTrainingStep(),
            "Evaluation": EvaluationStep(),
            "Fairness Analysis": FairnessAnalysisStep()
        }

        # Initialize session state
        if "preprocessing_module" not in st.session_state:
            st.session_state.preprocessing_module = None
        if "model_module" not in st.session_state:
            st.session_state.model_module = None
        if "current_step" not in st.session_state:
            st.session_state.current_step = "Data Preprocessing"

    def render_header(self):
        """Render the application header."""
        st.title("MIMIC-IV Data Pipeline")
        st.markdown("---")

    def render_main_content(self):
        """Render the main content with tabs for navigation."""
        # Create tabs for each step
        tabs = st.tabs(list(self.steps.keys()))

        # Render the content for each tab
        for i, (step_name, step) in enumerate(self.steps.items()):
            with tabs[i]:
                step.render()

    def render_footer(self):
        """Render the footer."""
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center'>
            <p>MIMIC-IV Data Pipeline Dashboard | Created with ❤️ using Streamlit</p>
        </div>
        """, unsafe_allow_html=True)

    def run(self):
        """Run the application."""
        # Set page config
        st.set_page_config(
            page_title="MIMIC-IV Data Pipeline",
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="collapsed"
        )

        # Render the application
        self.render_header()
        self.render_main_content()
        self.render_footer()


# Run the application
if __name__ == "__main__":
    app = StreamlitApp()
    app.run()
