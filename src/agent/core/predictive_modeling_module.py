import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import calendar
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report
from sklearn.calibration import calibration_curve
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import shap


class PredictiveModeling:
    """Class for predictive modeling of discharge in MIMIC-IV dataset."""

    def __init__(self, data_loader):
        """Initialize the predictive modeling module with a data loader.

        Args:
            data_loader (MIMICDataLoader): Data loader object with preprocessed data
        """
        self.data_loader = data_loader
        self.data = data_loader.preprocessed
        self.models = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.features = None
        self.target = None
        self.preprocessor = None

    def feature_selection(self):
        """Display feature selection interface for discharge prediction models."""
        if 'admissions' not in self.data or self.data['admissions'] is None:
            st.warning("Admissions data not available for feature selection.")
            return

        if 'patients' not in self.data or self.data['patients'] is None:
            st.warning("Patient data not available for feature selection.")
            return

        st.subheader("Feature Selection for Discharge Prediction Models")

        # Define feature categories
        st.write("#### Feature Categories")

        col1, col2 = st.columns(2)

        with col1:
            # Demographics features
            st.write("**Demographics**")
            demo_features = []

            if 'patients' in self.data and self.data['patients'] is not None:
                patients_df = self.data['patients']
                available_demo = []

                if 'gender' in patients_df.columns:
                    available_demo.append("gender")
                if 'age' in patients_df.columns:
                    available_demo.append("age")
                if 'race' in patients_df.columns:
                    available_demo.append("race")

                demo_features = st.multiselect(
                    "Select demographic features",
                    options=available_demo,
                    default=available_demo
                )

            # Admission features
            st.write("**Admission Characteristics**")
            adm_features = []

            if 'admissions' in self.data and self.data['admissions'] is not None:
                admissions_df = self.data['admissions']
                available_adm = []

                if 'admission_type' in admissions_df.columns:
                    available_adm.append("admission_type")
                if 'admission_location' in admissions_df.columns:
                    available_adm.append("admission_location")
                if 'insurance' in admissions_df.columns:
                    available_adm.append("insurance")
                if 'language' in admissions_df.columns:
                    available_adm.append("language")
                if 'marital_status' in admissions_df.columns:
                    available_adm.append("marital_status")

                adm_features = st.multiselect(
                    "Select admission features",
                    options=available_adm,
                    default=available_adm[:3] if len(available_adm) >= 3 else available_adm
                )

        with col2:
            # Transfer/movement features
            st.write("**Hospital Transfers**")
            transfer_features = []

            if 'transfers' in self.data and self.data['transfers'] is not None:
                transfers_df = self.data['transfers']
                available_transfer = []

                if 'careunit' in transfers_df.columns:
                    available_transfer.append("first_careunit")
                    available_transfer.append("last_careunit")
                if 'unit_los_hours' in transfers_df.columns:
                    available_transfer.append("transfers_count")
                    available_transfer.append("icu_stay")

                transfer_features = st.multiselect(
                    "Select transfer features",
                    options=available_transfer,
                    default=available_transfer
                )

            # Order features
            st.write("**Provider Orders**")
            order_features = []

            if 'poe' in self.data and self.data['poe'] is not None:
                poe_df = self.data['poe']
                available_order = []

                if 'order_type' in poe_df.columns:
                    available_order.append("medication_count")
                    available_order.append("lab_count")
                    available_order.append("imaging_count")
                    available_order.append("procedure_count")

                order_features = st.multiselect(
                    "Select order features",
                    options=available_order,
                    default=available_order
                )

        # Target variable definition
        st.write("#### Target Variable Definition")

        target_options = [
            "24h_discharge_readiness",
            "48h_discharge_readiness",
            "72h_discharge_readiness",
            "early_discharge",
            "extended_stay"
        ]

        target_variable = st.selectbox(
            "Select target variable",
            options=target_options,
            index=1,
            help="Define what you want to predict about discharge"
        )

        # Combine all selected features
        all_selected_features = demo_features + adm_features + transfer_features + order_features

        if len(all_selected_features) > 0:
            st.success(f"Selected {len(all_selected_features)} features for modeling")

            # Store selected features and target for use in other methods
            self.features = all_selected_features
            self.target = target_variable

            # Display feature engineering approach
            with st.expander("Feature Engineering Details"):
                st.write("""
                **Feature Engineering Process:**

                1. **Demographics**: Patient characteristics like age, gender, and ethnicity
                2. **Admission Details**: Type of admission, insurance, and entry point
                3. **Hospital Transfers**: Care unit history and movement patterns
                4. **Provider Orders**: Medication, lab, imaging, and procedure orders

                **Preprocessing Steps:**
                - Categorical features will be one-hot encoded
                - Numerical features will be standardized
                - Missing values will be imputed using appropriate strategies
                - Temporal features will be aggregated to patient-admission level
                """)

                # Show example of engineered features
                st.write("**Example of Engineered Features:**")

                # Create sample data for demonstration
                np.random.seed(42)
                n_samples = 5

                sample_data = {
                    "subject_id": np.random.randint(10000, 99999, n_samples),
                    "hadm_id": np.random.randint(1000000, 9999999, n_samples),
                    "age": np.random.normal(65, 15, n_samples).round().astype(int).clip(18, 90),
                    "gender": np.random.choice(["M", "F"], n_samples),
                    "admission_type": np.random.choice(["EMERGENCY", "ELECTIVE", "URGENT"], n_samples),
                    "first_careunit": np.random.choice(["MICU", "SICU", "ED", "MED"], n_samples),
                    "transfers_count": np.random.randint(1, 6, n_samples),
                    "medication_count": np.random.randint(3, 15, n_samples),
                    "lab_count": np.random.randint(5, 20, n_samples),
                    "48h_discharge_readiness": np.random.choice([0, 1], n_samples)
                }

                sample_df = pd.DataFrame(sample_data)
                st.dataframe(sample_df)
        else:
            st.warning("Please select at least one feature for modeling")

    def model_training(self):
        """Display model training interface for discharge prediction."""
        if self.features is None or self.target is None:
            st.warning("Please select features and target variable in the Feature Selection tab first.")
            return

        st.subheader("Model Training for Discharge Prediction")

        # Model selection
        st.write("#### Select Model")

        model_type = st.selectbox(
            "Model Type",
            options=["Random Forest", "XGBoost", "LSTM Neural Network"],
            index=0
        )

        # Training parameters
        st.write("#### Training Parameters")

        col1, col2 = st.columns(2)

        with col1:
            # Common parameters
            test_size = st.slider(
                "Test Set Size (%)",
                min_value=10,
                max_value=40,
                value=20,
                help="Percentage of data to use for testing"
            ) / 100

            random_state = st.number_input(
                "Random Seed",
                min_value=1,
                max_value=1000,
                value=42,
                help="Random seed for reproducibility"
            )

        with col2:
            # Model-specific parameters
            if model_type == "Random Forest":
                n_estimators = st.slider(
                    "Number of Trees",
                    min_value=50,
                    max_value=500,
                    value=100,
                    step=50
                )

                max_depth = st.slider(
                    "Maximum Tree Depth",
                    min_value=3,
                    max_value=20,
                    value=10
                )

                model_params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "random_state": random_state
                }

            elif model_type == "XGBoost":
                learning_rate = st.slider(
                    "Learning Rate",
                    min_value=0.01,
                    max_value=0.3,
                    value=0.1,
                    step=0.01
                )

                n_estimators = st.slider(
                    "Number of Boosting Rounds",
                    min_value=50,
                    max_value=500,
                    value=100,
                    step=50
                )

                max_depth = st.slider(
                    "Maximum Tree Depth",
                    min_value=3,
                    max_value=10,
                    value=6
                )

                model_params = {
                    "learning_rate": learning_rate,
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "random_state": random_state
                }

            elif model_type == "LSTM Neural Network":
                units = st.slider(
                    "LSTM Units",
                    min_value=32,
                    max_value=256,
                    value=128,
                    step=32
                )

                dropout = st.slider(
                    "Dropout Rate",
                    min_value=0.1,
                    max_value=0.5,
                    value=0.2,
                    step=0.1
                )

                epochs = st.slider(
                    "Training Epochs",
                    min_value=10,
                    max_value=100,
                    value=50,
                    step=10
                )

                model_params = {
                    "units": units,
                    "dropout": dropout,
                    "epochs": epochs
                }

        # Training button
        st.write("#### Train Model")

        if st.button("Train Model", key="train_model"):
            # This would be implemented with actual data
            # For now, we'll use synthetic data for demonstration

            with st.spinner(f"Training {model_type} model..."):
                # Create synthetic dataset
                np.random.seed(int(random_state))
                n_samples = 1000
                n_features = len(self.features)

                # Generate synthetic features and target
                X = np.random.randn(n_samples, n_features)
                y = (np.random.randn(n_samples) > 0).astype(int)  # Binary classification

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=int(random_state)
                )

                # Store for later use
                self.X_train = X_train
                self.X_test = X_test
                self.y_train = y_train
                self.y_test = y_test

                # Train model based on selection
                if model_type == "Random Forest":
                    model = RandomForestClassifier(**model_params)
                    model.fit(X_train, y_train)

                    # Make predictions
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    y_pred = model.predict(X_test)

                    # Store model
                    self.models["current"] = {
                        "type": "Random Forest",
                        "model": model,
                        "y_test": y_test,
                        "y_pred": y_pred,
                        "y_pred_proba": y_pred_proba,
                        "feature_importance": model.feature_importances_
                    }

                elif model_type == "XGBoost":
                    model = xgb.XGBClassifier(**model_params)
                    model.fit(X_train, y_train)

                    # Make predictions
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    y_pred = model.predict(X_test)

                    # Store model
                    self.models["current"] = {
                        "type": "XGBoost",
                        "model": model,
                        "y_test": y_test,
                        "y_pred": y_pred,
                        "y_pred_proba": y_pred_proba,
                        "feature_importance": model.feature_importances_
                    }

                elif model_type == "LSTM Neural Network":
                    # For LSTM, we would reshape the data and use sequences
                    # This is a simplified version for demonstration

                    # Build a simple LSTM model
                    model = Sequential([
                        LSTM(units=model_params["units"], input_shape=(X_train.shape[1], 1), return_sequences=True),
                        Dropout(model_params["dropout"]),
                        LSTM(units=model_params["units"]//2),
                        Dropout(model_params["dropout"]),
                        Dense(1, activation='sigmoid')
                    ])

                    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

                    # Reshape data for LSTM (samples, timesteps, features)
                    X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

                    # Train model
                    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

                    history = model.fit(
                        X_train_reshaped, y_train,
                        epochs=model_params["epochs"],
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=[early_stopping],
                        verbose=0
                    )

                    # Make predictions
                    y_pred_proba = model.predict(X_test_reshaped).flatten()
                    y_pred = (y_pred_proba > 0.5).astype(int)

                    # Store model
                    self.models["current"] = {
                        "type": "LSTM",
                        "model": model,
                        "y_test": y_test,
                        "y_pred": y_pred,
                        "y_pred_proba": y_pred_proba,
                        "history": history.history
                    }

                st.success(f"{model_type} model trained successfully!")

                # Display basic performance metrics
                accuracy = (y_pred == y_test).mean()
                st.metric("Test Accuracy", f"{accuracy:.2%}")

    def performance_visualization(self):
        """Display performance visualization for trained models."""
        if "current" not in self.models:
            st.warning("Please train a model in the Model Training tab first.")
            return

        st.subheader("Model Performance Visualization")

        # Get current model data
        model_data = self.models["current"]
        model_type = model_data["type"]
        y_test = model_data["y_test"]
        y_pred = model_data["y_pred"]
        y_pred_proba = model_data["y_pred_proba"]

        # ROC Curve
        st.write("#### ROC Curve")

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        fig = go.Figure()

        # Add ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{model_type} (AUC = {roc_auc:.3f})',
            line=dict(color='royalblue', width=2)
        ))

        # Add diagonal reference line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Reference',
            line=dict(color='gray', width=1, dash='dash')
        ))

        fig.update_layout(
            title=f'ROC Curve for {model_type}',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=700,
            height=500,
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)')
        )

        st.plotly_chart(fig, use_container_width=True)

        # Precision-Recall Curve
        st.write("#### Precision-Recall Curve")

        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)

        fig = go.Figure()

        # Add PR curve
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'{model_type} (AUC = {pr_auc:.3f})',
            line=dict(color='forestgreen', width=2)
        ))

        # Add baseline
        baseline = sum(y_test) / len(y_test)
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[baseline, baseline],
            mode='lines',
            name=f'Baseline ({baseline:.3f})',
            line=dict(color='gray', width=1, dash='dash')
        ))

        fig.update_layout(
            title=f'Precision-Recall Curve for {model_type}',
            xaxis_title='Recall',
            yaxis_title='Precision',
            yaxis=dict(range=[0, 1.05]),
            xaxis=dict(range=[0, 1.05]),
            width=700,
            height=500,
            legend=dict(x=0.01, y=0.01, bgcolor='rgba(255, 255, 255, 0.5)')
        )

        st.plotly_chart(fig, use_container_width=True)

        # Calibration Plot
        st.write("#### Calibration Plot")

        prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)

        fig = go.Figure()

        # Add calibration curve
        fig.add_trace(go.Scatter(
            x=prob_pred, y=prob_true,
            mode='lines+markers',
            name=f'{model_type}',
            line=dict(color='darkorange', width=2),
            marker=dict(size=8)
        ))

        # Add perfect calibration reference
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Perfectly Calibrated',
            line=dict(color='gray', width=1, dash='dash')
        ))

        fig.update_layout(
            title=f'Calibration Plot for {model_type}',
            xaxis_title='Mean Predicted Probability',
            yaxis_title='Fraction of Positives',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=700,
            height=500,
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)')
        )

        st.plotly_chart(fig, use_container_width=True)

        # Confusion Matrix
        st.write("#### Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        # Create heatmap
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted Label", y="True Label", color="Count"),
            x=['Negative', 'Positive'],
            y=['Negative', 'Positive'],
            text_auto=True,
            color_continuous_scale='Blues'
        )

        fig.update_layout(
            title=f'Confusion Matrix for {model_type}',
            width=600,
            height=500
        )

        col1, col2 = st.columns([2, 3])

        with col1:
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.write("**Performance Metrics:**")

            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Sensitivity (Recall)', 'Specificity', 'Precision (PPV)', 'NPV'],
                'Value': [f"{accuracy:.3f}", f"{sensitivity:.3f}", f"{specificity:.3f}", f"{ppv:.3f}", f"{npv:.3f}"]
            })

            st.dataframe(metrics_df, hide_index=True)

            st.write("**Classification Report:**")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.3f}"))

    def feature_importance(self):
        """Display feature importance visualization."""
        if "current" not in self.models:
            st.warning("Please train a model in the Model Training tab first.")
            return

        st.subheader("Feature Importance Visualization")

        # Get current model data
        model_data = self.models["current"]
        model_type = model_data["type"]
        model = model_data["model"]

        # Feature importance
        st.write("#### Feature Importance")

        if model_type in ["Random Forest", "XGBoost"]:
            # Get feature importance
            feature_importance = model_data["feature_importance"]

            # Create feature names (synthetic for demonstration)
            feature_names = self.features if len(self.features) == len(feature_importance) else [f"Feature {i+1}" for i in range(len(feature_importance))]

            # Create dataframe
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            })

            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)

            # Create bar chart
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title=f"Feature Importance for {model_type}",
                color='Importance',
                color_continuous_scale='Viridis'
            )

            fig.update_layout(
                yaxis={'categoryorder':'total ascending'},
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # SHAP values (simplified for demonstration)
            st.write("#### SHAP Values")

            st.info("""
            SHAP (SHapley Additive exPlanations) values show the contribution of each feature to the prediction.
            This would be implemented with actual model and data using the SHAP library.
            """)

            # Example SHAP visualization
            st.image("https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/shap_header.png",
                    caption="Example SHAP summary plot (source: SHAP library)", use_column_width=True)

        elif model_type == "LSTM":
            st.info("""
            Feature importance for LSTM models requires specialized techniques like:
            - Integrated Gradients
            - Occlusion sensitivity
            - Attention mechanisms

            This would be implemented with actual model and data.
            """)

            # Example attention visualization
            st.image("https://miro.medium.com/max/1400/1*3bPlAJdSj1VD0Dtz9VrYcA.png",
                    caption="Example attention visualization for sequence data (source: Medium)", use_column_width=True)

        # Partial Dependence Plots
        st.write("#### Partial Dependence Plots")

        st.info("""
        Partial Dependence Plots show the marginal effect of a feature on the predicted outcome.
        This would be implemented with actual model and data.
        """)

        # Example PDP visualization
        cols = st.columns(2)

        with cols[0]:
            # Create synthetic PDP for age
            np.random.seed(42)
            x = np.linspace(20, 90, 100)
            y = 0.2 + 0.4 * np.exp(-(x - 60)**2 / 400) + 0.05 * np.random.randn(100)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                name='Age',
                line=dict(color='royalblue', width=2)
            ))

            fig.update_layout(
                title='Partial Dependence Plot: Age',
                xaxis_title='Age',
                yaxis_title='Marginal Effect on Prediction',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            st.write("""This plot shows how the predicted probability of discharge readiness changes with patient age,
                    holding all other features constant. The peak around age 60 suggests that middle-aged patients
                    may have the highest likelihood of discharge readiness.""")

        with cols[1]:
            # Create synthetic PDP for length of stay
            x = np.linspace(0, 14, 100)
            y = 0.8 - 0.6 * np.exp(-(x - 1)**2 / 10) + 0.05 * np.random.randn(100)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                name='Length of Stay',
                line=dict(color='forestgreen', width=2)
            ))

            fig.update_layout(
                title='Partial Dependence Plot: Length of Stay',
                xaxis_title='Length of Stay (days)',
                yaxis_title='Marginal Effect on Prediction',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            st.write("""This plot shows how the predicted probability of discharge readiness changes with length of stay,
                    holding all other features constant. The curve suggests that discharge readiness increases with
                    longer stays but plateaus after approximately 7 days.""")

    def model_explanation(self):
        """Display model explanation and interpretability analysis."""
        if "current" not in self.models:
            st.warning("Please train a model in the Model Training tab first.")
            return

        st.subheader("Model Explanation and Interpretability")

        # Get current model data
        model_data = self.models["current"]
        model_type = model_data["type"]
        model = model_data["model"]

        # Explanation methods
        st.write("#### Explanation Methods")

        explanation_type = st.radio(
            "Select explanation method",
            options=["SHAP Values", "Feature Interactions", "What-If Analysis"],
            index=0
        )

        if explanation_type == "SHAP Values":
            st.write("**SHAP (SHapley Additive exPlanations) Analysis**")

            st.info("""
            SHAP values help understand the contribution of each feature to individual predictions.
            This would be implemented with actual model and data using the SHAP library.
            """)

            # Example SHAP visualizations
            cols = st.columns(2)

            with cols[0]:
                st.image("https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/shap_waterfall.png",
                        caption="Example SHAP waterfall plot for a single prediction", use_column_width=True)

            with cols[1]:
                st.image("https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/shap_beeswarm.png",
                        caption="Example SHAP beeswarm plot showing feature impact", use_column_width=True)

            st.write("""
            **Interpretation:**

            SHAP values show how each feature contributes to pushing the model output from the base value
            (average prediction) to the actual prediction for a specific instance. Features pushing the prediction
            higher are shown in red, while those pushing it lower are in blue.

            In a discharge prediction model, this could reveal that:
            - High age might push predictions toward extended stays
            - Certain admission types might strongly indicate early discharge
            - Specific lab values might be critical indicators of discharge readiness
            """)

        elif explanation_type == "Feature Interactions":
            st.write("**Feature Interaction Analysis**")

            st.info("""
            Feature interactions reveal how combinations of features affect predictions beyond their individual effects.
            This would be implemented with actual model and data.
            """)

            # Example interaction plot
            st.image("https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/shap_interaction_waterfall.png",
                    caption="Example feature interaction visualization", use_column_width=True)

            st.write("""
            **Interpretation:**

            Feature interactions can reveal important clinical insights, such as:
            - How age and comorbidity count might interact (older patients with multiple comorbidities may have
              significantly longer stays than would be predicted by either factor alone)
            - How admission type and time of day might interact (emergency admissions at night might have
              different discharge patterns)
            - How lab values and medication counts might interact (patients with abnormal labs and multiple
              medications might require more complex discharge planning)
            """)

        elif explanation_type == "What-If Analysis":
            st.write("**What-If Analysis Tool**")

            st.info("""
            What-If Analysis allows exploration of how predictions would change if feature values were different.
            This would be implemented with actual model and data.
            """)

            # Example what-if interface
            st.write("**Example What-If Scenario:**")

            cols = st.columns(2)

            with cols[0]:
                st.write("**Patient Characteristics:**")
                st.write("- Age: 65")
                st.write("- Gender: Male")
                st.write("- Admission Type: Emergency")
                st.write("- First Care Unit: MICU")
                st.write("- Length of Stay: 4.2 days")
                st.write("- Lab Count: 12")
                st.write("- Medication Count: 8")

                st.write("**Base Prediction:**")
                st.metric("48h Discharge Readiness", "62%")

            with cols[1]:
                st.write("**What-If Scenarios:**")

                scenario1, scenario2 = st.columns(2)

                with scenario1:
                    st.write("**If Length of Stay → 6.5 days:**")
                    st.metric("48h Discharge Readiness", "78%", delta="+16%")

                with scenario2:
                    st.write("**If Medication Count → 5:**")
                    st.metric("48h Discharge Readiness", "71%", delta="+9%")

                st.write("**If Care Unit → Step-Down Unit:**")
                st.metric("48h Discharge Readiness", "85%", delta="+23%")

            st.write("""
            **Interpretation:**

            What-If analysis allows clinicians to explore how different care decisions might affect discharge readiness.
            In this example, we can see that:
            - Extending the length of stay would increase discharge readiness probability
            - Reducing medication complexity would moderately improve discharge readiness
            - Transferring to a step-down unit would significantly increase discharge readiness

            These insights can guide clinical decision-making and resource allocation.
            """)

    def deployment_considerations(self):
        """Display model deployment considerations and implementation guidance."""
        st.subheader("Model Deployment Considerations")

        st.write("""
        #### Implementation in Clinical Workflow

        Successful implementation of discharge prediction models requires careful integration into clinical workflows:

        1. **Integration Points**
           - EHR system integration via API
           - Daily patient list with discharge probability scores
           - Automated alerts for patients with high discharge readiness
           - Mobile applications for care team coordination

        2. **User Interface Design**
           - Simple, intuitive presentation of predictions
           - Clear explanation of contributing factors
           - Actionable recommendations based on predictions
           - Feedback mechanisms for clinicians

        3. **Clinical Decision Support**
           - Highlight patients ready for discharge planning
           - Identify potential barriers to discharge
           - Suggest interventions to address barriers
           - Track discharge planning progress
        """)

        # Example mockup of clinical implementation
        st.image("https://www.researchgate.net/publication/340002972/figure/fig1/AS:868599425961985@1584143451044/Clinical-decision-support-system-CDSS-embedded-in-the-electronic-health-record-EHR.png",
                caption="Example of clinical decision support integration (source: ResearchGate)", use_column_width=True)

        st.write("""
        #### Ethical and Regulatory Considerations

        Deployment of predictive models in healthcare requires addressing several important considerations:

        1. **Fairness and Bias**
           - Regular audits for demographic bias
           - Balanced training data across populations
           - Monitoring for disparate impact
           - Transparency in model limitations

        2. **Privacy and Security**
           - HIPAA compliance for all data handling
           - Secure API connections for model serving
           - Audit trails for prediction access
           - Data minimization principles

        3. **Regulatory Approval**
           - FDA considerations for Clinical Decision Support
           - Documentation of validation studies
           - Quality management system
           - Post-deployment monitoring plan
        """)

        # Example validation framework
        validation_data = {
            'Phase': ['Internal Validation', 'Prospective Validation', 'External Validation', 'Clinical Impact Study'],
            'Description': [
                'Retrospective validation on hold-out data from same institution',
                'Prospective validation in limited clinical settings',
                'Validation on data from different institutions',
                'Randomized study measuring impact on length of stay and readmissions'
            ],
            'Status': ['Complete', 'In Progress', 'Planned', 'Planned']
        }

        validation_df = pd.DataFrame(validation_data)
        st.write("**Model Validation Framework:**")
        st.dataframe(validation_df, hide_index=True)

        st.write("""
        #### Performance Monitoring

        Continuous monitoring ensures the model remains accurate and valuable over time:

        1. **Key Metrics to Track**
           - Prediction accuracy vs. actual discharges
           - False positive/negative rates
           - Calibration drift over time
           - User adoption and feedback

        2. **Model Updating Strategy**
           - Scheduled retraining (quarterly)
           - Trigger-based retraining (performance drops)
           - A/B testing for model improvements
           - Version control and rollback capability
        """)

        # Example monitoring dashboard mockup
        cols = st.columns(2)

        with cols[0]:
            st.metric("Current Model AUC", "0.82", delta="-0.01", delta_color="inverse")
            st.metric("False Negative Rate", "8.3%", delta="+0.5%", delta_color="inverse")

        with cols[1]:
            st.metric("User Adoption", "76%", delta="+12%")
            st.metric("Avg. Time Saved", "22 min/patient", delta="+3 min")
