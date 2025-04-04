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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import networkx as nx


class PatientTrajectoryAnalysis:
    """Class for analyzing patient trajectories in MIMIC-IV dataset."""

    def __init__(self, data_loader):
        """Initialize the patient trajectory analysis module with a data loader.

        Args:
            data_loader (MIMICDataLoader): Data loader object with preprocessed data
        """
        self.data_loader = data_loader
        self.data = data_loader.preprocessed

    def patient_journey(self):
        """Display interactive patient journey visualization."""
        if 'transfers' not in self.data or self.data['transfers'] is None:
            st.warning("Transfers data not available for patient journey visualization.")
            return

        if 'admissions' not in self.data or self.data['admissions'] is None:
            st.warning("Admissions data not available for patient journey visualization.")
            return

        if 'patients' not in self.data or self.data['patients'] is None:
            st.warning("Patients data not available for patient journey visualization.")
            return

        transfers_df = self.data['transfers']
        admissions_df = self.data['admissions']
        patients_df = self.data['patients']

        st.subheader("Interactive Patient Journey Visualization")

        # Patient selector
        st.write("#### Select Patient")

        # Get list of patients with transfers
        patient_ids = sorted(transfers_df['subject_id'].unique())

        if len(patient_ids) > 100:
            # If there are many patients, allow searching by ID
            patient_id_input = st.text_input("Enter Patient ID (subject_id)")

            if patient_id_input:
                try:
                    patient_id = int(patient_id_input)
                    if patient_id in patient_ids:
                        selected_patient_id = patient_id
                    else:
                        st.warning(f"Patient ID {patient_id} not found in transfers data.")
                        return
                except ValueError:
                    st.warning("Please enter a valid numeric Patient ID.")
                    return
            else:
                # If no input, select a random patient
                selected_patient_id = patient_ids[0]
        else:
            # If there are few patients, use a selectbox
            selected_patient_id = st.selectbox(
                "Select Patient ID",
                options=patient_ids
            )

        # Get patient information
        patient_info = patients_df[patients_df['subject_id'] == selected_patient_id].iloc[0]

        # Display patient information
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Patient ID", selected_patient_id)

        with col2:
            if 'gender' in patient_info:
                st.metric("Gender", patient_info['gender'])

        with col3:
            if 'age' in patient_info:
                st.metric("Age", int(patient_info['age']))

        # Get patient admissions
        patient_admissions = admissions_df[admissions_df['subject_id'] == selected_patient_id]

        # Allow selecting an admission if multiple exist
        if len(patient_admissions) > 1:
            st.write("#### Select Admission")

            # Format admission options with dates
            admission_options = []
            for _, row in patient_admissions.iterrows():
                admit_date = row['admittime'].strftime('%Y-%m-%d') if 'admittime' in row else 'Unknown'
                discharge_date = row['dischtime'].strftime('%Y-%m-%d') if 'dischtime' in row and pd.notna(row['dischtime']) else 'Unknown'
                admission_type = row['admission_type'] if 'admission_type' in row else 'Unknown'
                option_text = f"ID: {row['hadm_id']} | {admit_date} to {discharge_date} | {admission_type}"
                admission_options.append((row['hadm_id'], option_text))

            selected_admission_id = st.selectbox(
                "Select Admission",
                options=[opt[0] for opt in admission_options],
                format_func=lambda x: next(opt[1] for opt in admission_options if opt[0] == x)
            )
        elif len(patient_admissions) == 1:
            selected_admission_id = patient_admissions.iloc[0]['hadm_id']
        else:
            st.warning("No admissions found for this patient.")
            return

        # Get transfers for selected admission
        patient_transfers = transfers_df[
            (transfers_df['subject_id'] == selected_patient_id) &
            (transfers_df['hadm_id'] == selected_admission_id)
        ].sort_values('intime')

        if len(patient_transfers) == 0:
            st.warning("No transfers found for this admission.")
            return

        # Get admission details
        admission_details = patient_admissions[patient_admissions['hadm_id'] == selected_admission_id].iloc[0]

        # Display admission information
        col1, col2, col3 = st.columns(3)

        with col1:
            if 'admission_type' in admission_details:
                st.metric("Admission Type", admission_details['admission_type'])

        with col2:
            if 'los_days' in admission_details:
                st.metric("Length of Stay (days)", f"{admission_details['los_days']:.1f}")

        with col3:
            if 'discharge_location' in admission_details:
                st.metric("Discharge Location", admission_details['discharge_location'])

        # Journey visualization
        st.write("#### Patient Journey")

        # Prepare data for timeline
        timeline_data = []

        for _, transfer in patient_transfers.iterrows():
            careunit = transfer['careunit'] if 'careunit' in transfer else 'Unknown'
            intime = transfer['intime'] if 'intime' in transfer else None
            outtime = transfer['outtime'] if 'outtime' in transfer else None

            if intime is not None and outtime is not None:
                # Calculate duration in hours
                duration_hours = (outtime - intime).total_seconds() / 3600

                timeline_data.append({
                    'Care Unit': careunit,
                    'Start Time': intime,
                    'End Time': outtime,
                    'Duration (hours)': duration_hours
                })

        # Create timeline dataframe
        timeline_df = pd.DataFrame(timeline_data)

        if len(timeline_df) > 0:
            # Calculate relative time from admission start
            admission_start = timeline_df['Start Time'].min()
            timeline_df['Hours from Admission'] = (timeline_df['Start Time'] - admission_start).dt.total_seconds() / 3600
            timeline_df['End Hours from Admission'] = (timeline_df['End Time'] - admission_start).dt.total_seconds() / 3600

            # Create Gantt chart
            fig = px.timeline(
                timeline_df,
                x_start="Hours from Admission",
                x_end="End Hours from Admission",
                y="Care Unit",
                color="Care Unit",
                title="Patient Journey Timeline",
                labels={"Hours from Admission": "Hours from Admission Start"}
            )

            fig.update_yaxes(autorange="reversed")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Add detailed transfer information
            st.write("#### Transfer Details")

            # Format transfer details table
            transfer_details = patient_transfers.copy()

            # Format datetime columns
            datetime_cols = ['intime', 'outtime']
            for col in datetime_cols:
                if col in transfer_details.columns:
                    transfer_details[col] = transfer_details[col].dt.strftime('%Y-%m-%d %H:%M')

            # Select columns to display
            display_cols = ['careunit', 'intime', 'outtime']
            if 'unit_los_hours' in transfer_details.columns:
                display_cols.append('unit_los_hours')

            # Display only selected columns
            display_df = transfer_details[display_cols].copy()

            # Rename columns for display
            display_df.columns = [col.replace('_', ' ').title() for col in display_cols]

            # Format LOS hours if present
            if 'Unit Los Hours' in display_df.columns:
                display_df['Unit Los Hours'] = display_df['Unit Los Hours'].round(1)
                display_df = display_df.rename(columns={'Unit Los Hours': 'LOS (hours)'})

            st.dataframe(display_df)
        else:
            st.warning("No timeline data available for this patient journey.")

        # If POE data is available, show orders during the admission
        if 'poe' in self.data and self.data['poe'] is not None:
            poe_df = self.data['poe']

            patient_orders = poe_df[
                (poe_df['subject_id'] == selected_patient_id) &
                (poe_df['hadm_id'] == selected_admission_id)
            ].sort_values('ordertime')

            if len(patient_orders) > 0:
                st.write("#### Orders During Admission")

                # Group orders by type
                if 'order_type' in patient_orders.columns:
                    order_counts = patient_orders['order_type'].value_counts().reset_index()
                    order_counts.columns = ['Order Type', 'Count']

                    # Create bar chart
                    fig = px.bar(
                        order_counts,
                        x='Order Type',
                        y='Count',
                        color='Order Type',
                        title="Orders by Type"
                    )
                    fig.update_layout(
                        xaxis_title='Order Type',
                        yaxis_title='Count',
                        xaxis={'categoryorder':'total descending'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Order timeline
                if 'ordertime' in patient_orders.columns and 'order_type' in patient_orders.columns:
                    # Calculate hours from admission
                    patient_orders['hours_from_admission'] = (patient_orders['ordertime'] - admission_start).dt.total_seconds() / 3600

                    # Create scatter plot
                    fig = px.scatter(
                        patient_orders,
                        x='hours_from_admission',
                        y='order_type',
                        color='order_type',
                        title="Order Timeline",
                        labels={
                            'hours_from_admission': 'Hours from Admission Start',
                            'order_type': 'Order Type'
                        }
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

    def pathway_clustering(self):
        """Display clustering of similar patient pathways."""
        if 'transfers' not in self.data or self.data['transfers'] is None:
            st.warning("Transfers data not available for pathway clustering.")
            return

        if 'admissions' not in self.data or self.data['admissions'] is None:
            st.warning("Admissions data not available for pathway clustering.")
            return

        transfers_df = self.data['transfers']
        admissions_df = self.data['admissions']

        st.subheader("Clustering of Similar Patient Pathways")

        # This is a placeholder for the actual pathway clustering
        # The actual implementation would depend on the specific data structure and availability
        st.info("""
        Pathway clustering requires feature engineering to represent patient journeys in a format suitable for clustering algorithms.

        This analysis would typically include:
        1. Feature extraction from patient transfers and care unit sequences
        2. Representation of temporal patterns and transitions
        3. Application of clustering algorithms to identify similar pathways
        4. Visualization of cluster characteristics and representative pathways

        The implementation depends on the specific transfer and care unit data available in your MIMIC-IV dataset.
        """)

        # Clustering parameters
        st.write("#### Clustering Parameters")

        col1, col2 = st.columns(2)

        with col1:
            # Clustering algorithm
            clustering_algorithm = st.selectbox(
                "Clustering Algorithm",
                options=["K-Means", "DBSCAN"],
                index=0
            )

        with col2:
            # Number of clusters (for K-Means)
            if clustering_algorithm == "K-Means":
                n_clusters = st.slider(
                    "Number of Clusters",
                    min_value=2,
                    max_value=10,
                    value=5
                )
            else:  # DBSCAN
                eps = st.slider(
                    "DBSCAN Epsilon",
                    min_value=0.1,
                    max_value=2.0,
                    value=0.5,
                    step=0.1
                )

        # Example visualization (placeholder)
        st.write("#### Example Pathway Clusters")

        # Create sample data for demonstration
        np.random.seed(42)
        n_samples = 200

        # Create 5 clusters with different characteristics
        cluster_centers = [
            (2, 5),   # Cluster 1
            (8, 8),   # Cluster 2
            (3, 1),   # Cluster 3
            (8, 2),   # Cluster 4
            (5, 5)    # Cluster 5
        ]

        # Generate cluster data
        X = []
        y = []
        cluster_sizes = [50, 40, 30, 50, 30]  # Number of samples in each cluster

        for i, (center_x, center_y) in enumerate(cluster_centers):
            n_cluster_samples = cluster_sizes[i]
            cluster_x = np.random.normal(center_x, 0.5, n_cluster_samples)
            cluster_y = np.random.normal(center_y, 0.5, n_cluster_samples)

            for j in range(n_cluster_samples):
                X.append([cluster_x[j], cluster_y[j]])
                y.append(i)

        # Create dataframe
        example_df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
        example_df['Cluster'] = y
        example_df['Cluster'] = example_df['Cluster'].map(lambda x: f"Cluster {x+1}")

        # Create scatter plot
        fig = px.scatter(
            example_df,
            x='Feature 1',
            y='Feature 2',
            color='Cluster',
            title="Example: Patient Pathway Clusters",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(
            xaxis_title='Feature 1 (e.g., LOS)',
            yaxis_title='Feature 2 (e.g., Number of Transfers)'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Example cluster characteristics
        st.write("#### Example Cluster Characteristics")

        # Create sample data for cluster characteristics
        cluster_chars = [
            {'Cluster': 'Cluster 1', 'Avg LOS': 3.2, 'Avg Transfers': 1.5, 'Common First Unit': 'ED', 'Common Last Unit': 'Medicine', 'Discharge Home %': 85},
            {'Cluster': 'Cluster 2', 'Avg LOS': 8.7, 'Avg Transfers': 4.2, 'Common First Unit': 'ED', 'Common Last Unit': 'Rehabilitation', 'Discharge Home %': 62},
            {'Cluster': 'Cluster 3', 'Avg LOS': 2.1, 'Avg Transfers': 1.2, 'Common First Unit': 'ED', 'Common Last Unit': 'Surgery', 'Discharge Home %': 90},
            {'Cluster': 'Cluster 4', 'Avg LOS': 12.5, 'Avg Transfers': 5.8, 'Common First Unit': 'ED', 'Common Last Unit': 'ICU', 'Discharge Home %': 45},
            {'Cluster': 'Cluster 5', 'Avg LOS': 5.4, 'Avg Transfers': 2.7, 'Common First Unit': 'ED', 'Common Last Unit': 'Medicine', 'Discharge Home %': 75},
        ]

        # Create dataframe
        cluster_chars_df = pd.DataFrame(cluster_chars)

        # Display as table
        st.dataframe(cluster_chars_df)

        # Example representative pathways
        st.write("#### Example Representative Pathways")

        # Create sample data for representative pathways
        pathways = [
            {'Cluster': 'Cluster 1', 'Pathway': 'ED → Medicine → Discharge', 'Frequency': 65},
            {'Cluster': 'Cluster 2', 'Pathway': 'ED → Surgery → ICU → Step Down → Rehabilitation → Discharge', 'Frequency': 48},
            {'Cluster': 'Cluster 3', 'Pathway': 'ED → Surgery → Discharge', 'Frequency': 72},
            {'Cluster': 'Cluster 4', 'Pathway': 'ED → ICU → Step Down → ICU → Step Down → Medicine → Discharge', 'Frequency': 38},
            {'Cluster': 'Cluster 5', 'Pathway': 'ED → Medicine → Step Down → Medicine → Discharge', 'Frequency': 55},
        ]

        # Create dataframe
        pathways_df = pd.DataFrame(pathways)

        # Create bar chart
        fig = px.bar(
            pathways_df,
            x='Cluster',
            y='Frequency',
            color='Cluster',
            title="Example: Representative Pathways by Cluster",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            text='Pathway'
        )
        fig.update_layout(
            xaxis_title='Cluster',
            yaxis_title='Frequency (%)',
            height=500
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    def dimensionality_reduction(self):
        """Display dimensionality reduction plots of patient states."""
        if 'transfers' not in self.data or self.data['transfers'] is None:
            st.warning("Transfers data not available for dimensionality reduction.")
            return

        if 'admissions' not in self.data or self.data['admissions'] is None:
            st.warning("Admissions data not available for dimensionality reduction.")
            return

        transfers_df = self.data['transfers']
        admissions_df = self.data['admissions']

        st.subheader("Dimensionality Reduction Plots")

        # This is a placeholder for the actual dimensionality reduction
        # The actual implementation would depend on the specific data structure and availability
        st.info("""
        Dimensionality reduction requires feature engineering to represent patient states in a high-dimensional space.

        This analysis would typically include:
        1. Feature extraction from patient data, transfers, and orders
        2. Normalization and scaling of features
        3. Application of dimensionality reduction techniques (PCA, t-SNE)
        4. Visualization of reduced dimensions with clinical annotations

        The implementation depends on the specific patient data available in your MIMIC-IV dataset.
        """)

        # Dimensionality reduction parameters
        st.write("#### Dimensionality Reduction Parameters")

        col1, col2 = st.columns(2)

        with col1:
            # Dimensionality reduction technique
            reduction_technique = st.selectbox(
                "Reduction Technique",
                options=["PCA", "t-SNE"],
                index=0
            )

        with col2:
            # Color by
            color_by = st.selectbox(
                "Color By",
                options=["Discharge Location", "Length of Stay", "Admission Type"],
                index=0
            )

        # Example PCA plot
        st.write("#### Example PCA Plot")

        # Create sample data for demonstration
        np.random.seed(42)
        n_samples = 200

        # Generate sample data with 3 clusters
        X = []
        colors = []

        # Cluster 1: Discharged Home
        for i in range(100):
            X.append([np.random.normal(2, 0.5), np.random.normal(2, 0.5)])
            colors.append("Home")

        # Cluster 2: Skilled Nursing Facility
        for i in range(60):
            X.append([np.random.normal(4, 0.5), np.random.normal(4, 0.5)])
            colors.append("SNF")

        # Cluster 3: Rehabilitation
        for i in range(40):
            X.append([np.random.normal(1, 0.5), np.random.normal(4, 0.5)])
            colors.append("Rehab")

        # Create dataframe
        example_df = pd.DataFrame(X, columns=['PC1', 'PC2'])
        example_df['Discharge Location'] = colors

        # Create scatter plot
        fig = px.scatter(
            example_df,
            x='PC1',
            y='PC2',
            color='Discharge Location',
            title="Example: PCA of Patient States",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(
            xaxis_title='Principal Component 1',
            yaxis_title='Principal Component 2'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Example t-SNE plot
        st.write("#### Example t-SNE Plot")

        # Create sample data for demonstration
        np.random.seed(42)

        # Generate more complex clusters for t-SNE
        X = []
        colors = []
        los = []

        # Cluster 1: Short LOS
        for i in range(80):
            X.append([np.random.normal(2, 0.3), np.random.normal(2, 0.3)])
            colors.append("Home")
            los.append(np.random.normal(2, 0.5))

        # Cluster 2: Medium LOS
        for i in range(70):
            X.append([np.random.normal(4, 0.3), np.random.normal(4, 0.3)])
            colors.append("SNF")
            los.append(np.random.normal(7, 1.5))

        # Cluster 3: Long LOS
        for i in range(50):
            X.append([np.random.normal(1, 0.3), np.random.normal(4, 0.3)])
            colors.append("Rehab")
            los.append(np.random.normal(12, 2.5))

        # Create dataframe
        example_df = pd.DataFrame(X, columns=['TSNE1', 'TSNE2'])
        example_df['Discharge Location'] = colors
        example_df['Length of Stay'] = [max(1, x) for x in los]  # Ensure no negative LOS

        # Create scatter plot
        fig = px.scatter(
            example_df,
            x='TSNE1',
            y='TSNE2',
            color='Length of Stay',
            hover_data=['Discharge Location'],
            title="Example: t-SNE of Patient States",
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            xaxis_title='t-SNE Dimension 1',
            yaxis_title='t-SNE Dimension 2'
        )
        st.plotly_chart(fig, use_container_width=True)

    def survival_analysis(self):
        """Display survival analysis for time-to-discharge."""
        if 'admissions' not in self.data or self.data['admissions'] is None:
            st.warning("Admissions data not available for survival analysis.")
            return

        admissions_df = self.data['admissions']

        st.subheader("Survival Analysis for Time-to-Discharge")

        # Check if we have the necessary data for survival analysis
        required_columns = ['admittime', 'dischtime', 'deathtime', 'discharge_location']
        missing_columns = [col for col in required_columns if col not in admissions_df.columns]

        if missing_columns:
            st.warning(f"Missing required columns for survival analysis: {', '.join(missing_columns)}")
            st.info("""
            Survival analysis for time-to-discharge requires defining the event (discharge) and censoring criteria.

            This analysis would typically include:
            1. Definition of the time-to-event variable (time to discharge)
            2. Handling of censoring (e.g., death, transfer to another facility)
            3. Application of survival analysis techniques (Kaplan-Meier, Cox Proportional Hazards)
            4. Visualization of survival curves and hazard ratios

            The implementation depends on the specific admission and discharge data available in your MIMIC-IV dataset.
            """)

            # Use example data for demonstration
            use_example_data = True
        else:
            # We have the necessary data, ask if user wants to use real data or example data
            use_example_data = not st.checkbox("Use actual MIMIC-IV data for analysis", value=False,
                                           help="If unchecked, example data will be used for demonstration")

        # Survival analysis parameters
        st.write("#### Survival Analysis Parameters")

        col1, col2 = st.columns(2)

        with col1:
            # Stratification variable options based on available data
            stratify_options = ["None"]

            if not use_example_data:
                if 'admission_type' in admissions_df.columns:
                    stratify_options.append("Admission Type")
                if 'patients' in self.data and self.data['patients'] is not None:
                    if 'gender' in self.data['patients'].columns:
                        stratify_options.append("Gender")
                    if 'age_group' in self.data['patients'].columns:
                        stratify_options.append("Age Group")
            else:
                stratify_options = ["Admission Type", "Gender", "Age Group", "None"]

            # Stratification variable
            stratify_by = st.selectbox(
                "Stratify By",
                options=stratify_options,
                index=0 if "None" in stratify_options else stratify_options.index("Admission Type")
            )

        with col2:
            # Maximum follow-up time
            max_time = st.slider(
                "Maximum Follow-up Time (days)",
                min_value=1,
                max_value=30,
                value=14
            )

        # Prepare data for analysis
        if use_example_data:
            # Create sample data for demonstration
            st.write("#### Kaplan-Meier Survival Curves (Example Data)")
            np.random.seed(42)
            n_samples = 1000

            # Generate sample data
            times = []
            events = []
            groups = []

            # Group 1: Emergency admissions (faster discharge)
            for i in range(500):
                time = np.random.exponential(scale=5)
                if time > max_time:
                    time = max_time
                    event = 0  # Censored
                else:
                    event = 1  # Discharged
                times.append(time)
                events.append(event)
                groups.append("Emergency")

            # Group 2: Elective admissions (slower discharge)
            for i in range(500):
                time = np.random.exponential(scale=8)
                if time > max_time:
                    time = max_time
                    event = 0  # Censored
                else:
                    event = 1  # Discharged
                times.append(time)
                events.append(event)
                groups.append("Elective")

            # Create dataframe
            analysis_df = pd.DataFrame({
                'time': times,
                'event': events,
                'group': groups
            })

            # Add additional variables for Cox model
            analysis_df['age'] = np.random.normal(65, 15, size=len(analysis_df))
            analysis_df['age'] = analysis_df['age'].clip(18, 90)
            analysis_df['gender'] = np.random.choice(['M', 'F'], size=len(analysis_df))
            analysis_df['comorbidity_score'] = np.random.normal(3, 2, size=len(analysis_df))
            analysis_df['comorbidity_score'] = analysis_df['comorbidity_score'].clip(0, 10)

            # Stratification variable
            if stratify_by == "Admission Type":
                strata = 'group'
            elif stratify_by == "Gender":
                strata = 'gender'
            elif stratify_by == "Age Group":
                # Create age groups
                bins = [0, 30, 50, 70, 100]
                labels = ['<30', '30-50', '51-70', '>70']
                analysis_df['age_group'] = pd.cut(analysis_df['age'], bins=bins, labels=labels, right=False)
                strata = 'age_group'
            else:
                strata = None
        else:
            # Use actual MIMIC-IV data
            st.write("#### Kaplan-Meier Survival Curves (MIMIC-IV Data)")

            # Prepare the data for survival analysis
            analysis_df = admissions_df.copy()

            # Calculate time to discharge in days
            analysis_df['time'] = (analysis_df['dischtime'] - analysis_df['admittime']).dt.total_seconds() / (24 * 3600)

            # Cap at max_time and determine event status
            analysis_df['time'] = analysis_df['time'].clip(upper=max_time)

            # Event is 1 if discharged within max_time, 0 if censored (still in hospital or died)
            analysis_df['event'] = 1

            # Censoring: if time equals max_time or patient died
            analysis_df.loc[analysis_df['time'] >= max_time, 'event'] = 0
            if 'deathtime' in analysis_df.columns:
                analysis_df.loc[~analysis_df['deathtime'].isna(), 'event'] = 0

            # Determine stratification variable
            if stratify_by == "Admission Type" and 'admission_type' in analysis_df.columns:
                strata = 'admission_type'
            elif stratify_by == "Gender" and 'patients' in self.data and self.data['patients'] is not None:
                # Merge with patients data to get gender
                patients_df = self.data['patients']
                if 'gender' in patients_df.columns:
                    analysis_df = analysis_df.merge(patients_df[['subject_id', 'gender']], on='subject_id', how='left')
                    strata = 'gender'
                else:
                    strata = None
            elif stratify_by == "Age Group" and 'patients' in self.data and self.data['patients'] is not None:
                # Merge with patients data to get age group
                patients_df = self.data['patients']
                if 'age_group' in patients_df.columns:
                    analysis_df = analysis_df.merge(patients_df[['subject_id', 'age_group']], on='subject_id', how='left')
                    strata = 'age_group'
                else:
                    strata = None
            else:
                strata = None

        # Create Kaplan-Meier curves
        if strata is not None:
            # Get unique strata values
            strata_values = analysis_df[strata].unique()

            # Create Plotly figure
            fig = go.Figure()

            # Color palette
            colors = px.colors.qualitative.Plotly

            # Add trace for each stratum
            for i, value in enumerate(strata_values):
                # Filter data for this stratum
                mask = analysis_df[strata] == value
                if mask.sum() > 0:  # Only proceed if we have data
                    kmf = KaplanMeierFitter()
                    kmf.fit(analysis_df.loc[mask, 'time'], analysis_df.loc[mask, 'event'], label=str(value))

                    # Get survival function values
                    sf = kmf.survival_function_

                    # Add to plot
                    fig.add_trace(go.Scatter(
                        x=sf.index.values,
                        y=sf['KM_estimate'].values,
                        mode='lines',
                        name=str(value),
                        line=dict(color=colors[i % len(colors)]),
                        hovertemplate='Time: %{x:.1f} days<br>Survival: %{y:.2f}<extra></extra>'
                    ))

                    # Add confidence intervals
                    if hasattr(kmf, 'confidence_interval_'):
                        ci = kmf.confidence_interval_
                        fig.add_trace(go.Scatter(
                            x=list(ci.index.values) + list(ci.index.values[::-1]),
                            y=list(ci['KM_estimate_lower_0.95'].values) + list(ci['KM_estimate_upper_0.95'].values[::-1]),
                            fill='toself',
                            fillcolor=colors[i % len(colors)].replace('rgb', 'rgba').replace(')', ', 0.2)'),
                            line=dict(color='rgba(0,0,0,0)'),
                            name=f"{value} (95% CI)",
                            showlegend=False,
                            hoverinfo='skip'
                        ))

            # Update layout
            fig.update_layout(
                title=f"Kaplan-Meier Survival Curves by {stratify_by}",
                xaxis_title="Time (days)",
                yaxis_title="Probability of Remaining in Hospital",
                yaxis=dict(range=[0, 1]),
                hovermode='closest',
                legend=dict(title=stratify_by)
            )

            # Display the plot
            st.plotly_chart(fig, use_container_width=True)

            # Perform log-rank test if we have more than one group
            if len(strata_values) > 1 and len(strata_values) <= 5:  # Limit to 5 groups for simplicity
                st.write("#### Log-Rank Test Results")

                # Prepare data for pairwise log-rank tests
                results = []

                for i, val1 in enumerate(strata_values):
                    for val2 in strata_values[i+1:]:
                        # Get data for both groups
                        group1 = analysis_df[analysis_df[strata] == val1]
                        group2 = analysis_df[analysis_df[strata] == val2]

                        # Perform log-rank test
                        try:
                            result = logrank_test(
                                group1['time'], group2['time'],
                                group1['event'], group2['event']
                            )

                            # Add to results
                            results.append({
                                'Group 1': val1,
                                'Group 2': val2,
                                'p-value': result.p_value,
                                'Significant': 'Yes' if result.p_value < 0.05 else 'No'
                            })
                        except Exception as e:
                            st.warning(f"Could not perform log-rank test between {val1} and {val2}: {e}")

                # Display results as table
                if results:
                    results_df = pd.DataFrame(results)
                    results_df['p-value'] = results_df['p-value'].apply(lambda x: f"{x:.4f}")
                    st.dataframe(results_df)

                    # Interpretation
                    significant_pairs = results_df[results_df['Significant'] == 'Yes']
                    if len(significant_pairs) > 0:
                        st.info(f"There are statistically significant differences in time-to-discharge between some groups (p < 0.05).")
                    else:
                        st.info("No statistically significant differences in time-to-discharge between groups.")
        else:
            # Single Kaplan-Meier curve for all data
            kmf = KaplanMeierFitter()
            kmf.fit(analysis_df['time'], analysis_df['event'], label='All Patients')

            # Create Plotly figure
            fig = go.Figure()

            # Get survival function values
            sf = kmf.survival_function_

            # Add to plot
            fig.add_trace(go.Scatter(
                x=sf.index.values,
                y=sf['KM_estimate'].values,
                mode='lines',
                name='All Patients',
                line=dict(color='rgb(31, 119, 180)'),
                hovertemplate='Time: %{x:.1f} days<br>Survival: %{y:.2f}<extra></extra>'
            ))

            # Add confidence intervals
            if hasattr(kmf, 'confidence_interval_'):
                ci = kmf.confidence_interval_
                fig.add_trace(go.Scatter(
                    x=list(ci.index.values) + list(ci.index.values[::-1]),
                    y=list(ci['KM_estimate_lower_0.95'].values) + list(ci['KM_estimate_upper_0.95'].values[::-1]),
                    fill='toself',
                    fillcolor='rgba(31, 119, 180, 0.2)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name='95% CI',
                    showlegend=False,
                    hoverinfo='skip'
                ))

            # Update layout
            fig.update_layout(
                title="Kaplan-Meier Survival Curve for All Patients",
                xaxis_title="Time (days)",
                yaxis_title="Probability of Remaining in Hospital",
                yaxis=dict(range=[0, 1]),
                hovermode='closest'
            )

            # Display the plot
            st.plotly_chart(fig, use_container_width=True)

        # Cox Proportional Hazards Model
        st.write("#### Cox Proportional Hazards Model")

        if use_example_data:
            # Use example data with predictors
            cox_df = analysis_df.copy()

            # Prepare predictors
            predictors = ['age', 'comorbidity_score']
            if 'gender' in cox_df.columns:
                # Convert gender to binary
                cox_df['gender_binary'] = (cox_df['gender'] == 'M').astype(int)
                predictors.append('gender_binary')
            if 'group' in cox_df.columns:
                # Convert admission type to binary (1 for Emergency)
                cox_df['emergency'] = (cox_df['group'] == 'Emergency').astype(int)
                predictors.append('emergency')

            # Fit Cox model
            cph = CoxPHFitter()
            try:
                cph.fit(cox_df[predictors + ['time', 'event']], duration_col='time', event_col='event')

                # Get summary
                summary_df = cph.summary

                # Format for display
                display_df = summary_df[['coef', 'exp(coef)', 'se(coef)', 'p']].copy()
                display_df.columns = ['Coefficient', 'Hazard Ratio', 'Standard Error', 'p-value']
                display_df['p-value'] = display_df['p-value'].apply(lambda x: f"{x:.4f}")
                display_df['Hazard Ratio'] = display_df['Hazard Ratio'].apply(lambda x: f"{x:.2f}")
                display_df['Coefficient'] = display_df['Coefficient'].apply(lambda x: f"{x:.4f}")
                display_df['Standard Error'] = display_df['Standard Error'].apply(lambda x: f"{x:.4f}")

                # Add interpretation column
                display_df['Interpretation'] = display_df.apply(
                    lambda row: "Faster discharge" if float(row['Hazard Ratio']) > 1 else "Slower discharge",
                    axis=1
                )

                # Display results
                st.dataframe(display_df)

                # Plot hazard ratios
                hr_fig = go.Figure()

                # Add reference line at HR=1
                hr_fig.add_shape(
                    type="line",
                    x0=0.9,
                    x1=1.1,
                    y0=-0.5,
                    y1=len(predictors) - 0.5,
                    line=dict(color="gray", width=1, dash="dash")
                )

                # Add hazard ratios
                y_pos = list(range(len(predictors)))
                hr_values = summary_df['exp(coef)'].values
                hr_lower = summary_df['exp(coef)'].values - 1.96 * summary_df['se(coef)'].values
                hr_upper = summary_df['exp(coef)'].values + 1.96 * summary_df['se(coef)'].values

                hr_fig.add_trace(go.Scatter(
                    x=hr_values,
                    y=y_pos,
                    mode='markers',
                    marker=dict(size=10, color='blue'),
                    error_x=dict(
                        type='data',
                        symmetric=False,
                        array=hr_upper - hr_values,
                        arrayminus=hr_values - hr_lower
                    ),
                    name='Hazard Ratio'
                ))

                # Update layout
                hr_fig.update_layout(
                    title="Hazard Ratios with 95% Confidence Intervals",
                    xaxis_title="Hazard Ratio",
                    yaxis=dict(
                        tickmode='array',
                        tickvals=y_pos,
                        ticktext=predictors
                    ),
                    xaxis=dict(type='log'),
                    showlegend=False
                )

                # Display the plot
                st.plotly_chart(hr_fig, use_container_width=True)

                # Interpretation
                st.write("#### Interpretation")
                st.write("""
                The Cox Proportional Hazards model estimates the effect of different factors on the time to discharge:
                - **Hazard Ratio > 1**: Factor is associated with faster discharge (shorter hospital stay)
                - **Hazard Ratio < 1**: Factor is associated with slower discharge (longer hospital stay)
                - **p-value < 0.05**: The effect is statistically significant
                """)

                # Specific interpretations based on the model
                significant_predictors = display_df[display_df['p-value'].apply(lambda x: float(x) < 0.05)]
                if len(significant_predictors) > 0:
                    st.write("**Significant predictors of discharge time:**")
                    for idx, row in significant_predictors.iterrows():
                        predictor = idx
                        hr = row['Hazard Ratio']
                        interp = row['Interpretation']
                        st.write(f"- **{predictor}**: Hazard Ratio = {hr} ({interp})")
                else:
                    st.write("No statistically significant predictors were found in this model.")

            except Exception as e:
                st.error(f"Error fitting Cox model: {e}")
                st.info("Cox Proportional Hazards model requires sufficient data with predictors and events.")
        else:
            # Use actual MIMIC-IV data if available
            st.info("""
            To fit a Cox Proportional Hazards model with actual MIMIC-IV data, you would need:
            1. Time-to-event data (time to discharge)
            2. Event indicator (discharged or censored)
            3. Predictors (e.g., age, gender, comorbidities, admission type)

            The implementation would depend on the specific predictors available in your dataset.
            """)

            # Check if we have necessary data for Cox model
            if not use_example_data and 'patients' in self.data and self.data['patients'] is not None:
                st.write("Would you like to fit a Cox model with the available data?")
                fit_cox = st.checkbox("Fit Cox Proportional Hazards model", value=False)

                if fit_cox:
                    # This would be implemented with actual MIMIC-IV data
                    # Similar to the example implementation above
                    st.warning("Cox model with actual MIMIC-IV data not implemented in this demo.")

        # Additional information about survival analysis
        with st.expander("About Survival Analysis in Healthcare"):
            st.write("""
            **Survival analysis** is particularly useful in healthcare for analyzing time-to-event data, such as:
            - Length of hospital stay
            - Time to readmission
            - Mortality analysis
            - Treatment effectiveness

            **Key concepts:**
            - **Censoring**: When the event of interest doesn't occur during the observation period
            - **Kaplan-Meier curves**: Non-parametric estimation of survival function
            - **Log-rank test**: Comparing survival curves between groups
            - **Cox Proportional Hazards**: Modeling the effect of predictors on survival

            **Interpretation:**
            - Survival curves show the probability of remaining in the hospital over time
            - Steeper curves indicate faster discharge rates
            - Hazard ratios quantify the effect of factors on discharge timing
            """)
