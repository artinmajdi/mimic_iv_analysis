# Standard library imports
import os
import logging
import datetime
import pickle
from io import BytesIO
from collections import defaultdict, Counter

# Data processing imports
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Visualization imports
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine learning imports
from scipy.cluster.hierarchy import dendrogram
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

# Streamlit import
import streamlit as st

# Local application imports (assuming these exist in the specified structure)
try:
    from mimic_iv_analysis.core import (
        MIMICClusteringAnalysis,
        MIMICClusterAnalyzer,
        MIMICFeatureEngineer,
        MIMICDataLoader,
        MIMICVisualizer
    )
    from mimic_iv_analysis.visualization.app_components import FilteringTab
except ImportError:
    # Provide dummy classes if the core library is not available
    # This allows the Streamlit app structure to be viewed/run partially
    st.warning("Could not import core 'mimic_iv_analysis' library. Using dummy classes.")
    class DummyClass:
        def __getattr__(self, name):
            def method(*args, **kwargs):
                st.info(f"Called dummy method: {name}")
                if name in ['create_order_frequency_matrix', 'extract_temporal_order_sequences',
                            'get_order_type_distributions', 'create_order_timing_features',
                            'preprocess_data', 'apply_dimensionality_reduction',
                            'run_kmeans_clustering', 'find_optimal_k_for_kmeans',
                            'run_hierarchical_clustering', 'run_dbscan_clustering',
                            'find_optimal_eps_for_dbscan', 'run_lda_topic_modeling',
                            'get_top_terms_per_topic', 'calculate_length_of_stay',
                            'compare_los_across_clusters', 'visualize_order_patterns',
                            'statistical_testing', 'generate_cluster_characterization',
                            'calculate_feature_importance', 'generate_html_report',
                            'save_features', 'save_model', 'load_model', 'evaluate_clustering',
                            'scan_mimic_directory', 'load_mimic_table', 'get_table_info',
                            'display_data_preview', 'display_dataset_statistics', 'display_visualizations',
                            'detect_order_columns', 'detect_temporal_columns', 'detect_patient_id_column']:
                     # Return dummy data structures where appropriate
                    if name == 'create_order_frequency_matrix': return pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]}, index=[101, 102])
                    if name == 'extract_temporal_order_sequences': return {101: ['orderA', 'orderB'], 102: ['orderC']}
                    if name == 'get_order_type_distributions': return pd.DataFrame({'order_type': ['A', 'B'], 'frequency': [10, 5]}), pd.DataFrame({'patient_id': [101, 101, 102], 'order_type': ['A', 'B', 'C'], 'frequency': [1, 1, 1]})
                    if name == 'create_order_timing_features': return pd.DataFrame({'total_orders': [2, 1]}, index=[101, 102])
                    if name == 'preprocess_data': return pd.DataFrame({'f1': [0.1, 0.2], 'f2': [0.3, 0.4]})
                    if name == 'apply_dimensionality_reduction': return pd.DataFrame({'dim1': [1, 2], 'dim2': [3, 4]})
                    if name == 'run_kmeans_clustering': return pd.Series([0, 1]), KMeans(n_clusters=2)
                    if name == 'find_optimal_k_for_kmeans': return 3, pd.DataFrame({'k': [2, 3, 4], 'metric': [0.5, 0.7, 0.6]})
                    if name == 'run_hierarchical_clustering': return pd.Series([0, 1]), {'linkage_matrix': np.array([[0,1,0.5,2]])}
                    if name == 'run_dbscan_clustering': return pd.Series([0, -1]), DBSCAN(eps=0.5)
                    if name == 'find_optimal_eps_for_dbscan': return 0.5, np.array([0.1, 0.2, 0.5, 0.6])
                    if name == 'run_lda_topic_modeling': return LatentDirichletAllocation(), pd.DataFrame({'Topic 1': [0.8, 0.2]}), pd.DataFrame({'Term 1': [0.6, 0.1]})
                    if name == 'get_top_terms_per_topic': return pd.DataFrame({'Topic 1': ['termA', 'termB']})
                    if name == 'calculate_length_of_stay': return pd.Series([5.2, 3.1], index=[101, 102])
                    if name == 'compare_los_across_clusters': return pd.DataFrame({'Cluster': [0, 1], 'Mean LOS': [5.2, 3.1]})
                    if name == 'visualize_order_patterns': return go.Figure()
                    if name == 'statistical_testing': return pd.DataFrame({'Feature': ['f1'], 'P-Value': [0.01], 'Significant': [True]})
                    if name == 'generate_cluster_characterization': return {0: {'size': 1, 'percentage': 50.0, 'features': {'f1': {'mean': 0.1}}}}
                    if name == 'calculate_feature_importance': return pd.DataFrame({'Feature': ['f1'], 'Importance': [0.9]})
                    if name == 'generate_html_report': return "<html><body>Dummy Report</body></html>"
                    if name == 'evaluate_clustering': return {'silhouette_score': 0.6, 'davies_bouldin_score': 0.8, 'calinski_harabasz_score': 150.0}
                    if name == 'scan_mimic_directory': return ({'hosp': ['patients', 'admissions']}, {('hosp', 'patients'): 'path1'}, {('hosp', 'patients'): 10.5}, {('hosp', 'patients'): 'Patients (10.5 MB)'})
                    if name == 'load_mimic_table': return pd.DataFrame({'subject_id': [1, 2], 'colA': ['a', 'b']}), 2
                    if name == 'get_table_info': return "Dummy table info"
                    if name == 'detect_order_columns': return ['order_col']
                    if name == 'detect_temporal_columns': return ['time_col']
                    if name == 'detect_patient_id_column': return 'subject_id'
                return None # Default return for other methods
            return method

    MIMICClusteringAnalysis = DummyClass
    MIMICClusterAnalyzer = DummyClass
    MIMICFeatureEngineer = DummyClass
    MIMICDataLoader = DummyClass
    MIMICVisualizer = DummyClass
    FilteringTab = DummyClass # Assuming FilteringTab is also part of the library


# Constants
DEFAULT_MIMIC_PATH      = "/Users/artinmajdi/Documents/GitHubs/Career/mimic_iv/dataset/mimic-iv-3.1" # Replace with your path or leave empty
LARGE_FILE_THRESHOLD_MB = 100
DEFAULT_SAMPLE_SIZE     = 1000
RANDOM_STATE            = 42

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Tab Classes ---

class FeatureEngineeringTab:
    """ Encapsulates the UI and logic for the Feature Engineering tab. """
    def __init__(self, feature_engineer):
        self.feature_engineer = feature_engineer

    def _render_export_options(self, st, data, feature_type):
        """ Helper function to display export options for engineered features. """
        with st.expander("#### Export Options"):
            save_format = st.radio(f"Save Format ({feature_type})", ["CSV", "Parquet"], horizontal=True, key=f"export_fmt_{feature_type}")
            if st.button(f"Save {feature_type.replace('_', ' ').title()}"):
                try:
                    filepath = self.feature_engineer.save_features(
                        features=data,
                        feature_type=feature_type,
                        base_path=os.path.dirname(st.session_state.current_file_path),
                        format=save_format.lower()
                    )
                    st.success(f"Saved {feature_type.replace('_', ' ')} to {filepath}")
                except Exception as e:
                    st.error(f"Error saving {feature_type.replace('_', ' ')}: {str(e)}")

    def render(self, st, all_columns):
        """ Renders the content of the Feature Engineering tab. """
        st.markdown("<h2 class='sub-header'>Order Data Feature Engineering</h2>", unsafe_allow_html=True)
        st.info("This section allows you to transform raw MIMIC-IV order data into structured features for analysis and machine learning. Choose one of the feature engineering methods below to get started.")

        feature_tabs = st.tabs([
            "📊 Order Frequency Matrix",
            "⏱️ Temporal Order Sequences",
            "📈 Order Type Distributions",
            "🕒 Order Timing Analysis"
        ])

        # 1. Order Frequency Matrix tab
        with feature_tabs[0]:
            st.markdown("### Create Order Frequency Matrix")
            st.info("Creates a matrix where rows are patients and columns are order types, with cells showing frequency.")

            col1, col2 = st.columns(2)
            with col1:
                patient_id_col = st.selectbox("Select Patient ID Column", all_columns, index=all_columns.index(st.session_state.detected_patient_id_col) if st.session_state.detected_patient_id_col in all_columns else 0, key="freq_pid", help="Column containing unique patient identifiers")
            with col2:
                default_order_idx = all_columns.index(st.session_state.detected_order_cols[0]) if st.session_state.detected_order_cols and st.session_state.detected_order_cols[0] in all_columns else 0
                order_col = st.selectbox("Select Order Type Column", all_columns, index=default_order_idx, key="freq_order", help="Column containing order types/names")

            col1, col2 = st.columns(2)
            with col1:
                normalize = st.checkbox("Normalize by Patient", value=False, key="freq_norm", help="Convert frequencies to percentages")
            with col2:
                top_n = st.number_input("Top N Order Types", min_value=0, max_value=100, value=20, key="freq_topn", help="Limit to most frequent order types (0 = include all)")

            if st.button("Generate Order Frequency Matrix"):
                try:
                    with st.spinner("Generating order frequency matrix..."):
                        freq_matrix = self.feature_engineer.create_order_frequency_matrix(st.session_state.df, patient_id_col, order_col, normalize, top_n)
                        st.session_state.freq_matrix = freq_matrix
                        st.session_state.clustering_input_data = freq_matrix # Default input for clustering
                except Exception as e:
                    st.error(f"Error generating frequency matrix: {str(e)}")

            if st.session_state.freq_matrix is not None:
                st.markdown("<h4>Order Frequency Matrix Preview</h4>", unsafe_allow_html=True)
                st.dataframe(st.session_state.freq_matrix.head(10), use_container_width=True)
                st.markdown(f"<div class='info-box'>Matrix size: {st.session_state.freq_matrix.shape[0]} patients × {st.session_state.freq_matrix.shape[1]} order types</div>", unsafe_allow_html=True)

                st.markdown("<h4>Frequency Heatmap (Sample)</h4>", unsafe_allow_html=True)
                fig = px.imshow(st.session_state.freq_matrix.sample(min(50, len(st.session_state.freq_matrix)), random_state=RANDOM_STATE).T, labels=dict(x="Patient ID", y="Order Type", color="Count"), aspect="auto")
                st.plotly_chart(fig, use_container_width=True)
                self._render_export_options(st, st.session_state.freq_matrix, 'order_frequency_matrix')

        # 2. Temporal Order Sequences tab
        with feature_tabs[1]:
            st.markdown("<h3>Extract Temporal Order Sequences</h3>", unsafe_allow_html=True)
            st.info("Extracts chronological sequences of orders for each patient.")

            col1, col2, col3 = st.columns(3)
            with col1:
                seq_patient_id_col = st.selectbox("Select Patient ID Column", all_columns, index=all_columns.index(st.session_state.detected_patient_id_col) if st.session_state.detected_patient_id_col in all_columns else 0, key="seq_pid", help="Column containing unique patient identifiers")
            with col2:
                default_order_idx = all_columns.index(st.session_state.detected_order_cols[0]) if st.session_state.detected_order_cols and st.session_state.detected_order_cols[0] in all_columns else 0
                seq_order_col = st.selectbox("Select Order Type Column", all_columns, index=default_order_idx, key="seq_order", help="Column containing order types/names")
            with col3:
                default_time_idx = all_columns.index(st.session_state.detected_time_cols[0]) if st.session_state.detected_time_cols and st.session_state.detected_time_cols[0] in all_columns else 0
                seq_time_col = st.selectbox("Select Timestamp Column", all_columns, index=default_time_idx, key="seq_time", help="Column containing order timestamps")

            max_seq_length = st.slider("Maximum Sequence Length", min_value=5, max_value=100, value=20, key="seq_len", help="Max orders per sequence")

            if st.button("Extract Order Sequences"):
                try:
                    with st.spinner("Extracting temporal order sequences..."):
                        sequences = self.feature_engineer.extract_temporal_order_sequences(st.session_state.df, seq_patient_id_col, seq_order_col, seq_time_col, max_seq_length)
                        st.session_state.order_sequences = sequences
                        # Also generate transition matrix automatically
                        transition_matrix = self.feature_engineer.calculate_order_transition_matrix(sequences=sequences, top_n=15)
                        st.session_state.transition_matrix = transition_matrix
                except Exception as e:
                    st.error(f"Error extracting order sequences: {str(e)}")

            if st.session_state.order_sequences is not None:
                num_patients = len(st.session_state.order_sequences)
                avg_len = np.mean([len(seq) for seq in st.session_state.order_sequences.values()]) if num_patients > 0 else 0
                st.markdown("<h4>Sequence Statistics</h4>", unsafe_allow_html=True)
                st.markdown(f"<div class='info-box'><p><strong>Patients:</strong> {num_patients}</p><p><strong>Avg. Length:</strong> {avg_len:.2f} orders</p></div>", unsafe_allow_html=True)

                st.markdown("<h4>Sample Order Sequences</h4>", unsafe_allow_html=True)
                sample_patients = list(st.session_state.order_sequences.keys())[:5]
                for patient in sample_patients:
                    sequence_str = " → ".join(map(str, st.session_state.order_sequences[patient]))
                    st.markdown(f"<strong>Patient {patient}:</strong> {sequence_str}", unsafe_allow_html=True)
                    st.markdown("<hr>", unsafe_allow_html=True)

                if st.session_state.transition_matrix is not None:
                    st.markdown("<h4>Order Transition Matrix (Top 15)</h4>", unsafe_allow_html=True)
                    fig = px.imshow(st.session_state.transition_matrix, labels=dict(x="Next Order", y="Current Order", color="Prob."), color_continuous_scale='Blues')
                    fig.update_layout(height=700)
                    st.plotly_chart(fig, use_container_width=True)

                self._render_export_options(st, st.session_state.order_sequences, 'temporal_order_sequences') # Note: Saving dict might need JSON format

        # 3. Order Type Distributions tab
        with feature_tabs[2]:
            st.markdown("<h3>Analyze Order Type Distributions</h3>", unsafe_allow_html=True)
            st.info("Analyzes the distribution of order types across the dataset and per patient.")

            col1, col2 = st.columns(2)
            with col1:
                dist_patient_id_col = st.selectbox("Select Patient ID Column", all_columns, index=all_columns.index(st.session_state.detected_patient_id_col) if st.session_state.detected_patient_id_col in all_columns else 0, key="dist_pid", help="Column containing unique patient identifiers")
            with col2:
                default_order_idx = all_columns.index(st.session_state.detected_order_cols[0]) if st.session_state.detected_order_cols and st.session_state.detected_order_cols[0] in all_columns else 0
                dist_order_col = st.selectbox("Select Order Type Column", all_columns, index=default_order_idx, key="dist_order", help="Column containing order types/names")

            if st.button("Analyze Order Distributions"):
                try:
                    with st.spinner("Analyzing order type distributions..."):
                        overall_dist, patient_dist = self.feature_engineer.get_order_type_distributions(st.session_state.df, dist_patient_id_col, dist_order_col)
                        st.session_state.order_dist = overall_dist
                        st.session_state.patient_order_dist = patient_dist
                except Exception as e:
                    st.error(f"Error analyzing order distributions: {str(e)}")

            if st.session_state.order_dist is not None:
                st.markdown("<h4>Overall Order Type Distribution</h4>", unsafe_allow_html=True)
                top_n_orders = 15
                top_orders = st.session_state.order_dist.head(top_n_orders)
                if len(st.session_state.order_dist) > top_n_orders:
                    others_sum = st.session_state.order_dist.iloc[top_n_orders:]['frequency'].sum()
                    other_row = pd.DataFrame({dist_order_col: ['Other'], 'frequency': [others_sum]})
                    pie_data = pd.concat([top_orders, other_row], ignore_index=True)
                else:
                    pie_data = top_orders

                fig_pie = px.pie(pie_data, values='frequency', names=dist_order_col, title=f"Overall Distribution (Top {top_n_orders})")
                st.plotly_chart(fig_pie, use_container_width=True)

                top_20 = st.session_state.order_dist.head(20)
                fig_bar = px.bar(top_20, x=dist_order_col, y='frequency', title=f"Top 20 {dist_order_col} by Frequency")
                st.plotly_chart(fig_bar, use_container_width=True)

                if st.session_state.patient_order_dist is not None and not st.session_state.patient_order_dist.empty:
                    st.markdown("<h4>Patient-Level Order Type Distribution (Sample)</h4>", unsafe_allow_html=True)
                    patients = st.session_state.patient_order_dist['patient_id'].unique()
                    sample_patients = patients[:min(5, len(patients))]
                    fig_sub = make_subplots(rows=len(sample_patients), cols=1, subplot_titles=[f"Patient {p}" for p in sample_patients])
                    for i, patient in enumerate(sample_patients):
                        patient_data = st.session_state.patient_order_dist[st.session_state.patient_order_dist['patient_id'] == patient].head(10)
                        fig_sub.add_trace(go.Bar(x=patient_data[dist_order_col], y=patient_data['frequency'], name=f"Patient {patient}"), row=i+1, col=1)
                    fig_sub.update_layout(height=200*len(sample_patients), showlegend=False)
                    st.plotly_chart(fig_sub, use_container_width=True)

                self._render_export_options(st, st.session_state.order_dist, 'overall_order_distribution')
                if st.session_state.patient_order_dist is not None:
                     self._render_export_options(st, st.session_state.patient_order_dist, 'patient_order_distribution')

        # 4. Order Timing Analysis tab
        with feature_tabs[3]:
            st.markdown("<h3>Analyze Order Timing</h3>", unsafe_allow_html=True)
            st.info("Analyzes order timing relative to admission/discharge.")

            col1, col2 = st.columns(2)
            with col1:
                timing_patient_id_col = st.selectbox("Select Patient ID Column", all_columns, index=all_columns.index(st.session_state.detected_patient_id_col) if st.session_state.detected_patient_id_col in all_columns else 0, key="time_pid")
            with col2:
                default_order_idx = all_columns.index(st.session_state.detected_order_cols[0]) if st.session_state.detected_order_cols and st.session_state.detected_order_cols[0] in all_columns else 0
                timing_order_col = st.selectbox("Select Order Type Column", all_columns, index=default_order_idx, key="time_order")

            col1, col2 = st.columns(2)
            with col1:
                default_time_idx = all_columns.index(st.session_state.detected_time_cols[0]) if st.session_state.detected_time_cols and st.session_state.detected_time_cols[0] in all_columns else 0
                order_time_col = st.selectbox("Select Order Time Column", all_columns, index=default_time_idx, key="time_ordertime")
            with col2:
                admission_time_col = st.selectbox("Select Admission Time Column (Optional)", ["None"] + all_columns, index=0, key="time_admtime")
                admission_time_col = None if admission_time_col == "None" else admission_time_col

            discharge_time_col = st.selectbox("Select Discharge Time Column (Optional)", ["None"] + all_columns, index=0, key="time_dischtime")
            discharge_time_col = None if discharge_time_col == "None" else discharge_time_col

            if st.button("Generate Timing Features"):
                try:
                    with st.spinner("Generating order timing features..."):
                        timing_features = self.feature_engineer.create_order_timing_features(st.session_state.df, timing_patient_id_col, timing_order_col, order_time_col, admission_time_col, discharge_time_col)
                        st.session_state.timing_features = timing_features
                except Exception as e:
                    st.error(f"Error generating timing features: {str(e)}")

            if st.session_state.timing_features is not None:
                st.markdown("<h4>Order Timing Features Preview</h4>", unsafe_allow_html=True)
                st.dataframe(st.session_state.timing_features.head(10), use_container_width=True)
                st.markdown("<h4>Order Timing Visualizations</h4>", unsafe_allow_html=True)

                viz_cols = st.session_state.timing_features.select_dtypes(include=['number']).columns
                if 'total_orders' in viz_cols:
                    fig_total = px.histogram(st.session_state.timing_features, x='total_orders', title="Dist. of Total Orders per Patient")
                    st.plotly_chart(fig_total, use_container_width=True)
                if admission_time_col and 'time_to_first_order_hours' in viz_cols:
                    fig_first = px.histogram(st.session_state.timing_features, x='time_to_first_order_hours', title="Time from Admission to First Order (hrs)")
                    st.plotly_chart(fig_first, use_container_width=True)

                self._render_export_options(st, st.session_state.timing_features, 'order_timing_features')


class ClusteringAnalysisTab:
    """ Encapsulates the UI and logic for the Clustering Analysis tab. """
    def __init__(self, clustering_analyzer, feature_engineer):
        self.clustering_analyzer = clustering_analyzer
        self.feature_engineer = feature_engineer # Needed for saving results

    def render(self, st):
        """ Renders the content of the Clustering Analysis tab. """
        st.markdown("<h2 class='sub-header'>Clustering Analysis</h2>", unsafe_allow_html=True)
        st.info("Apply clustering algorithms to discover patterns and patient groupings. Select data, preprocess, reduce dimensions, run algorithms, and evaluate.")

        clustering_tabs = st.tabs([
            "📋 Data Selection & Preprocessing",
            "📊 Dimensionality Reduction",
            "🔄 K-Means",
            "🌴 Hierarchical",
            "🔍 DBSCAN",
            "📝 LDA Topic Modeling",
            "📈 Evaluation & Comparison"
        ])

        # --- Data Selection Tab ---
        with clustering_tabs[0]:
            st.markdown("<h3>Select Input Data for Clustering</h3>", unsafe_allow_html=True)
            data_source = st.radio("Select Data Source", ["Current DataFrame", "Order Frequency Matrix", "Order Timing Features", "Upload Data"], horizontal=True, key="clust_source")

            input_data = None
            if data_source == "Current DataFrame":
                if st.session_state.df is not None:
                    numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
                    if numeric_cols:
                        selected_cols = st.multiselect("Select numeric columns for clustering", numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))], key="clust_df_cols")
                        if selected_cols:
                            input_data = st.session_state.df[selected_cols].copy()
                    else: st.warning("No numeric columns found in the current DataFrame.")
                else: st.warning("No DataFrame loaded.")
            elif data_source == "Order Frequency Matrix":
                if st.session_state.freq_matrix is not None: input_data = st.session_state.freq_matrix
                else: st.warning("Order frequency matrix not generated.")
            elif data_source == "Order Timing Features":
                if st.session_state.timing_features is not None:
                    numeric_cols = st.session_state.timing_features.select_dtypes(include=['number']).columns.tolist()
                    selected_cols = st.multiselect("Select timing features for clustering", numeric_cols, default=numeric_cols, key="clust_time_cols")
                    if selected_cols: input_data = st.session_state.timing_features[selected_cols].copy()
                else: st.warning("Order timing features not generated.")
            elif data_source == "Upload Data":
                uploaded_file = st.file_uploader("Upload CSV or Parquet file", type=["csv", "parquet"], key="clust_upload")
                if uploaded_file:
                    try:
                        if uploaded_file.name.endswith('.csv'): input_data = pd.read_csv(uploaded_file)
                        elif uploaded_file.name.endswith('.parquet'): input_data = pd.read_parquet(uploaded_file)
                    except Exception as e: st.error(f"Error loading file: {str(e)}")

            if input_data is not None:
                st.markdown(f"Selected data shape: {input_data.shape[0]} rows × {input_data.shape[1]} columns")
                st.dataframe(input_data.head(), use_container_width=True)

                st.markdown("<h4>Data Preprocessing</h4>", unsafe_allow_html=True)
                preprocess_col1, preprocess_col2 = st.columns(2)
                with preprocess_col1: preprocess_method = st.selectbox("Preprocessing Method", ["None", "Standard Scaling", "Min-Max Scaling", "Normalization"], index=1, key="clust_preproc")
                with preprocess_col2: handle_missing = st.selectbox("Handle Missing Values", ["Drop", "Fill with Mean", "Fill with Median", "Fill with Zero"], index=0, key="clust_missing")

                preprocess_method_map = {"None": None, "Standard Scaling": "standard", "Min-Max Scaling": "minmax", "Normalization": "normalize"}
                handle_missing_map = {"Drop": "drop", "Fill with Mean": "mean", "Fill with Median": "median", "Fill with Zero": "zero"}

                if st.button("Prepare Data for Clustering"):
                    try:
                        processed_data = self.clustering_analyzer.preprocess_data(input_data, method=preprocess_method_map[preprocess_method], handle_missing=handle_missing_map[handle_missing])
                        st.session_state.clustering_input_data = processed_data
                        st.success(f"Data preprocessed and ready! Shape: {processed_data.shape}")
                        st.dataframe(processed_data.head(), use_container_width=True)
                    except Exception as e: st.error(f"Error preparing data: {str(e)}")
            else:
                 st.info("Select or upload data to proceed with preprocessing.")


        # --- Dimensionality Reduction Tab ---
        with clustering_tabs[1]:
            st.markdown("<h3>Dimensionality Reduction</h3>", unsafe_allow_html=True)
            if st.session_state.clustering_input_data is not None:
                input_shape = st.session_state.clustering_input_data.shape
                st.markdown(f"<div class='info-box'>Reduce data dimensionality. Current shape: {input_shape[0]} rows × {input_shape[1]} columns</div>", unsafe_allow_html=True)

                reduction_col1, reduction_col2 = st.columns(2)
                with reduction_col1: reduction_method = st.selectbox("Method", ["PCA", "t-SNE", "UMAP", "SVD"], index=0, key="dimred_method")
                with reduction_col2: n_components = st.number_input("Components", min_value=2, max_value=min(10, input_shape[1]), value=2, key="dimred_comp")

                extra_params = {}
                if reduction_method == "t-SNE":
                    tsne_col1, tsne_col2 = st.columns(2)
                    with tsne_col1: perplexity = st.slider("Perplexity", 5, 50, 30, key="tsne_perp")
                    with tsne_col2: learning_rate = st.slider("Learning Rate", 10, 1000, 200, 10, key="tsne_lr")
                    n_iter = st.slider("Max Iterations", 250, 2000, 1000, 250, key="tsne_iter")
                    extra_params = {"perplexity": perplexity, "learning_rate": learning_rate, "n_iter": n_iter}
                elif reduction_method == "UMAP":
                    umap_col1, umap_col2 = st.columns(2)
                    with umap_col1: n_neighbors = st.slider("Neighbors", 2, 100, 15, key="umap_nn")
                    with umap_col2: min_dist = st.slider("Min Distance", 0.0, 0.99, 0.1, 0.05, key="umap_md")
                    metric = st.selectbox("Distance Metric", ["euclidean", "manhattan", "cosine", "correlation"], index=0, key="umap_metric")
                    extra_params = {"n_neighbors": n_neighbors, "min_dist": min_dist, "metric": metric}

                if st.button("Apply Dimensionality Reduction"):
                    try:
                        with st.spinner(f"Applying {reduction_method}..."):
                            method_map = {"PCA": "pca", "t-SNE": "tsne", "UMAP": "umap", "SVD": "svd"}
                            reduced_data = self.clustering_analyzer.apply_dimensionality_reduction(st.session_state.clustering_input_data, method=method_map[reduction_method], n_components=n_components, **extra_params)
                            st.session_state.reduced_data = reduced_data
                            st.success(f"Reduction complete! New shape: {reduced_data.shape}")
                            st.dataframe(reduced_data.head(), use_container_width=True)
                    except Exception as e: st.error(f"Error applying reduction: {str(e)}")

                if st.session_state.reduced_data is not None:
                    reduced_shape = st.session_state.reduced_data.shape
                    st.markdown("<h4>Visualization of Reduced Data</h4>", unsafe_allow_html=True)
                    if reduced_shape[1] == 2:
                        fig = px.scatter(st.session_state.reduced_data, x=st.session_state.reduced_data.columns[0], y=st.session_state.reduced_data.columns[1], title=f"2D Projection ({reduction_method})")
                        st.plotly_chart(fig, use_container_width=True)
                    elif reduced_shape[1] == 3:
                        fig = px.scatter_3d(st.session_state.reduced_data, x=st.session_state.reduced_data.columns[0], y=st.session_state.reduced_data.columns[1], z=st.session_state.reduced_data.columns[2], title=f"3D Projection ({reduction_method})")
                        st.plotly_chart(fig, use_container_width=True)
                    else: st.info("Select 2 or 3 components for visualization.")

                    # Save options (using FeatureEngineeringTab's helper for consistency)
                    fe_tab = FeatureEngineeringTab(self.feature_engineer)
                    fe_tab._render_export_options(st, st.session_state.reduced_data, f"{reduction_method.lower()}_reduced_data")
            else: st.warning("Prepare data in the 'Data Selection' tab first.")


        # --- K-Means Tab ---
        with clustering_tabs[2]:
            st.markdown("<h3>K-Means Clustering</h3>", unsafe_allow_html=True)
            if st.session_state.clustering_input_data is not None:
                use_reduced = st.checkbox("Use reduced data", value=(st.session_state.reduced_data is not None), key="kmeans_reduced")
                data_to_use = st.session_state.reduced_data if use_reduced and st.session_state.reduced_data is not None else st.session_state.clustering_input_data

                st.markdown("<h4>Find Optimal k (Elbow/Silhouette)</h4>", unsafe_allow_html=True)
                k_col1, k_col2, k_col3 = st.columns(3)
                with k_col1: k_min = st.number_input("Min k", 2, 10, 2, key="kmeans_kmin")
                with k_col2: k_max = st.number_input("Max k", 3, 20, 10, key="kmeans_kmax")
                with k_col3: metric = st.selectbox("Metric", ["Silhouette", "Inertia"], key="kmeans_kmetric")

                if st.button("Find Optimal k"):
                    try:
                        with st.spinner("Finding optimal k..."):
                             metric_map = {"Silhouette": "silhouette", "Inertia": "inertia"}
                             optimal_k, k_metrics = self.clustering_analyzer.find_optimal_k_for_kmeans(data_to_use, k_range=range(k_min, k_max + 1), metric=metric_map[metric])
                             st.session_state.optimal_k = optimal_k
                             st.success(f"Optimal k based on {metric}: {optimal_k}")

                             fig = go.Figure()
                             metric_col = 'silhouette' if metric == "Silhouette" else 'inertia'
                             y_label = "Silhouette Score (Higher is Better)" if metric == "Silhouette" else "Inertia (Lower is Better - Elbow)"
                             fig.add_trace(go.Scatter(x=k_metrics['k'], y=k_metrics[metric_col], mode='lines+markers'))
                             fig.add_vline(x=optimal_k, line_dash="dash", line_color="red", annotation_text=f"Optimal k={optimal_k}")
                             fig.update_layout(title=f"{metric} Method for Optimal k", xaxis_title="Number of Clusters (k)", yaxis_title=y_label)
                             st.plotly_chart(fig, use_container_width=True)
                    except Exception as e: st.error(f"Error finding optimal k: {str(e)}")

                st.markdown("<h4>Run K-means</h4>", unsafe_allow_html=True)
                default_k = st.session_state.optimal_k if 'optimal_k' in st.session_state and st.session_state.optimal_k else 5
                n_clusters = st.number_input("Number of Clusters (k)", 2, 20, default_k, key="kmeans_k")
                n_init = st.slider("Initializations", 1, 20, 10, key="kmeans_init")
                max_iter_kmeans = st.slider("Max Iterations", 100, 1000, 300, 100, key="kmeans_iter")

                if st.button("Run K-means Clustering"):
                    try:
                        with st.spinner(f"Running K-means with k={n_clusters}..."):
                            labels, kmeans_model = self.clustering_analyzer.run_kmeans_clustering(data_to_use, n_clusters=n_clusters, n_init=n_init, max_iter=max_iter_kmeans)
                            st.session_state.kmeans_labels = pd.Series(labels, index=data_to_use.index) # Ensure index alignment
                            metrics = self.clustering_analyzer.evaluate_clustering(data_to_use, labels, "kmeans")
                            st.success(f"K-means complete! Silhouette: {metrics['silhouette_score']:.4f}, Davies-Bouldin: {metrics['davies_bouldin_score']:.4f}")

                            cluster_counts = pd.Series(labels).value_counts().sort_index()
                            fig_counts = px.bar(x=cluster_counts.index, y=cluster_counts.values, labels={'x': 'Cluster', 'y': 'Count'}, title="K-means Cluster Sizes")
                            st.plotly_chart(fig_counts, use_container_width=True)

                            if data_to_use.shape[1] == 2:
                                vis_data = data_to_use.copy()
                                vis_data['Cluster'] = labels
                                fig_2d = px.scatter(vis_data, x=vis_data.columns[0], y=vis_data.columns[1], color='Cluster', title="K-means Results (2D)")
                                st.plotly_chart(fig_2d, use_container_width=True)
                            elif data_to_use.shape[1] == 3:
                                vis_data = data_to_use.copy()
                                vis_data['Cluster'] = labels
                                fig_3d = px.scatter_3d(vis_data, x=vis_data.columns[0], y=vis_data.columns[1], z=vis_data.columns[2], color='Cluster', title="K-means Results (3D)")
                                st.plotly_chart(fig_3d, use_container_width=True)

                    except Exception as e: st.error(f"Error running K-means: {str(e)}")

                if 'kmeans_labels' in st.session_state and st.session_state.kmeans_labels is not None:
                    # Save options
                    fe_tab = FeatureEngineeringTab(self.feature_engineer) # Re-use export UI
                    fe_tab._render_export_options(st, pd.DataFrame({'cluster': st.session_state.kmeans_labels}), 'kmeans_cluster_assignments')
                    # Add model saving if needed (requires saving the kmeans_model object)

            else: st.warning("Prepare data in the 'Data Selection' tab first.")


        # --- Hierarchical Tab ---
        with clustering_tabs[3]:
            st.markdown("<h3>Hierarchical Clustering</h3>", unsafe_allow_html=True)
            if st.session_state.clustering_input_data is not None:
                use_reduced_hier = st.checkbox("Use reduced data", value=(st.session_state.reduced_data is not None), key="hier_reduced")
                data_to_use_hier = st.session_state.reduced_data if use_reduced_hier and st.session_state.reduced_data is not None else st.session_state.clustering_input_data

                hier_col1, hier_col2 = st.columns(2)
                with hier_col1: n_clusters_hier = st.number_input("Number of Clusters", 2, 20, 5, key="hier_k")
                with hier_col2: linkage_method = st.selectbox("Linkage Method", ["ward", "complete", "average", "single"], key="hier_link")
                distance_metric = st.selectbox("Distance Metric", ["euclidean", "manhattan", "cosine"], key="hier_metric")
                if linkage_method == "ward" and distance_metric != "euclidean":
                    st.warning("Ward linkage requires Euclidean distance. Using Euclidean.")
                    distance_metric = "euclidean"

                if st.button("Run Hierarchical Clustering"):
                    try:
                        with st.spinner(f"Running Hierarchical Clustering..."):
                             # Limit data size for performance
                            max_samples = 2000
                            if len(data_to_use_hier) > max_samples:
                                st.warning(f"Limiting to {max_samples} samples for Hierarchical clustering.")
                                data_sample = data_to_use_hier.sample(max_samples, random_state=RANDOM_STATE)
                            else:
                                data_sample = data_to_use_hier

                            labels_hier, linkage_data = self.clustering_analyzer.run_hierarchical_clustering(data_sample, n_clusters=n_clusters_hier, linkage_method=linkage_method, distance_metric=distance_metric)
                            # Store labels aligned with the *sampled* data index
                            st.session_state.hierarchical_labels = pd.Series(labels_hier, index=data_sample.index)
                            metrics_hier = self.clustering_analyzer.evaluate_clustering(data_sample, labels_hier, "hierarchical")
                            st.success(f"Hierarchical complete! Silhouette: {metrics_hier['silhouette_score']:.4f}, Davies-Bouldin: {metrics_hier['davies_bouldin_score']:.4f}")

                            cluster_counts_hier = pd.Series(labels_hier).value_counts().sort_index()
                            fig_counts_hier = px.bar(x=cluster_counts_hier.index, y=cluster_counts_hier.values, labels={'x': 'Cluster', 'y': 'Count'}, title="Hierarchical Cluster Sizes")
                            st.plotly_chart(fig_counts_hier, use_container_width=True)

                            # Dendrogram
                            st.markdown("<h4>Dendrogram</h4>", unsafe_allow_html=True)
                            fig_dendro, ax_dendro = plt.subplots(figsize=(12, 7))
                            dendrogram(linkage_data['linkage_matrix'], ax=ax_dendro, truncate_mode='lastp', p=12, leaf_rotation=90., leaf_font_size=8., show_contracted=True)
                            ax_dendro.set_title('Hierarchical Clustering Dendrogram')
                            ax_dendro.set_xlabel('Sample index or (cluster size)')
                            ax_dendro.set_ylabel('Distance')
                            st.pyplot(fig_dendro)

                    except Exception as e: st.error(f"Error running Hierarchical: {str(e)}")

                if 'hierarchical_labels' in st.session_state and st.session_state.hierarchical_labels is not None:
                    fe_tab = FeatureEngineeringTab(self.feature_engineer)
                    fe_tab._render_export_options(st, pd.DataFrame({'cluster': st.session_state.hierarchical_labels}), 'hierarchical_cluster_assignments')

            else: st.warning("Prepare data in the 'Data Selection' tab first.")


        # --- DBSCAN Tab ---
        with clustering_tabs[4]:
            st.markdown("<h3>DBSCAN Clustering</h3>", unsafe_allow_html=True)
            if st.session_state.clustering_input_data is not None:
                use_reduced_db = st.checkbox("Use reduced data", value=(st.session_state.reduced_data is not None), key="db_reduced")
                data_to_use_db = st.session_state.reduced_data if use_reduced_db and st.session_state.reduced_data is not None else st.session_state.clustering_input_data

                st.markdown("<h4>Find Optimal Epsilon (ε) using k-distance graph</h4>", unsafe_allow_html=True)
                k_dist = st.slider("k for k-distance", 2, 20, max(2, min(10, data_to_use_db.shape[1] * 2)), key="db_kdist") # Heuristic default
                if st.button("Find Optimal Epsilon (ε)"):
                    try:
                        with st.spinner("Calculating k-distance graph..."):
                            suggested_eps, k_distances = self.clustering_analyzer.find_optimal_eps_for_dbscan(data_to_use_db, k_dist=k_dist)
                            st.session_state.optimal_eps = suggested_eps
                            st.success(f"Suggested epsilon (ε) based on k={k_dist}: {suggested_eps:.4f}")

                            fig_kdist = go.Figure()
                            fig_kdist.add_trace(go.Scatter(x=np.arange(len(k_distances)), y=k_distances, mode='lines'))
                            knee_idx = np.argmax(np.diff(k_distances)) + 1 # Simple knee detection
                            fig_kdist.add_trace(go.Scatter(x=[knee_idx], y=[k_distances[knee_idx]], mode='markers', marker=dict(color='red', size=10), name=f'Suggested ε ≈ {k_distances[knee_idx]:.4f}'))
                            fig_kdist.update_layout(title=f"{k_dist}-Distance Graph", xaxis_title="Points sorted by distance", yaxis_title=f"{k_dist}-distance")
                            st.plotly_chart(fig_kdist, use_container_width=True)
                    except Exception as e: st.error(f"Error finding optimal epsilon: {str(e)}")

                st.markdown("<h4>Run DBSCAN</h4>", unsafe_allow_html=True)
                db_col1, db_col2 = st.columns(2)
                default_eps = st.session_state.optimal_eps if 'optimal_eps' in st.session_state and st.session_state.optimal_eps else 0.5
                with db_col1: eps = st.number_input("Epsilon (ε)", 0.01, 5.0, default_eps, 0.05, key="db_eps")
                with db_col2: min_samples = st.number_input("Minimum Samples", 2, 100, k_dist, key="db_minpts") # Use k_dist as default
                metric_db = st.selectbox("Distance Metric", ["euclidean", "manhattan", "cosine"], key="db_metric")

                if st.button("Run DBSCAN Clustering"):
                    try:
                        with st.spinner(f"Running DBSCAN..."):
                            labels_db, dbscan_model = self.clustering_analyzer.run_dbscan_clustering(data_to_use_db, eps=eps, min_samples=min_samples, metric=metric_db)
                            st.session_state.dbscan_labels = pd.Series(labels_db, index=data_to_use_db.index) # Align index
                            n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
                            n_noise = list(labels_db).count(-1)
                            st.success(f"DBSCAN complete! Found {n_clusters_db} clusters and {n_noise} noise points.")

                            if n_clusters_db > 1:
                                non_noise_mask = labels_db != -1
                                metrics_db = self.clustering_analyzer.evaluate_clustering(data_to_use_db[non_noise_mask], labels_db[non_noise_mask], "dbscan")
                                st.markdown(f"Metrics (excluding noise): Silhouette: {metrics_db['silhouette_score']:.4f}, Davies-Bouldin: {metrics_db['davies_bouldin_score']:.4f}")
                            else: st.warning("Metrics require at least 2 clusters (excluding noise).")

                            cluster_counts_db = pd.Series(labels_db).value_counts().sort_index()
                            fig_counts_db = px.bar(x=cluster_counts_db.index.map(lambda x: "Noise" if x==-1 else f"Cluster {x}"), y=cluster_counts_db.values, labels={'x': 'Cluster', 'y': 'Count'}, title="DBSCAN Cluster Sizes")
                            st.plotly_chart(fig_counts_db, use_container_width=True)

                            # Visualization (similar to K-means)
                            if data_to_use_db.shape[1] == 2:
                                vis_data_db = data_to_use_db.copy()
                                vis_data_db['Cluster'] = pd.Series(labels_db, index=data_to_use_db.index).map(lambda x: "Noise" if x == -1 else f"Cluster {x}")
                                fig_2d_db = px.scatter(vis_data_db, x=vis_data_db.columns[0], y=vis_data_db.columns[1], color='Cluster', title="DBSCAN Results (2D)")
                                st.plotly_chart(fig_2d_db, use_container_width=True)
                            elif data_to_use_db.shape[1] == 3:
                                vis_data_db = data_to_use_db.copy()
                                vis_data_db['Cluster'] = pd.Series(labels_db, index=data_to_use_db.index).map(lambda x: "Noise" if x == -1 else f"Cluster {x}")
                                fig_3d_db = px.scatter_3d(vis_data_db, x=vis_data_db.columns[0], y=vis_data_db.columns[1], z=vis_data_db.columns[2], color='Cluster', title="DBSCAN Results (3D)")
                                st.plotly_chart(fig_3d_db, use_container_width=True)

                    except Exception as e: st.error(f"Error running DBSCAN: {str(e)}")

                if 'dbscan_labels' in st.session_state and st.session_state.dbscan_labels is not None:
                    fe_tab = FeatureEngineeringTab(self.feature_engineer)
                    fe_tab._render_export_options(st, pd.DataFrame({'cluster': st.session_state.dbscan_labels}), 'dbscan_cluster_assignments')

            else: st.warning("Prepare data in the 'Data Selection' tab first.")


        # --- LDA Tab ---
        with clustering_tabs[5]:
            st.markdown("<h3>LDA Topic Modeling</h3>", unsafe_allow_html=True)
            st.info("Discover latent topics in text data (e.g., order sequences or text columns).")

            documents = None
            lda_data_source = st.radio("Text Data Source", ["Order Sequences", "Text Column in Dataset"], key="lda_source")
            if lda_data_source == "Order Sequences":
                if st.session_state.order_sequences is not None:
                    documents = [" ".join(map(str, seq)) for seq in st.session_state.order_sequences.values()]
                    st.markdown(f"Using {len(documents)} patient order sequences.")
                else: st.warning("Generate order sequences first.")
            else: # Text Column
                if st.session_state.df is not None:
                    text_cols = st.session_state.df.select_dtypes(include=['object']).columns.tolist()
                    if text_cols:
                        text_col = st.selectbox("Select Text Column", text_cols, key="lda_textcol")
                        documents = st.session_state.df[text_col].fillna("").astype(str).tolist()
                    else: st.warning("No text columns found.")
                else: st.warning("Load a dataset first.")

            if documents:
                st.markdown("<h4>LDA Parameters</h4>", unsafe_allow_html=True)
                lda_col1, lda_col2 = st.columns(2)
                with lda_col1: n_topics = st.number_input("Number of Topics", 2, 20, 5, key="lda_k")
                with lda_col2: max_iter_lda = st.slider("Max Iterations", 10, 1000, 100, 10, key="lda_iter")
                lda_col3, lda_col4 = st.columns(2)
                with lda_col3: vectorizer_type = st.selectbox("Vectorizer", ["Count", "TF-IDF"], key="lda_vec")
                with lda_col4: max_features = st.number_input("Max Features", 100, 10000, 1000, 100, key="lda_feat")

                if st.button("Run LDA Topic Modeling"):
                    try:
                        with st.spinner(f"Running LDA with {n_topics} topics..."):
                            vec_map = {"Count": "count", "TF-IDF": "tfidf"}
                            lda_model, doc_topic, topic_term = self.clustering_analyzer.run_lda_topic_modeling(documents, n_topics=n_topics, vectorizer_type=vec_map[vectorizer_type], max_features=max_features, max_iter=max_iter_lda)
                            st.session_state.lda_results = {'doc_topic_matrix': doc_topic, 'topic_term_matrix': topic_term}
                            st.success("LDA complete!")

                            top_terms = self.clustering_analyzer.get_top_terms_per_topic(topic_term, n_terms=10)
                            st.markdown("<h4>Top Terms per Topic</h4>", unsafe_allow_html=True)
                            st.dataframe(top_terms, use_container_width=True)

                            st.markdown("<h4>Document-Topic Distribution (Sample)</h4>", unsafe_allow_html=True)
                            sample_size = min(20, doc_topic.shape[0])
                            fig_doc_topic = px.imshow(doc_topic.iloc[:sample_size], labels=dict(x="Topic", y="Document", color="Prob."), title=f"Doc-Topic Dist (Sample {sample_size})", color_continuous_scale="Viridis")
                            st.plotly_chart(fig_doc_topic, use_container_width=True)

                    except Exception as e: st.error(f"Error running LDA: {str(e)}")

                if 'lda_results' in st.session_state and st.session_state.lda_results:
                    fe_tab = FeatureEngineeringTab(self.feature_engineer)
                    fe_tab._render_export_options(st, st.session_state.lda_results['doc_topic_matrix'], 'lda_topic_distributions')
                    # Add model saving if needed

            else: st.warning("Select a valid text data source.")


        # --- Evaluation Tab ---
        with clustering_tabs[6]:
            st.markdown("<h3>Evaluation & Comparison</h3>", unsafe_allow_html=True)
            st.info("Compare performance metrics and cluster assignments across different algorithms.")

            metrics_data = {'Algorithm': [], 'Num Clusters': [], 'Silhouette': [], 'Davies-Bouldin': [], 'Calinski-Harabasz': []}
            results_available = []

            # Helper to add metrics
            def add_metrics(algo_name, labels_key, data_key='clustering_input_data'):
                if labels_key in st.session_state and st.session_state[labels_key] is not None:
                    results_available.append(algo_name)
                    labels = st.session_state[labels_key]
                    # Use reduced data if it was likely used for clustering, otherwise original preprocessed data
                    data_used = st.session_state.get('reduced_data') if st.session_state.get(f'{algo_name.lower()}_reduced') else st.session_state.get(data_key)

                    if data_used is not None:
                        # Align data and labels by index before evaluation
                        common_index = data_used.index.intersection(labels.index)
                        if not common_index.empty:
                            data_aligned = data_used.loc[common_index]
                            labels_aligned = labels.loc[common_index]

                            # Exclude noise for DBSCAN metrics
                            is_dbscan = algo_name == 'DBSCAN'
                            non_noise_mask = (labels_aligned != -1) if is_dbscan else pd.Series(True, index=labels_aligned.index)
                            num_clusters = len(set(labels_aligned[non_noise_mask]))

                            if num_clusters > 1:
                                metrics = self.clustering_analyzer.evaluate_clustering(data_aligned[non_noise_mask], labels_aligned[non_noise_mask], algo_name.lower())
                                metrics_data['Algorithm'].append(algo_name)
                                metrics_data['Num Clusters'].append(num_clusters)
                                metrics_data['Silhouette'].append(f"{metrics.get('silhouette_score', np.nan):.4f}")
                                metrics_data['Davies-Bouldin'].append(f"{metrics.get('davies_bouldin_score', np.nan):.4f}")
                                metrics_data['Calinski-Harabasz'].append(f"{metrics.get('calinski_harabasz_score', np.nan):.0f}")
                            else:
                                metrics_data['Algorithm'].append(algo_name)
                                metrics_data['Num Clusters'].append(num_clusters)
                                metrics_data['Silhouette'].append('N/A')
                                metrics_data['Davies-Bouldin'].append('N/A')
                                metrics_data['Calinski-Harabasz'].append('N/A')
                        else:
                            st.warning(f"Index mismatch between data and labels for {algo_name}. Cannot calculate metrics.")


            add_metrics('K-means', 'kmeans_labels')
            add_metrics('Hierarchical', 'hierarchical_labels')
            add_metrics('DBSCAN', 'dbscan_labels')

            if results_available:
                metrics_df = pd.DataFrame(metrics_data)
                st.markdown("<h4>Performance Metrics</h4>", unsafe_allow_html=True)
                st.dataframe(metrics_df, use_container_width=True)

                st.markdown("<h4>Metric Comparison</h4>", unsafe_allow_html=True)
                # Add bar charts comparing metrics if desired (similar to original code)

                st.markdown("<h4>Cluster Assignment Comparison</h4>", unsafe_allow_html=True)
                if len(results_available) >= 2:
                    compare_col1, compare_col2 = st.columns(2)
                    with compare_col1: algo1 = st.selectbox("Algorithm 1", results_available, key="comp_a1")
                    with compare_col2: algo2 = st.selectbox("Algorithm 2", results_available, index=min(1, len(results_available)-1), key="comp_a2")

                    if algo1 != algo2:
                        labels1 = st.session_state.get(f'{algo1.lower()}_labels')
                        labels2 = st.session_state.get(f'{algo2.lower()}_labels')

                        if labels1 is not None and labels2 is not None:
                             # Align labels based on common index before comparison
                            common_index = labels1.index.intersection(labels2.index)
                            if not common_index.empty:
                                labels1_aligned = labels1.loc[common_index]
                                labels2_aligned = labels2.loc[common_index]

                                # Exclude noise if DBSCAN is involved
                                mask1 = (labels1_aligned != -1) if algo1 == 'DBSCAN' else pd.Series(True, index=common_index)
                                mask2 = (labels2_aligned != -1) if algo2 == 'DBSCAN' else pd.Series(True, index=common_index)
                                combined_mask = mask1 & mask2

                                if combined_mask.sum() > 0:
                                    compare_labels1 = labels1_aligned[combined_mask]
                                    compare_labels2 = labels2_aligned[combined_mask]

                                    ari = adjusted_rand_score(compare_labels1, compare_labels2)
                                    ami = adjusted_mutual_info_score(compare_labels1, compare_labels2)
                                    st.markdown(f"**Agreement between {algo1} & {algo2} (excluding noise):**")
                                    st.markdown(f"- Adjusted Rand Index (ARI): {ari:.4f}")
                                    st.markdown(f"- Adjusted Mutual Info (AMI): {ami:.4f}")

                                    contingency = pd.crosstab(compare_labels1, compare_labels2, rownames=[f"{algo1} Clusters"], colnames=[f"{algo2} Clusters"])
                                    st.dataframe(contingency, use_container_width=True)
                                else:
                                    st.warning("No common non-noise points to compare.")
                            else:
                                st.warning("Labels have non-overlapping indices, cannot compare.")
                    else:
                        st.warning("Select two different algorithms to compare.")
                else:
                    st.info("Run at least two clustering algorithms to compare results.")
            else:
                st.warning("No clustering results available to evaluate.")


class AnalysisVisualizationTab:
    """ Encapsulates the UI and logic for the Analysis & Visualization tab. """
    def __init__(self, cluster_analyzer):
        self.cluster_analyzer = cluster_analyzer

    def render(self, st):
        """ Renders the content of the Analysis & Visualization tab. """
        st.markdown("<h2 class='sub-header'>Cluster Analysis & Visualization</h2>", unsafe_allow_html=True)
        st.info("Analyze identified clusters: compare length of stay, visualize order patterns, perform statistical tests, characterize clusters, and find important features.")

        # Check if clustering results are available
        available_results = []
        if 'kmeans_labels' in st.session_state and st.session_state.kmeans_labels is not None: available_results.append("K-means")
        if 'hierarchical_labels' in st.session_state and st.session_state.hierarchical_labels is not None: available_results.append("Hierarchical")
        if 'dbscan_labels' in st.session_state and st.session_state.dbscan_labels is not None: available_results.append("DBSCAN")

        if not available_results:
            st.warning("No clustering results found. Please run clustering algorithms in the 'Clustering Analysis' tab first.")
            return

        selected_clustering = st.selectbox("Select Clustering Results to Analyze", available_results, key="analysis_algo")

        # Get the corresponding labels and data
        cluster_labels = None
        if selected_clustering == "K-means": cluster_labels = st.session_state.kmeans_labels
        elif selected_clustering == "Hierarchical": cluster_labels = st.session_state.hierarchical_labels
        elif selected_clustering == "DBSCAN": cluster_labels = st.session_state.dbscan_labels

        # Determine which data was likely used for the selected clustering
        # This assumes a checkbox like 'use_reduced_data' was used and its state stored,
        # otherwise defaults to checking if reduced_data exists. A more robust approach
        # might store the data used alongside the labels.
        data_key_used = 'reduced_data' if st.session_state.get(f'{selected_clustering.lower()}_reduced', st.session_state.reduced_data is not None) else 'clustering_input_data'
        analysis_data_clustered = st.session_state.get(data_key_used)

        if analysis_data_clustered is None or cluster_labels is None:
             st.error(f"Could not find the data or labels used for {selected_clustering}. Ensure data was prepared and clustering was run.")
             return

        # Align data and labels using common index
        common_index = analysis_data_clustered.index.intersection(cluster_labels.index)
        if common_index.empty:
            st.error("Index mismatch between analysis data and cluster labels. Cannot proceed.")
            return

        analysis_data_aligned = analysis_data_clustered.loc[common_index].copy()
        cluster_labels_aligned = cluster_labels.loc[common_index]
        analysis_data_aligned['cluster'] = cluster_labels_aligned


        analysis_tabs = st.tabs([
            "🏥 Length of Stay",
            "📈 Order Patterns",
            "🔬 Statistical Tests",
            "👤 Cluster Profiles",
            "🔥 Feature Importance"
        ])

        # --- Length of Stay Tab ---
        with analysis_tabs[0]:
            st.markdown("<h3>Length of Stay Analysis by Cluster</h3>", unsafe_allow_html=True)
            if st.session_state.df is not None:
                # Try to find suitable columns automatically or let user select
                all_cols = st.session_state.df.columns.tolist()
                time_cols = st.session_state.df.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns.tolist()
                # Add potential date-like object columns
                for col in st.session_state.df.select_dtypes(include=['object']).columns:
                     try:
                         pd.to_datetime(st.session_state.df[col].dropna().iloc[:5])
                         if col not in time_cols: time_cols.append(col)
                     except (ValueError, TypeError): continue

                if time_cols:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        pid_idx = all_cols.index(st.session_state.detected_patient_id_col) if st.session_state.detected_patient_id_col in all_cols else 0
                        patient_id_col = st.selectbox("Patient ID Col", all_cols, index=pid_idx, key="los_pid")
                    with col2:
                        adm_idx = next((i for i, c in enumerate(time_cols) if 'admit' in c.lower() or 'start' in c.lower()), 0)
                        admission_col = st.selectbox("Admission Time Col", time_cols, index=adm_idx, key="los_adm")
                    with col3:
                        dis_idx = next((i for i, c in enumerate(time_cols) if 'disch' in c.lower() or 'end' in c.lower()), min(1, len(time_cols)-1))
                        discharge_col = st.selectbox("Discharge Time Col", time_cols, index=dis_idx, key="los_dis")

                    if st.button("Calculate & Compare Length of Stay"):
                        try:
                            with st.spinner("Calculating LOS..."):
                                df_copy = st.session_state.df.copy() # Work on a copy
                                # Ensure datetime conversion
                                df_copy[admission_col] = pd.to_datetime(df_copy[admission_col], errors='coerce')
                                df_copy[discharge_col] = pd.to_datetime(df_copy[discharge_col], errors='coerce')

                                # Calculate LOS (ensure patient IDs match between df and cluster labels)
                                los_series = self.cluster_analyzer.calculate_length_of_stay(df_copy, admission_col, discharge_col, patient_id_col)

                                # Merge LOS with cluster labels based on patient ID
                                # Ensure cluster_labels_aligned index is the patient ID
                                if cluster_labels_aligned.index.name != patient_id_col:
                                     st.warning(f"Cluster labels index name ('{cluster_labels_aligned.index.name}') does not match selected Patient ID ('{patient_id_col}'). Attempting merge anyway...")
                                     # This might fail if indices are not compatible

                                # Ensure los_series index is the patient ID
                                if los_series.index.name != patient_id_col:
                                     los_series = los_series.rename_axis(patient_id_col)


                                # Combine LOS and Cluster Info
                                los_cluster_df = pd.merge(
                                    los_series.rename('length_of_stay'),
                                    cluster_labels_aligned.rename('cluster'),
                                    left_index=True,
                                    right_index=True,
                                    how='inner' # Only keep patients with both LOS and cluster label
                                )


                                if los_cluster_df.empty:
                                     st.error("Could not merge LOS data with cluster labels. Check Patient ID columns and data alignment.")
                                else:
                                    st.session_state.length_of_stay_comparison = los_cluster_df # Store for potential reuse
                                    st.success(f"LOS calculated and merged for {len(los_cluster_df)} patients.")

                                    # Display comparison (mean/median)
                                    los_summary = los_cluster_df.groupby('cluster')['length_of_stay'].agg(['mean', 'median', 'std', 'count']).reset_index()
                                    st.dataframe(los_summary, use_container_width=True)

                                    # Boxplot
                                    fig_los = px.box(los_cluster_df, x='cluster', y='length_of_stay', title="Length of Stay by Cluster", labels={'cluster': 'Cluster', 'length_of_stay': 'LOS (days)'}, color='cluster')
                                    st.plotly_chart(fig_los, use_container_width=True)

                                    # Statistical Test (ANOVA/Kruskal-Wallis)
                                    st.markdown("<h4>Statistical Test for LOS Differences</h4>", unsafe_allow_html=True)
                                    groups = [group["length_of_stay"].dropna() for name, group in los_cluster_df.groupby('cluster') if name != -1] # Exclude noise
                                    if len(groups) >= 2:
                                        try:
                                            # Check normality (optional, guide choice)
                                            # Shapiro-Wilk test for normality (example)
                                            # normality_p = [stats.shapiro(g).pvalue for g in groups if len(g) >= 3] # Shapiro needs >= 3 samples
                                            # if all(p > 0.05 for p in normality_p): # If all normal, use ANOVA
                                            #     stat, p_val = stats.f_oneway(*groups)
                                            #     test_name = "ANOVA"
                                            # else: # If any non-normal, use Kruskal-Wallis
                                            #     stat, p_val = stats.kruskal(*groups)
                                            #     test_name = "Kruskal-Wallis"

                                            # Default to Kruskal-Wallis as it's non-parametric
                                            stat, p_val = stats.kruskal(*[g for g in groups if len(g) > 0]) # Ensure groups are not empty
                                            test_name = "Kruskal-Wallis"

                                            st.markdown(f"**{test_name} Test:** Statistic = {stat:.4f}, P-value = {p_val:.4g}")
                                            st.markdown(f"**Significant difference (p < 0.05): {'Yes' if p_val < 0.05 else 'No'}**")
                                        except ValueError as ve:
                                             st.warning(f"Could not perform statistical test: {ve}") # Handle cases with insufficient data per group
                                    else:
                                        st.info("Need at least two non-noise clusters for statistical comparison.")

                        except Exception as e:
                            st.error(f"Error calculating LOS: {str(e)}")
                            logging.exception("LOS Calculation Error") # Log traceback
                else:
                    st.warning("No suitable datetime columns found for LOS calculation.")
            else:
                st.warning("Load the base dataset first to calculate Length of Stay.")


        # --- Order Patterns Tab ---
        with analysis_tabs[1]:
            st.markdown("<h3>Order Pattern Visualization</h3>", unsafe_allow_html=True)
            if 'order_sequences' in st.session_state and st.session_state.order_sequences:
                st.info("Visualize common order sequences within each cluster.")
                # Ensure cluster labels are aligned with order sequence keys (patient IDs)
                seq_keys = list(st.session_state.order_sequences.keys())
                common_keys = list(set(seq_keys) & set(cluster_labels_aligned.index))

                if not common_keys:
                    st.warning("No common patient IDs between order sequences and selected cluster labels.")
                else:
                    aligned_sequences = {k: st.session_state.order_sequences[k] for k in common_keys}
                    aligned_labels = cluster_labels_aligned.loc[common_keys]

                    max_orders_viz = st.slider("Max Order Types to Show", 5, 50, 20, key="pattern_orders")
                    max_patients_viz = st.slider("Max Patients per Cluster", 10, 100, 30, key="pattern_patients")

                    if st.button("Generate Order Pattern Visualization"):
                        try:
                            with st.spinner("Generating visualization..."):
                                fig_patterns = self.cluster_analyzer.visualize_order_patterns(aligned_sequences, aligned_labels, max_orders=max_orders_viz, max_patients=max_patients_viz)
                                st.plotly_chart(fig_patterns, use_container_width=True)
                                # Add Sankey diagram generation here if desired (from original code)
                        except Exception as e:
                            st.error(f"Error generating pattern visualization: {str(e)}")
            else:
                st.warning("Generate 'Temporal Order Sequences' in the 'Feature Engineering' tab first.")


        # --- Statistical Tests Tab ---
        with analysis_tabs[2]:
            st.markdown("<h3>Statistical Testing Between Clusters</h3>", unsafe_allow_html=True)
            st.info("Identify features that significantly differ across clusters.")

            feature_cols_test = [col for col in analysis_data_aligned.columns if col != 'cluster' and pd.api.types.is_numeric_dtype(analysis_data_aligned[col])]
            if not feature_cols_test:
                st.warning("No numeric features available in the selected analysis data for testing.")
            else:
                selected_features_test = st.multiselect("Select Features for Testing", feature_cols_test, default=feature_cols_test[:min(10, len(feature_cols_test))], key="stat_feats")
                test_method = st.radio("Test Method", ["ANOVA (parametric)", "Kruskal-Wallis (non-parametric)"], index=1, horizontal=True, key="stat_method")
                method_map = {"ANOVA (parametric)": "anova", "Kruskal-Wallis (non-parametric)": "kruskal"}

                if st.button("Run Statistical Tests") and selected_features_test:
                    try:
                        with st.spinner("Running tests..."):
                            results_df = self.cluster_analyzer.statistical_testing(analysis_data_aligned, selected_features_test, cluster_col='cluster', method=method_map[test_method])
                            st.success("Statistical tests complete!")
                            st.markdown("<h4>Test Results (Adjusted p-values)</h4>", unsafe_allow_html=True)

                            # Format and display results table
                            results_display = results_df.copy()
                            results_display['P-Value'] = results_display['P-Value'].apply(lambda x: f"{x:.4g}" if pd.notnull(x) else "N/A")
                            results_display['Adjusted P-Value'] = results_display['Adjusted P-Value'].apply(lambda x: f"{x:.4g}" if pd.notnull(x) else "N/A")

                            def highlight_sig(val):
                                return 'background-color: #d4efdf' if val == True else ''
                            styled_results = results_display.style.applymap(highlight_sig, subset=['Significant (Adjusted)'])
                            st.dataframe(styled_results, use_container_width=True)

                            # P-value visualization (Manhattan-like plot)
                            st.markdown("<h4>Feature Significance (-log10 Adjusted P-Value)</h4>", unsafe_allow_html=True)
                            plot_data = results_df[results_df['Adjusted P-Value'].notna()].copy()
                            plot_data['-log10(adj_p)'] = -np.log10(plot_data['Adjusted P-Value'] + 1e-10) # Add epsilon for p=0
                            plot_data = plot_data.sort_values('-log10(adj_p)', ascending=False)
                            sig_threshold = -np.log10(0.05)

                            fig_pval = px.bar(plot_data, x='Feature', y='-log10(adj_p)', color='Significant (Adjusted)',
                                              color_discrete_map={True: 'red', False: 'grey'},
                                              title="Feature Significance",
                                              labels={'-log10(adj_p)': '-log10 (Adjusted P-value)'})
                            fig_pval.add_hline(y=sig_threshold, line_dash="dash", annotation_text="p=0.05", annotation_position="bottom right")
                            fig_pval.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig_pval, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error performing statistical tests: {str(e)}")


        # --- Cluster Profiles Tab ---
        with analysis_tabs[3]:
            st.markdown("<h3>Cluster Characterization Profiles</h3>", unsafe_allow_html=True)
            st.info("Generate descriptive statistics for key features within each cluster.")

            profile_feature_cols = [col for col in analysis_data_aligned.columns if col != 'cluster']
            selected_features_profile = st.multiselect("Select Features for Profiles", profile_feature_cols, default=profile_feature_cols[:min(10, len(profile_feature_cols))], key="profile_feats")

            if st.button("Generate Cluster Profiles") and selected_features_profile:
                try:
                    with st.spinner("Generating profiles..."):
                        characterization = self.cluster_analyzer.generate_cluster_characterization(analysis_data_aligned, cluster_col='cluster', important_features=selected_features_profile)
                        st.success("Profiles generated!")

                        cluster_tabs_profile = st.tabs([f"Cluster {c}" for c in sorted(characterization.keys())])
                        for i, cluster_name in enumerate(sorted(characterization.keys())):
                            with cluster_tabs_profile[i]:
                                stats = characterization[cluster_name]
                                st.markdown(f"**Size:** {stats['size']} ({stats['percentage']:.1f}%)")
                                st.markdown("##### Feature Statistics")
                                feature_stats_list = []
                                for feature, f_stats in stats['features'].items():
                                     f_stats['Feature'] = feature
                                     feature_stats_list.append(f_stats)
                                feature_df = pd.DataFrame(feature_stats_list)
                                # Reorder columns for better readability
                                cols_order = ['Feature', 'mean', 'median', 'std', 'min', 'max', 'diff_from_mean']
                                feature_df = feature_df[[col for col in cols_order if col in feature_df.columns]]
                                feature_df = feature_df.rename(columns={'diff_from_mean': 'Diff from Mean (%)'})
                                # Format and display
                                styled_df = feature_df.style.format({
                                    'mean': '{:.2f}', 'median': '{:.2f}', 'std': '{:.2f}',
                                    'min': '{:.2f}', 'max': '{:.2f}', 'Diff from Mean (%)': '{:+.1f}%'
                                }).background_gradient(cmap='coolwarm', subset=['Diff from Mean (%)'], vmin=-100, vmax=100)

                                st.dataframe(styled_df, use_container_width=True)

                                # Radar Chart (optional, can be added back from original code if needed)

                    # HTML Report Generation
                    st.markdown("--- \n ### Generate HTML Report")
                    report_title = st.text_input("Report Title", f"{selected_clustering} Cluster Analysis Report", key="report_title")
                    include_plots = st.checkbox("Include Visualizations in Report", True, key="report_plots")
                    if st.button("Generate & Download HTML Report"):
                         try:
                              html_content = self.cluster_analyzer.generate_html_report(title=report_title, include_plots=include_plots)
                              st.download_button(
                                   label="Download Report",
                                   data=html_content,
                                   file_name=f"{report_title.replace(' ', '_').lower()}_{datetime.datetime.now().strftime('%Y%m%d')}.html",
                                   mime="text/html"
                              )
                         except Exception as e:
                              st.error(f"Failed to generate report: {e}")

                except Exception as e:
                    st.error(f"Error generating profiles: {str(e)}")


        # --- Feature Importance Tab ---
        with analysis_tabs[4]:
            st.markdown("<h3>Feature Importance Analysis</h3>", unsafe_allow_html=True)
            st.info("Identify features most influential in separating the clusters (using Random Forest classifier).")

            if st.button("Calculate Feature Importance"):
                try:
                    with st.spinner("Calculating importance..."):
                        # Ensure data is numeric and labels are present
                        numeric_data = analysis_data_aligned.select_dtypes(include=np.number)
                        if 'cluster' not in numeric_data.columns:
                             st.error("Cluster column not found in numeric data.")
                             return

                        # Exclude noise points for importance calculation if DBSCAN
                        if selected_clustering == "DBSCAN":
                             numeric_data = numeric_data[numeric_data['cluster'] != -1]

                        if numeric_data['cluster'].nunique() < 2:
                             st.warning("Need at least 2 clusters (excluding noise) to calculate feature importance.")
                             return

                        importance_df = self.cluster_analyzer.calculate_feature_importance(numeric_data, cluster_col='cluster')
                        st.success("Feature importance calculated!")
                        st.markdown("<h4>Feature Importance Ranking</h4>", unsafe_allow_html=True)
                        st.dataframe(importance_df.style.format({'Importance': '{:.4f}'}), use_container_width=True)

                        # Bar chart
                        fig_imp = px.bar(importance_df.head(20), y='Feature', x='Importance', orientation='h', title="Top 20 Important Features", color='Importance', color_continuous_scale='Viridis')
                        fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig_imp, use_container_width=True)

                        # Heatmap of feature values (optional, can be added back)

                except Exception as e:
                    st.error(f"Error calculating feature importance: {str(e)}")


# --- Main App Class ---

class MIMICDashboardApp:
    def __init__(self):
        logging.info("Initializing MIMICDashboardApp...")
        # Initialize core components
        self.data_handler = MIMICDataLoader()
        self.visualizer = MIMICVisualizer()
        self.feature_engineer = MIMICFeatureEngineer()
        self.clustering_analyzer = MIMICClusteringAnalysis() # For running algorithms
        self.cluster_analyzer = MIMICClusterAnalyzer()       # For analyzing results
        self.filtering_tab = FilteringTab()

        # Initialize tab view components
        self.feature_engineering_tab_view = FeatureEngineeringTab(self.feature_engineer)
        self.clustering_analysis_tab_view = ClusteringAnalysisTab(self.clustering_analyzer, self.feature_engineer)
        self.analysis_visualization_tab_view = AnalysisVisualizationTab(self.cluster_analyzer)

        self.init_session_state()
        logging.info("MIMICDashboardApp initialized.")


    @staticmethod
    def init_session_state():
        """ Function to initialize session state """
        logging.info("Initializing session state...")
        # Core states
        if 'loader' not in st.session_state: st.session_state.loader = None
        if 'datasets' not in st.session_state: st.session_state.datasets = {}
        if 'selected_module' not in st.session_state: st.session_state.selected_module = None
        if 'selected_table' not in st.session_state: st.session_state.selected_table = None
        if 'df' not in st.session_state: st.session_state.df = None
        if 'sample_size' not in st.session_state: st.session_state.sample_size = DEFAULT_SAMPLE_SIZE
        if 'available_tables' not in st.session_state: st.session_state.available_tables = {}
        if 'file_paths' not in st.session_state: st.session_state.file_paths = {}
        if 'file_sizes' not in st.session_state: st.session_state.file_sizes = {}
        if 'table_display_names' not in st.session_state: st.session_state.table_display_names = {}
        if 'current_file_path' not in st.session_state: st.session_state.current_file_path = None
        if 'mimic_path' not in st.session_state: st.session_state.mimic_path = DEFAULT_MIMIC_PATH
        if 'total_row_count' not in st.session_state: st.session_state.total_row_count = 0
        if 'use_dask' not in st.session_state: st.session_state.use_dask = False

        # Feature engineering states
        if 'detected_order_cols' not in st.session_state: st.session_state.detected_order_cols = []
        if 'detected_time_cols' not in st.session_state: st.session_state.detected_time_cols = []
        if 'detected_patient_id_col' not in st.session_state: st.session_state.detected_patient_id_col = None
        if 'freq_matrix' not in st.session_state: st.session_state.freq_matrix = None
        if 'order_sequences' not in st.session_state: st.session_state.order_sequences = None
        if 'timing_features' not in st.session_state: st.session_state.timing_features = None
        if 'order_dist' not in st.session_state: st.session_state.order_dist = None
        if 'patient_order_dist' not in st.session_state: st.session_state.patient_order_dist = None
        if 'transition_matrix' not in st.session_state: st.session_state.transition_matrix = None

        # Clustering states
        if 'clustering_input_data' not in st.session_state: st.session_state.clustering_input_data = None # Preprocessed data
        if 'reduced_data' not in st.session_state: st.session_state.reduced_data = None
        if 'kmeans_labels' not in st.session_state: st.session_state.kmeans_labels = None
        if 'hierarchical_labels' not in st.session_state: st.session_state.hierarchical_labels = None
        if 'dbscan_labels' not in st.session_state: st.session_state.dbscan_labels = None
        if 'lda_results' not in st.session_state: st.session_state.lda_results = None
        # Note: cluster_metrics were stored in the analyzer object, might need adjustment if needed globally
        if 'optimal_k' not in st.session_state: st.session_state.optimal_k = None
        if 'optimal_eps' not in st.session_state: st.session_state.optimal_eps = None

        # Analysis states
        if 'length_of_stay_comparison' not in st.session_state: st.session_state.length_of_stay_comparison = None

        # Filtering states (assuming FilteringTab manages its own state or uses these)
        if 'filter_params' not in st.session_state:
             st.session_state.filter_params = { # Default filters
                'apply_encounter_timeframe': False, 'encounter_timeframe': [],
                'apply_age_range': False, 'min_age': 18, 'max_age': 90,
                # Add other filter defaults as needed
             }
        if 'current_view' not in st.session_state: st.session_state.current_view = 'data_explorer'

        logging.info("Session state initialized.")


    def run(self):
        """Run the main application loop."""
        logging.info("Starting MIMICDashboardApp run...")
        # Set page config (do this only once at the start)
        st.set_page_config(
        	page_title="MIMIC-IV Explorer",
        	page_icon="🏥",
        	layout="wide"
        )

        # Custom CSS
        st.markdown("""
        <style>
        .main .block-container {padding-top: 2rem;}
        .sub-header {margin-top: 20px; margin-bottom: 10px; color: #1E88E5; font-weight: bold;}
        .info-box {background-color: #e1f5fe; border-left: 5px solid #0288d1; padding: 10px; margin-bottom: 10px; border-radius: 4px;}
        .stTabs [data-baseweb="tab-list"] {gap: 12px;}
        .stTabs [data-baseweb="tab"] {height: 45px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0px 0px; gap: 1px; padding: 10px 15px;}
        .stTabs [aria-selected="true"] {background-color: #e3f2fd;}
        h3 { color: #0d47a1; margin-top: 15px; border-bottom: 1px solid #eee; padding-bottom: 5px;}
        h4 { color: #1565c0; margin-top: 15px;}
        </style>
        """, unsafe_allow_html=True)

        self._display_sidebar()

        if st.session_state.current_view == 'data_explorer':
            self._show_all_tabs()
        else: # Filtering view
            # Pass necessary components to the filtering tab's render method
            # This depends on the FilteringTab implementation
            try:
                self.filtering_tab.render(st=st, data_handler=self.data_handler, feature_engineer=self.feature_engineer)
            except TypeError: # Handle potential changes in render signature
                 st.error("FilteringTab render method signature might be outdated.")
                 # Attempt a simpler call if possible
                 try:
                      self.filtering_tab.render()
                 except Exception as e:
                      st.error(f"Could not render FilteringTab: {e}")


        logging.info("MIMICDashboardApp run finished.")


    def _display_sidebar(self):
        """Handles the display and logic of the sidebar components."""
        st.sidebar.image("https://physionet.org/static/images/physionet-logo.svg", width=150) # Example logo
        st.sidebar.markdown("## MIMIC-IV Explorer")

        # View selection
        st.sidebar.markdown("### Navigation")
        view_options = ["Data Explorer", "Filtering"]
        # Find current index based on session state
        current_view_index = 0 if st.session_state.current_view == 'data_explorer' else 1
        selected_view = st.sidebar.radio("Select View", view_options, index=current_view_index, key="nav_view")
        # Update session state based on selection
        st.session_state.current_view = 'data_explorer' if selected_view == "Data Explorer" else 'filtering'

        st.sidebar.markdown("### Dataset Configuration")
        mimic_path = st.sidebar.text_input("MIMIC-IV Path", value=st.session_state.mimic_path, help="Path to MIMIC-IV v3.1 dataset")
        st.session_state.mimic_path = mimic_path

        if st.sidebar.button("Scan Directory"):
            if not mimic_path or not os.path.isdir(mimic_path):
                st.sidebar.error("Please enter a valid directory path.")
            else:
                with st.spinner("Scanning directory..."):
                    try:
                        available_tables, file_paths, file_sizes, table_display_names = self.data_handler.scan_mimic_directory(mimic_path)
                        if available_tables:
                            st.session_state.available_tables = available_tables
                            st.session_state.file_paths = file_paths
                            st.session_state.file_sizes = file_sizes
                            st.session_state.table_display_names = table_display_names
                            st.sidebar.success(f"Found {sum(len(t) for t in available_tables.values())} tables.")
                            # Reset selected table if module changes or table disappears
                            if st.session_state.selected_module not in available_tables or \
                               st.session_state.selected_table not in available_tables.get(st.session_state.selected_module, []):
                                st.session_state.selected_module = list(available_tables.keys())[0] if available_tables else None
                                st.session_state.selected_table = available_tables[st.session_state.selected_module][0] if st.session_state.selected_module else None
                                st.session_state.df = None # Clear old dataframe
                        else:
                            st.sidebar.error("No MIMIC-IV tables found.")
                            st.session_state.available_tables = {} # Clear old tables
                    except Exception as e:
                        st.sidebar.error(f"Error scanning directory: {e}")
                        logging.exception("Directory Scan Error")

        if st.session_state.available_tables:
            module = st.sidebar.selectbox("Module", list(st.session_state.available_tables.keys()), key="sb_module",
                                          index=list(st.session_state.available_tables.keys()).index(st.session_state.selected_module) if st.session_state.selected_module in st.session_state.available_tables else 0)

            if module != st.session_state.selected_module:
                 st.session_state.selected_module = module
                 # Reset table selection if module changes
                 st.session_state.selected_table = st.session_state.available_tables[module][0] if module in st.session_state.available_tables and st.session_state.available_tables[module] else None
                 st.session_state.df = None # Clear old dataframe

            if module in st.session_state.available_tables:
                table_options = st.session_state.available_tables[module]
                table_display_options = [st.session_state.table_display_names.get((module, t), t) for t in table_options]
                display_to_table = {display: table for table, display in zip(table_options, table_display_options)}

                # Find index of current selection
                current_table_display = st.session_state.table_display_names.get((module, st.session_state.selected_table), st.session_state.selected_table)
                try:
                    selected_display_index = table_display_options.index(current_table_display)
                except ValueError:
                    selected_display_index = 0
                    st.session_state.selected_table = display_to_table[table_display_options[0]] if table_display_options else None # Reset if not found

                selected_display = st.sidebar.selectbox("Table", table_display_options, index=selected_display_index, key="sb_table")
                selected_table_new = display_to_table[selected_display]

                if selected_table_new != st.session_state.selected_table:
                     st.session_state.selected_table = selected_table_new
                     st.session_state.df = None # Clear old dataframe

                # Show table info
                table_info = self.data_handler.get_table_info(module, st.session_state.selected_table)
                st.sidebar.caption(table_info) # Use caption for less prominent info

                with st.sidebar.expander("Load Options"):
                    encoding = st.selectbox("Encoding", ["latin-1", "utf-8"], index=0, key="sb_enc")
                    st.session_state.sample_size = st.number_input("Sample Size (Rows)", 100, 1000000, st.session_state.sample_size, 100, key="sb_sample")
                    st.session_state.use_dask = st.checkbox("Use Dask (for large files)", value=st.session_state.use_dask, key="sb_dask")

                if st.sidebar.button("Load Table"):
                    file_path = st.session_state.file_paths.get((module, st.session_state.selected_table))
                    if file_path:
                        st.session_state.current_file_path = file_path
                        framework = "Dask" if st.session_state.use_dask else "Pandas"
                        with st.spinner(f"Loading {st.session_state.selected_table} using {framework}..."):
                            try:
                                df, total_rows = self.data_handler.load_mimic_table(
                                    file_path=file_path,
                                    sample_size=st.session_state.sample_size,
                                    encoding=encoding,
                                    use_dask=st.session_state.use_dask
                                )
                                st.session_state.total_row_count = total_rows
                                if df is not None:
                                    st.session_state.df = df
                                    # Auto-detect columns after loading
                                    st.session_state.detected_order_cols = self.feature_engineer.detect_order_columns(df)
                                    st.session_state.detected_time_cols = self.feature_engineer.detect_temporal_columns(df)
                                    st.session_state.detected_patient_id_col = self.feature_engineer.detect_patient_id_column(df)
                                    st.success(f"Loaded {len(df)} rows ({total_rows} total).")
                                    # Reset downstream states when new data is loaded
                                    self._reset_downstream_states()
                                    st.rerun() # Rerun to update the main view immediately
                                else:
                                     st.error("Failed to load data.")
                            except Exception as e:
                                st.error(f"Error loading table: {e}")
                                logging.exception("Table Load Error")
                    else:
                        st.sidebar.error("File path not found for selected table.")


    def _reset_downstream_states(self):
        """ Resets states that depend on the loaded dataframe. """
        keys_to_reset = [
            'freq_matrix', 'order_sequences', 'timing_features', 'order_dist',
            'patient_order_dist', 'transition_matrix', 'clustering_input_data',
            'reduced_data', 'kmeans_labels', 'hierarchical_labels', 'dbscan_labels',
            'lda_results', 'optimal_k', 'optimal_eps', 'length_of_stay_comparison'
        ]
        for key in keys_to_reset:
            if key in st.session_state:
                st.session_state[key] = None
        logging.info("Reset downstream session states.")


    def _show_all_tabs(self):
        """Handles the display of the main content area with tabs."""
        if st.session_state.df is not None:
            st.markdown(f"<h2 class='sub-header'>Exploring: {st.session_state.selected_module.upper()} / {st.session_state.selected_table}</h2>", unsafe_allow_html=True)
            st.markdown(f"<div class='info-box'>", unsafe_allow_html=True)
            cols = st.columns(4)
            cols[0].metric("Module", st.session_state.selected_module)
            cols[1].metric("Table", st.session_state.selected_table)
            # Format file size
            file_size_mb = st.session_state.file_sizes.get((st.session_state.selected_module, st.session_state.selected_table), 0)
            size_str = f"{file_size_mb:.1f} MB" if file_size_mb < 1000 else f"{file_size_mb/1000:.1f} GB"
            cols[2].metric("File Size", size_str)
            cols[3].metric("Rows Loaded", f"{len(st.session_state.df):,} / {st.session_state.total_row_count:,}")
            st.markdown("</div>", unsafe_allow_html=True)

            # Create tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Explore & Visualize",
                "🛠️ Feature Engineering",
                "🧩 Clustering Analysis",
                "📈 Cluster Analysis & Viz",
                "📤 Export"
            ])

            # Tab 1: Exploration & Visualization
            with tab1:
                self.visualizer.display_data_preview(st.session_state.df)
                self.visualizer.display_dataset_statistics(st.session_state.df)
                self.visualizer.display_visualizations(st.session_state.df)

            # Tab 2: Feature Engineering
            with tab2:
                self.feature_engineering_tab_view.render(st, st.session_state.df.columns.tolist())

            # Tab 3: Clustering Analysis
            with tab3:
                self.clustering_analysis_tab_view.render(st)

            # Tab 4: Analysis & Visualization
            with tab4:
                self.analysis_visualization_tab_view.render(st)

            # Tab 5: Export Options
            with tab5:
                st.markdown("<h2 class='sub-header'>Export Current Data</h2>", unsafe_allow_html=True)
                st.info("Export the currently loaded (and potentially sampled) data.")
                col1, col2 = st.columns(2)
                export_filename = f"mimic_iv_{st.session_state.selected_module}_{st.session_state.selected_table}_sample"

                with col1:
                    try:
                        csv_data = st.session_state.df.to_csv(index=False).encode('utf-8')
                        st.download_button(label="Download CSV", data=csv_data, file_name=f"{export_filename}.csv", mime="text/csv")
                    except Exception as e:
                        st.error(f"Error preparing CSV: {e}")

                with col2:
                    try:
                        # Convert to Parquet in memory
                        parquet_buffer = BytesIO()
                        table = pa.Table.from_pandas(st.session_state.df)
                        pq.write_table(table, parquet_buffer)
                        st.download_button(label="Download Parquet", data=parquet_buffer.getvalue(), file_name=f"{export_filename}.parquet", mime="application/octet-stream")
                    except Exception as e:
                        st.error(f"Error preparing Parquet: {e}")
        else:
            # Welcome message
            st.markdown("<div class='info-box'><h2 class='sub-header'>Welcome to the MIMIC-IV Explorer</h2><p>Please configure the dataset path, scan the directory, and load a table using the sidebar to begin.</p></div>", unsafe_allow_html=True)
            st.markdown("""
            <h2 class='sub-header'>About MIMIC-IV</h2>
            <div class='info-box'>
            <p>MIMIC-IV (Medical Information Mart for Intensive Care) is a large, freely available database comprising deidentified health-related data associated with over 250,000 patients admitted to Beth Israel Deaconess Medical Center in Boston, MA.</p>
            <p>Key Modules:</p>
            <ul>
                <li><b>hosp:</b> Hospital-wide data (admissions, labs, diagnoses, prescriptions).</li>
                <li><b>icu:</b> ICU-specific data (chartevents, procedureevents, outputevents).</li>
            </ul>
            <p>Use the sidebar to connect to your local MIMIC-IV dataset and start exploring.</p>
            <p><i>Ensure you have completed the necessary training and data use agreements on PhysioNet to access MIMIC-IV.</i></p>
            <p><a href="https://physionet.org/content/mimiciv/3.1/" target="_blank">MIMIC-IV on PhysioNet</a></p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    # Ensure necessary directories exist if saving features/models locally
    # Example: os.makedirs('output/features', exist_ok=True)
    # Example: os.makedirs('output/models', exist_ok=True)
    # Example: os.makedirs('output/reports', exist_ok=True)

    app = MIMICDashboardApp()
    app.run()
