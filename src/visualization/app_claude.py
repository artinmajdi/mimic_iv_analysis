# Standard libraries
import os
import warnings

# Data processing libraries
import dask.dataframe as dd
import numpy as np
import pandas as pd
from scipy import stats

# Visualization libraries
import plotly.express as px
import streamlit as st

# Machine learning libraries
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import umap

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="MIMIC-IV Order Clustering Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================================
# Data Loading and Processing Module
# ===================================

class MimicDataLoader:
    """Class to handle loading and initial processing of MIMIC-IV data"""

    def __init__(self, base_path):
        """Initialize with path to MIMIC-IV dataset"""
        self.base_path = base_path
        self.patients_df = None
        self.admissions_df = None
        self.diagnoses_df = None
        self.poe_df = None
        self.poe_detail_df = None
        self.d_icd_diagnoses_df = None

    def validate_path(self):
        """Check if the specified path exists and contains MIMIC-IV data"""
        required_dirs = ['hosp', 'icu']
        for dir_name in required_dirs:
            if not os.path.exists(os.path.join(self.base_path, dir_name)):
                return False
        return True

    def load_patients(self):
        """Load patients table using dask for memory efficiency"""
        file_path = os.path.join(self.base_path, 'hosp', 'patients.csv')

        # Define dtypes for patients table
        dtypes = {
            'subject_id': 'int64',
            'gender': 'object',
            'anchor_age': 'float64',
            'anchor_year': 'float64',
            'anchor_year_group': 'object',
            'dod': 'object'
        }

        self.patients_df = dd.read_csv(file_path, dtype=dtypes).compute()

        # Convert date columns
        if 'dod' in self.patients_df.columns:
            self.patients_df['dod'] = pd.to_datetime(self.patients_df['dod'])

        return self.patients_df

    def load_admissions(self):
        """Load admissions table using dask"""
        file_path = os.path.join(self.base_path, 'hosp', 'admissions.csv')

        # Define dtypes for datetime columns
        dtypes = {
            'subject_id': 'int64',
            'hadm_id': 'int64',
            'admittime': 'object',
            'dischtime': 'object',
            'deathtime': 'object',
            'edregtime': 'object',
            'edouttime': 'object',
            'hospital_expire_flag': 'int64'
        }

        # Load data with specified dtypes
        self.admissions_df = dd.read_csv(file_path, dtype=dtypes).compute()

        # Convert datetime columns
        datetime_cols = ['admittime', 'dischtime', 'deathtime', 'edregtime', 'edouttime']
        for col in datetime_cols:
            if col in self.admissions_df.columns:
                self.admissions_df[col] = pd.to_datetime(self.admissions_df[col], format='%Y-%m-%d %H:%M:%S')

        # Calculate length of stay in days
        self.admissions_df['los_days'] = (
            self.admissions_df['dischtime'] - self.admissions_df['admittime']
        ).dt.total_seconds() / (24 * 3600)

        return self.admissions_df

    def load_diagnoses(self):
        """Load diagnoses_icd table using dask"""
        file_path = os.path.join(self.base_path, 'hosp', 'diagnoses_icd.csv')

        # Define dtypes for diagnoses table
        dtypes = {
            'subject_id': 'int64',
            'hadm_id': 'int64',
            'seq_num': 'int64',
            'icd_code': 'object',
            'icd_version': 'int64'
        }

        self.diagnoses_df = dd.read_csv(file_path, dtype=dtypes).compute()
        return self.diagnoses_df

    def load_d_icd_diagnoses(self):
        """Load d_icd_diagnoses table using dask"""
        file_path = os.path.join(self.base_path, 'hosp', 'd_icd_diagnoses.csv')

        # Define dtypes for ICD diagnoses lookup table
        dtypes = {
            'icd_code': 'object',
            'icd_version': 'int64',
            'long_title': 'object',
            'short_title': 'object'
        }

        self.d_icd_diagnoses_df = dd.read_csv(file_path, dtype=dtypes).compute()
        return self.d_icd_diagnoses_df

    def load_poe(self):
        """Load poe table using dask"""
        file_path = os.path.join(self.base_path, 'hosp', 'poe.csv')

        # Define dtypes for POE table
        dtypes = {
            'subject_id': 'int64',
            'hadm_id': 'int64',
            'orderid': 'int64',
            'order_type': 'object',
            'order_subtype': 'object',
            'transaction_type': 'object',
            'order_status': 'object',
            'ordertime': 'object',
            'status_change_time': 'object'
        }

        self.poe_df = dd.read_csv(file_path, dtype=dtypes).compute()

        # Convert datetime columns
        datetime_cols = ['ordertime', 'status_change_time']
        for col in datetime_cols:
            if col in self.poe_df.columns:
                self.poe_df[col] = pd.to_datetime(self.poe_df[col], format='%Y-%m-%d %H:%M:%S')

        return self.poe_df

    def load_poe_detail(self):
        """Load poe_detail table using dask"""
        file_path = os.path.join(self.base_path, 'hosp', 'poe_detail.csv')

        # Define dtypes for POE detail table
        dtypes = {
            'subject_id': 'int64',
            'hadm_id': 'int64',
            'orderid': 'int64',
            'field_name': 'object',
            'field_value': 'object'
        }

        self.poe_detail_df = dd.read_csv(file_path, dtype=dtypes).compute()
        return self.poe_detail_df

    def load_all_tables(self):
        """Load all required tables for analysis"""
        with st.spinner('Loading patients table...'):
            self.load_patients()
        with st.spinner('Loading admissions table...'):
            self.load_admissions()
        with st.spinner('Loading diagnoses table...'):
            self.load_diagnoses()
        with st.spinner('Loading ICD diagnoses lookup...'):
            self.load_d_icd_diagnoses()
        with st.spinner('Loading provider orders (POE) table...'):
            self.load_poe()
        with st.spinner('Loading POE details table...'):
            self.load_poe_detail()
        return {
            'patients': self.patients_df,
            'admissions': self.admissions_df,
            'diagnoses': self.diagnoses_df,
            'd_icd_diagnoses': self.d_icd_diagnoses_df,
            'poe': self.poe_df,
            'poe_detail': self.poe_detail_df
        }


class DataPreprocessor:
    """Class to preprocess MIMIC-IV data for analysis"""

    def __init__(self, data_dict):
        """Initialize with loaded data tables"""
        self.data = data_dict
        self.t2dm_cohort = None
        self.orders_data = None
        self.feature_matrix = None

    def identify_t2dm_patients(self):
        """Identify patients with Type 2 Diabetes based on ICD codes"""
        # Filter diagnoses for T2DM (ICD-10 code starting with E11)
        t2dm_codes = self.data['diagnoses'][
            self.data['diagnoses']['icd_code'].str.startswith('E11', na=False)
        ]

        # Get unique hadm_id for T2DM patients
        t2dm_hadm_ids = t2dm_codes['hadm_id'].unique()

        # Filter admissions for these patients
        t2dm_admissions = self.data['admissions'][
            self.data['admissions']['hadm_id'].isin(t2dm_hadm_ids)
        ]

        # Join with patients data
        t2dm_cohort = t2dm_admissions.merge(
            self.data['patients'],
            on='subject_id',
            how='inner'
        )

        # Apply exclusion criteria
        self.t2dm_cohort = t2dm_cohort[
            # Adult patients (18-75 years)
            (t2dm_cohort['anchor_age'] >= 18) &
            (t2dm_cohort['anchor_age'] <= 75) &
            # Exclude in-hospital deaths
            (t2dm_cohort['hospital_expire_flag'] == 0) &
            # Ensure discharge time exists (for LOS calculation)
            (~t2dm_cohort['dischtime'].isna())
        ].copy()

        return self.t2dm_cohort

    def extract_orders_for_cohort(self, time_window=48):
        """Extract provider orders for the T2DM cohort within specified time window after admission"""
        if self.t2dm_cohort is None:
            raise ValueError("T2DM cohort has not been identified yet")

        # Get all orders for patients in cohort
        hadm_ids = self.t2dm_cohort['hadm_id'].tolist()

        # Filter orders for cohort patients
        cohort_orders = self.data['poe'][
            self.data['poe']['hadm_id'].isin(hadm_ids)
        ].copy()

        # Merge with admissions to get admission time
        cohort_orders = cohort_orders.merge(
            self.data['admissions'][['hadm_id', 'admittime']],
            on='hadm_id',
            how='left'
        )

        # Calculate hours since admission
        cohort_orders['hours_since_admission'] = (
            cohort_orders['ordertime'] - cohort_orders['admittime']
        ).dt.total_seconds() / 3600

        # Filter for orders within time window of admission
        self.orders_data = cohort_orders[
            (cohort_orders['hours_since_admission'] >= 0) &
            (cohort_orders['hours_since_admission'] <= time_window)
        ].copy()

        return self.orders_data

    def create_feature_matrix(self):
        """Create feature matrix for clustering based on order patterns"""
        if self.orders_data is None:
            raise ValueError("Orders data has not been extracted yet")

        # Create order count matrix
        order_counts = self.orders_data.groupby(['hadm_id', 'order_type']).size().unstack(fill_value=0)

        # Create order timing features
        order_timing = self.orders_data.groupby('hadm_id').agg({
            'hours_since_admission': ['min', 'max', 'mean', 'std', 'count']
        })
        order_timing.columns = ['_'.join(col).strip() for col in order_timing.columns.values]

        # Merge features
        self.feature_matrix = pd.merge(
            order_counts,
            order_timing,
            left_index=True,
            right_index=True,
            how='left'
        )

        # Fill missing values
        self.feature_matrix = self.feature_matrix.fillna(0)

        # Merge with outcome data (length of stay)
        self.feature_matrix = pd.merge(
            self.feature_matrix,
            self.t2dm_cohort[['hadm_id', 'los_days']],
            left_index=True,
            right_on='hadm_id',
            how='left'
        ).set_index('hadm_id')

        # Scale numerical features
        scaler = StandardScaler()
        numeric_cols = self.feature_matrix.select_dtypes(include=['float64', 'int64']).columns
        self.feature_matrix[numeric_cols] = scaler.fit_transform(self.feature_matrix[numeric_cols])

        return self.feature_matrix


# ===================================
# Clustering Analysis Module
# ===================================

class ClusteringAnalysis:
    """Class to perform different clustering techniques on order data"""

    def __init__(self, feature_matrix):
        """Initialize with preprocessed feature matrix"""
        self.feature_matrix = feature_matrix
        self.X = feature_matrix.drop('los_days', axis=1) if 'los_days' in feature_matrix.columns else feature_matrix
        self.y = feature_matrix['los_days'] if 'los_days' in feature_matrix.columns else None
        self.cluster_labels = None
        self.model = None
        self.reduced_features = None

    def apply_lda(self, n_components=8, max_iter=20):
        """Apply Latent Dirichlet Allocation for topic modeling of orders"""
        # Ensure non-negative data
        X_non_neg = self.X.copy()
        X_non_neg[X_non_neg < 0] = 0

        # Create and fit LDA model
        lda = LatentDirichletAllocation(
            n_components=n_components,
            max_iter=max_iter,
            learning_method='online',
            random_state=42
        )

        transformed_data = lda.fit_transform(X_non_neg)

        # Get dominant topic for each admission
        self.cluster_labels = np.argmax(transformed_data, axis=1)
        self.model = lda
        self.reduced_features = transformed_data

        return self.cluster_labels, transformed_data, lda

    def apply_kmeans(self, n_clusters=8, random_state=42):
        """Apply K-means clustering to the feature matrix"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.cluster_labels = kmeans.fit_predict(self.X)
        self.model = kmeans

        return self.cluster_labels, kmeans

    def apply_dbscan(self, eps=0.5, min_samples=5):
        """Apply DBSCAN clustering to the feature matrix"""
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.cluster_labels = dbscan.fit_predict(self.X)
        self.model = dbscan

        return self.cluster_labels, dbscan

    def apply_hierarchical(self, n_clusters=8, linkage='ward'):
        """Apply hierarchical clustering to the feature matrix"""
        hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        self.cluster_labels = hc.fit_predict(self.X)
        self.model = hc

        return self.cluster_labels, hc

    def apply_dimensionality_reduction(self, method='pca', n_components=2):
        """Apply dimensionality reduction for visualization"""
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42)
        elif method == 'umap':
            reducer = umap.UMAP(n_components=n_components, random_state=42)
        elif method == 'mds':
            reducer = MDS(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")

        self.reduced_features = reducer.fit_transform(self.X)
        return self.reduced_features, reducer

    def evaluate_clusters(self):
        """Evaluate clustering quality using various metrics"""
        if self.cluster_labels is None:
            raise ValueError("Clustering has not been performed yet")

        # Filter out noise points (label -1 in DBSCAN)
        valid_indices = self.cluster_labels != -1
        X_valid = self.X.iloc[valid_indices] if isinstance(self.X, pd.DataFrame) else self.X[valid_indices]
        labels_valid = self.cluster_labels[valid_indices]

        # Only compute metrics if we have more than one cluster
        unique_labels = np.unique(labels_valid)
        if len(unique_labels) <= 1:
            return {
                'silhouette_score': float('nan'),
                'calinski_harabasz_score': float('nan'),
                'num_clusters': len(unique_labels)
            }

        # Compute silhouette score
        try:
            silhouette = silhouette_score(X_valid, labels_valid)
        except:
            silhouette = float('nan')

        # Compute Calinski-Harabasz score
        try:
            calinski_harabasz = calinski_harabasz_score(X_valid, labels_valid)
        except:
            calinski_harabasz = float('nan')

        return {
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski_harabasz,
            'num_clusters': len(unique_labels)
        }

    def analyze_outcomes_by_cluster(self):
        """Analyze length of stay and other outcomes by cluster"""
        if self.cluster_labels is None or self.y is None:
            raise ValueError("Clustering has not been performed or outcome data is missing")

        # Create DataFrame with cluster labels and outcomes
        results_df = pd.DataFrame({
            'cluster': self.cluster_labels,
            'los_days': self.y
        })

        # Calculate statistics by cluster
        cluster_stats = results_df.groupby('cluster').agg({
            'los_days': ['count', 'mean', 'median', 'std', 'min', 'max']
        })

        # Flatten column names
        cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]

        # Calculate p-value for differences between clusters
        # ANOVA test
        groups = [results_df[results_df.cluster == c]['los_days'] for c in results_df.cluster.unique()]
        f_stat, p_value = stats.f_oneway(*groups)

        return cluster_stats, f_stat, p_value


class FeatureImportanceAnalyzer:
    """Class to analyze feature importance for clusters and outcomes"""

    def __init__(self, feature_matrix, cluster_labels):
        """Initialize with feature matrix and cluster labels"""
        self.feature_matrix = feature_matrix
        self.cluster_labels = cluster_labels
        self.X = feature_matrix.drop('los_days', axis=1) if 'los_days' in feature_matrix.columns else feature_matrix
        self.y = feature_matrix['los_days'] if 'los_days' in feature_matrix.columns else None

    def characterize_clusters(self):
        """Characterize each cluster based on distinctive features"""
        # For each cluster, identify distinctive orders
        cluster_profiles = []

        for cluster_id in np.unique(self.cluster_labels):
            # Skip noise cluster (-1) if present
            if cluster_id == -1:
                continue

            # Get admissions in this cluster
            cluster_indices = np.where(self.cluster_labels == cluster_id)[0]
            cluster_members = self.X.iloc[cluster_indices]

            # Calculate mean feature values for this cluster
            cluster_mean = cluster_members.mean(axis=0)

            # Compare to overall mean
            overall_mean = self.X.mean(axis=0)

            # Calculate ratio or difference
            feature_importance = (cluster_mean - overall_mean) / (overall_mean.std() + 1e-6)

            # Sort by importance (both positive and negative)
            feature_importance = feature_importance.sort_values(ascending=False)

            # Get most distinctive features (top positive and negative)
            top_positive = feature_importance.head(10)
            top_negative = feature_importance.tail(10).iloc[::-1]  # Reverse order

            cluster_profiles.append({
                'cluster_id': cluster_id,
                'size': len(cluster_members),
                'top_positive_features': top_positive,
                'top_negative_features': top_negative
            })

        return cluster_profiles

    def predict_los_from_features(self):
        """Use random forest to predict LOS from order features"""
        if self.y is None:
            raise ValueError("Outcome data (length of stay) is missing")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )

        # Train random forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        # Evaluate model
        r2_train = rf.score(X_train, y_train)
        r2_test = rf.score(X_test, y_test)

        return feature_importance, rf, r2_train, r2_test


# ===================================
# Visualization Module
# ===================================

class DataVisualizer:
    """Class to create visualizations for the analysis"""

    def __init__(self, feature_matrix, cluster_labels=None, reduced_features=None):
        """Initialize with data and clustering results"""
        self.feature_matrix = feature_matrix
        self.cluster_labels = cluster_labels
        self.reduced_features = reduced_features

    def plot_reduced_dimensions(self, method_name='Dimensionality Reduction'):
        """Plot data in reduced dimensions colored by cluster"""
        if self.reduced_features is None or self.cluster_labels is None:
            raise ValueError("Reduced features or cluster labels are missing")

        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'Component 1': self.reduced_features[:, 0],
            'Component 2': self.reduced_features[:, 1] if self.reduced_features.shape[1] > 1 else np.zeros(len(self.reduced_features)),
            'Cluster': self.cluster_labels
        })

        # Create Plotly scatter plot
        fig = px.scatter(
            plot_df,
            x='Component 1',
            y='Component 2',
            color='Cluster',
            color_continuous_scale=px.colors.qualitative.G10 if len(np.unique(self.cluster_labels)) <= 10 else px.colors.qualitative.Alphabet,
            title=f'Cluster Visualization using {method_name}',
            opacity=0.7
        )

        fig.update_layout(
            height=600,
            legend_title_text='Cluster'
        )

        return fig

    def plot_los_by_cluster(self):
        """Plot length of stay distribution by cluster"""
        if self.cluster_labels is None or 'los_days' not in self.feature_matrix.columns:
            raise ValueError("Cluster labels or length of stay data is missing")

        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'Cluster': self.cluster_labels,
            'Length of Stay (days)': self.feature_matrix['los_days']
        })

        # Create box plot
        fig = px.box(
            plot_df,
            x='Cluster',
            y='Length of Stay (days)',
            color='Cluster',
            title='Length of Stay Distribution by Cluster',
            points='all'
        )

        fig.update_layout(
            height=500,
            showlegend=False
        )

        return fig

    def plot_feature_importance(self, feature_importance_df, top_n=20):
        """Plot feature importance from model"""
        # Get top features
        top_features = feature_importance_df.head(top_n)

        # Create bar plot
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title=f'Top {top_n} Most Important Features',
            color='importance',
            color_continuous_scale='viridis'
        )

        fig.update_layout(
            height=600,
            yaxis={'categoryorder': 'total ascending'}
        )

        return fig

    def plot_cluster_characteristics(self, cluster_profiles, cluster_id):
        """Plot distinctive features for a specific cluster"""
        # Find profile for specified cluster
        profile = None
        for p in cluster_profiles:
            if p['cluster_id'] == cluster_id:
                profile = p
                break

        if profile is None:
            raise ValueError(f"No profile found for cluster {cluster_id}")

        # Create DataFrame for plotting
        positive_df = pd.DataFrame({
            'Feature': profile['top_positive_features'].index,
            'Importance': profile['top_positive_features'].values,
            'Type': 'Overrepresented'
        })

        negative_df = pd.DataFrame({
            'Feature': profile['top_negative_features'].index,
            'Importance': -profile['top_negative_features'].values,  # Make negative for visualization
            'Type': 'Underrepresented'
        })

        plot_df = pd.concat([positive_df, negative_df])

        # Create bar plot
        fig = px.bar(
            plot_df,
            x='Importance',
            y='Feature',
            color='Type',
            orientation='h',
            title=f'Distinctive Features for Cluster {cluster_id}',
            color_discrete_map={'Overrepresented': 'green', 'Underrepresented': 'red'}
        )

        fig.update_layout(
            height=700,
            yaxis={'categoryorder': 'total ascending'}
        )

        return fig

    def plot_temporal_patterns(self, orders_data):
        """Plot temporal patterns of orders by cluster"""
        if self.cluster_labels is None:
            raise ValueError("Cluster labels are missing")

        # Merge orders with cluster labels
        orders_with_clusters = orders_data.copy()
        cluster_df = pd.DataFrame({
            'hadm_id': self.feature_matrix.index,
            'cluster': self.cluster_labels
        })
        orders_with_clusters = orders_with_clusters.merge(
            cluster_df,
            on='hadm_id',
            how='inner'
        )

        # Bin hours since admission
        orders_with_clusters['hour_bin'] = pd.cut(
            orders_with_clusters['hours_since_admission'],
            bins=np.arange(0, 49, 4),  # 4-hour bins
            labels=[f'{i}-{i+4}h' for i in range(0, 48, 4)]
        )

        # Count orders by cluster, hour bin, and order type
        order_counts = orders_with_clusters.groupby(['cluster', 'hour_bin', 'order_type']).size().reset_index(name='count')

        # Normalize by cluster size
        cluster_sizes = orders_with_clusters.groupby('cluster')['hadm_id'].nunique().reset_index(name='size')
        order_counts = order_counts.merge(cluster_sizes, on='cluster', how='left')
        order_counts['normalized_count'] = order_counts['count'] / order_counts['size']

        # Get top order types
        top_orders = order_counts.groupby('order_type')['count'].sum().sort_values(ascending=False).head(10).index
        filtered_counts = order_counts[order_counts['order_type'].isin(top_orders)]

        # Create line plot
        fig = px.line(
            filtered_counts,
            x='hour_bin',
            y='normalized_count',
            color='order_type',
            facet_row='cluster',
            title='Temporal Order Patterns by Cluster',
            labels={'normalized_count': 'Orders per Patient', 'hour_bin': 'Hours Since Admission'}
        )

        fig.update_layout(
            height=800,
            showlegend=True
        )

        return fig


# ===================================
# Streamlit App Interface
# ===================================

class StreamlitAppClaude:
    """Main Streamlit application class"""

    def __init__(self):
        """Initialize the Streamlit app interface"""
        self.data_loader = None
        self.preprocessor = None
        self.clustering = None
        self.feature_analyzer = None
        self.visualizer = None

        # App state management
        self.data_loaded = False
        self.preprocessing_done = False
        self.clustering_done = False

    def setup_sidebar(self):
        """Set up the application sidebar for navigation and controls"""
        st.sidebar.title("MIMIC-IV Order Clustering")

        # Navigation
        self.page = st.sidebar.radio(
            "Navigation",
            ["Home", "Data Loading", "Preprocessing", "Clustering Analysis", "Outcome Analysis", "Cluster Interpretation"]
        )

        # Data path input (shown only on Data Loading page)
        if self.page == "Data Loading":
            self.mimic_path = st.sidebar.text_input(
                "MIMIC-IV dataset path",
                value="/Users/artinmajdi/Documents/GitHubs/Career/mimic_iv/dataset/mimic-iv-3.1"
            )

        # Preprocessing options
        if self.page == "Preprocessing" and self.data_loaded:
            st.sidebar.subheader("Preprocessing Options")
            self.time_window = st.sidebar.slider(
                "Order time window (hours after admission)",
                min_value=6,
                max_value=72,
                value=48,
                step=6
            )

        # Clustering options
        if self.page == "Clustering Analysis" and self.preprocessing_done:
            st.sidebar.subheader("Clustering Options")

            self.cluster_method = st.sidebar.selectbox(
                "Clustering Method",
                ["LDA", "K-means", "DBSCAN", "Hierarchical"]
            )

            if self.cluster_method == "LDA":
                self.n_topics = st.sidebar.slider("Number of Topics", 2, 20, 8)
                self.lda_max_iter = st.sidebar.slider("Max Iterations", 10, 100, 20)

            elif self.cluster_method == "K-means":
                self.n_clusters = st.sidebar.slider("Number of Clusters", 2, 20, 8)

            elif self.cluster_method == "DBSCAN":
                self.eps = st.sidebar.slider("Epsilon", 0.1, 2.0, 0.5, 0.1)
                self.min_samples = st.sidebar.slider("Min Samples", 2, 20, 5)

            elif self.cluster_method == "Hierarchical":
                self.n_clusters_hc = st.sidebar.slider("Number of Clusters", 2, 20, 8)
                self.linkage = st.sidebar.selectbox("Linkage Method", ["ward", "complete", "average", "single"])

            # Dimensionality reduction for visualization
            st.sidebar.subheader("Visualization Options")
            self.dim_reduction_method = st.sidebar.selectbox(
                "Dimensionality Reduction",
                ["PCA", "t-SNE", "UMAP", "MDS"]
            )

        # Cluster interpretation options
        if self.page == "Cluster Interpretation" and self.clustering_done:
            st.sidebar.subheader("Interpretation Options")

            # Get available clusters
            available_clusters = np.unique(self.clustering.cluster_labels)
            available_clusters = available_clusters[available_clusters != -1]  # Remove noise cluster

            self.selected_cluster = st.sidebar.selectbox(
                "Select Cluster to Interpret",
                available_clusters
            )

    def render_home_page(self):
        """Render the home page with app overview"""
        st.title("MIMIC-IV Order Clustering Analysis")

        st.markdown("""
        ## Overview
        This application analyzes provider order patterns in the MIMIC-IV database to identify clusters
        associated with shorter length of stay for Type 2 Diabetes patients.

        ### Key Features:
        - **Data Loading**: Efficiently load large MIMIC-IV tables using Dask for out-of-core processing
        - **Patient Cohort Selection**: Identify T2DM patients and extract their provider orders
        - **Feature Engineering**: Create meaningful features from order patterns and temporal data
        - **Multiple Clustering Techniques**: LDA, K-means, DBSCAN, and Hierarchical clustering
        - **Outcome Analysis**: Correlate clusters with length of stay and other outcomes
        - **Cluster Interpretation**: Understand what distinguishes each cluster of practice patterns

        ### How to Use:
        1. Start with the **Data Loading** page to load MIMIC-IV data
        2. Proceed to **Preprocessing** to prepare the data for analysis
        3. Conduct **Clustering Analysis** to identify order patterns
        4. Analyze outcomes on the **Outcome Analysis** page
        5. Interpret findings on the **Cluster Interpretation** page

        ### Getting Started:
        Use the navigation panel on the left to move through the analysis workflow.
        """)

        # Show status indicators
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Data Loaded",
                value="Yes" if self.data_loaded else "No"
            )

        with col2:
            st.metric(
                label="Preprocessing Done",
                value="Yes" if self.preprocessing_done else "No"
            )

        with col3:
            st.metric(
                label="Clustering Done",
                value="Yes" if self.clustering_done else "No"
            )

    def render_data_loading_page(self):
        """Render the data loading page"""
        st.title("Data Loading")

        st.markdown("""
        ## MIMIC-IV Data Loading

        Specify the path to your MIMIC-IV dataset (version 3.1) and load the required tables.
        The data loader uses Dask for efficient handling of large CSV files.
        """)

        if st.button("Load MIMIC-IV Data"):
            # Initialize data loader
            self.data_loader = MimicDataLoader(self.mimic_path)

            # Validate path
            if not self.data_loader.validate_path():
                st.error(f"Invalid MIMIC-IV path: {self.mimic_path}")
                return

            # Load data
            try:
                with st.spinner("Loading MIMIC-IV tables..."):
                    self.data_tables = self.data_loader.load_all_tables()

                # Display basic statistics
                st.success("Data loaded successfully!")

                # Show table statistics
                stats = {}
                for table_name, df in self.data_tables.items():
                    stats[table_name] = {
                        'rows': len(df),
                        'columns': len(df.columns)
                    }

                st.subheader("Dataset Statistics")
                stats_df = pd.DataFrame.from_dict(stats, orient='index')
                st.dataframe(stats_df)

                # Show preview of tables
                st.subheader("Data Previews")
                for table_name, df in self.data_tables.items():
                    with st.expander(f"{table_name} Table"):
                        st.dataframe(df.head())

                # Set data loaded flag
                self.data_loaded = True

            except Exception as e:
                st.error(f"Error loading data: {str(e)}")

    def render_preprocessing_page(self):
        """Render the data preprocessing page"""
        st.title("Data Preprocessing")

        if not self.data_loaded:
            st.warning("Please load the data first on the Data Loading page.")
            return

        st.markdown("""
        ## Data Preprocessing

        Preprocess the MIMIC-IV data for analysis:
        1. Identify Type 2 Diabetes patients (using ICD-10 codes)
        2. Extract provider orders for these patients
        3. Create features for clustering analysis
        """)

        if st.button("Preprocess Data"):
            try:
                # Initialize preprocessor
                with st.spinner("Initializing preprocessor..."):
                    self.preprocessor = DataPreprocessor(self.data_tables)

                # Identify T2DM patients
                with st.spinner("Identifying T2DM patients..."):
                    t2dm_cohort = self.preprocessor.identify_t2dm_patients()
                    st.success(f"Found {len(t2dm_cohort)} admissions with T2DM diagnosis.")

                # Extract orders
                with st.spinner(f"Extracting orders within {self.time_window} hours of admission..."):
                    orders_data = self.preprocessor.extract_orders_for_cohort(time_window=self.time_window)
                    st.success(f"Extracted {len(orders_data)} orders for T2DM patients.")

                # Create feature matrix
                with st.spinner("Creating feature matrix..."):
                    feature_matrix = self.preprocessor.create_feature_matrix()
                    st.success(f"Created feature matrix with shape {feature_matrix.shape}.")

                # Show data
                st.subheader("T2DM Patient Cohort")
                st.dataframe(t2dm_cohort.head())

                st.subheader("Order Data Sample")
                st.dataframe(orders_data.head())

                st.subheader("Feature Matrix Sample")
                st.dataframe(feature_matrix.head())

                # Distribution of length of stay
                st.subheader("Length of Stay Distribution")
                fig = px.histogram(
                    t2dm_cohort,
                    x='los_days',
                    nbins=50,
                    title='Distribution of Length of Stay for T2DM Patients'
                )
                st.plotly_chart(fig)

                # Order type distribution
                st.subheader("Order Type Distribution")
                order_counts = orders_data['order_type'].value_counts().reset_index()
                order_counts.columns = ['Order Type', 'Count']

                fig = px.bar(
                    order_counts.head(20),
                    x='Count',
                    y='Order Type',
                    orientation='h',
                    title='Top 20 Most Common Order Types'
                )
                st.plotly_chart(fig)

                # Set preprocessing done flag
                self.preprocessing_done = True

            except Exception as e:
                st.error(f"Error during preprocessing: {str(e)}")
                st.exception(e)

    def render_clustering_page(self):
        """Render the clustering analysis page"""
        st.title("Clustering Analysis")

        if not self.preprocessing_done:
            st.warning("Please complete data preprocessing first.")
            return

        st.markdown(f"""
        ## Clustering Analysis

        Apply {self.cluster_method} clustering to identify patterns in provider orders.
        """)

        if st.button("Run Clustering"):
            try:
                # Initialize clustering
                with st.spinner("Initializing clustering..."):
                    self.clustering = ClusteringAnalysis(self.preprocessor.feature_matrix)

                # Apply dimensionality reduction
                with st.spinner(f"Applying {self.dim_reduction_method} dimensionality reduction..."):
                    method_mapping = {'PCA': 'pca', 't-SNE': 'tsne', 'UMAP': 'umap', 'MDS': 'mds'}
                    reduced_features, reducer = self.clustering.apply_dimensionality_reduction(
                        method=method_mapping[self.dim_reduction_method]
                    )

                # Apply clustering
                with st.spinner(f"Applying {self.cluster_method} clustering..."):
                    if self.cluster_method == "LDA":
                        cluster_labels, transformed_data, model = self.clustering.apply_lda(
                            n_components=self.n_topics,
                            max_iter=self.lda_max_iter
                        )

                    elif self.cluster_method == "K-means":
                        cluster_labels, model = self.clustering.apply_kmeans(
                            n_clusters=self.n_clusters
                        )

                    elif self.cluster_method == "DBSCAN":
                        cluster_labels, model = self.clustering.apply_dbscan(
                            eps=self.eps,
                            min_samples=self.min_samples
                        )

                    elif self.cluster_method == "Hierarchical":
                        cluster_labels, model = self.clustering.apply_hierarchical(
                            n_clusters=self.n_clusters_hc,
                            linkage=self.linkage
                        )

                # Evaluate clustering
                with st.spinner("Evaluating clustering quality..."):
                    metrics = self.clustering.evaluate_clusters()

                    st.success(f"Clustering complete! Found {metrics['num_clusters']} clusters.")

                    # Show metrics
                    st.subheader("Clustering Metrics")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            label="Number of Clusters",
                            value=metrics['num_clusters']
                        )

                    with col2:
                        st.metric(
                            label="Silhouette Score",
                            value=f"{metrics['silhouette_score']:.4f}"
                        )

                # Initialize visualizer
                self.visualizer = DataVisualizer(
                    self.preprocessor.feature_matrix,
                    self.clustering.cluster_labels,
                    self.clustering.reduced_features
                )

                # Visualize clusters
                st.subheader("Cluster Visualization")
                fig = self.visualizer.plot_reduced_dimensions(method_name=self.dim_reduction_method)
                st.plotly_chart(fig)

                # Show cluster distribution
                cluster_counts = pd.Series(self.clustering.cluster_labels).value_counts().reset_index()
                cluster_counts.columns = ['Cluster', 'Count']

                fig = px.pie(
                    cluster_counts,
                    values='Count',
                    names='Cluster',
                    title='Distribution of Clusters'
                )
                st.plotly_chart(fig)

                # Set clustering done flag
                self.clustering_done = True

                # Initialize feature analyzer
                self.feature_analyzer = FeatureImportanceAnalyzer(
                    self.preprocessor.feature_matrix,
                    self.clustering.cluster_labels
                )

            except Exception as e:
                st.error(f"Error during clustering: {str(e)}")
                st.exception(e)

    def render_outcome_analysis_page(self):
        """Render the outcome analysis page"""
        st.title("Outcome Analysis")

        if not self.clustering_done:
            st.warning("Please complete clustering analysis first.")
            return

        st.markdown("""
        ## Outcome Analysis

        Analyze how length of stay and other outcomes vary across the identified clusters.
        """)

        try:
            # Analyze outcomes by cluster
            with st.spinner("Analyzing outcomes by cluster..."):
                cluster_stats, f_stat, p_value = self.clustering.analyze_outcomes_by_cluster()

                st.success("Outcome analysis complete!")

                # Show ANOVA results
                st.subheader("Statistical Analysis")
                st.markdown(f"""
                **ANOVA Test for Differences in Length of Stay Between Clusters:**
                - F-statistic: {f_stat:.4f}
                - p-value: {p_value:.4f}
                - Interpretation: {"There are significant differences between clusters" if p_value < 0.05 else "No significant differences between clusters"}
                """)

                # Show outcome statistics by cluster
                st.subheader("Length of Stay Statistics by Cluster")
                st.dataframe(cluster_stats)

                # Visualize LOS by cluster
                st.subheader("Length of Stay Distribution by Cluster")
                fig = self.visualizer.plot_los_by_cluster()
                st.plotly_chart(fig)

                # Feature importance for predicting LOS
                with st.spinner("Analyzing feature importance for length of stay..."):
                    feature_importance, rf_model, r2_train, r2_test = self.feature_analyzer.predict_los_from_features()

                    st.subheader("Feature Importance for Length of Stay")
                    st.markdown(f"""
                    **Random Forest Model Performance:**
                    - R² (Training): {r2_train:.4f}
                    - R² (Testing): {r2_test:.4f}
                    """)

                    # Visualize feature importance
                    fig = self.visualizer.plot_feature_importance(feature_importance)
                    st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error during outcome analysis: {str(e)}")
            st.exception(e)

    def render_interpretation_page(self):
        """Render the cluster interpretation page"""
        st.title("Cluster Interpretation")

        if not self.clustering_done:
            st.warning("Please complete clustering analysis first.")
            return

        st.markdown("""
        ## Cluster Interpretation

        Interpret what characterizes each cluster and how it relates to outcomes.
        """)

        try:
            # Characterize clusters
            with st.spinner("Characterizing clusters..."):
                cluster_profiles = self.feature_analyzer.characterize_clusters()

                st.success("Cluster characterization complete!")

                # Show cluster characteristics
                st.subheader(f"Characteristics of Cluster {self.selected_cluster}")

                # Find the profile for the selected cluster
                selected_profile = None
                for profile in cluster_profiles:
                    if profile['cluster_id'] == self.selected_cluster:
                        selected_profile = profile
                        break

                if selected_profile:
                    # Show basic stats
                    st.markdown(f"""
                    **Cluster Size:** {selected_profile['size']} patients
                    """)

                    # Show distinctive features
                    fig = self.visualizer.plot_cluster_characteristics(cluster_profiles, self.selected_cluster)
                    st.plotly_chart(fig)

                    # Temporal patterns
                    st.subheader("Temporal Order Patterns")
                    fig = self.visualizer.plot_temporal_patterns(self.preprocessor.orders_data)
                    st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error during cluster interpretation: {str(e)}")
            st.exception(e)

    def run(self):
        """Run the Streamlit app"""
        # Set up sidebar
        self.setup_sidebar()

        # Render appropriate page based on navigation
        if self.page == "Home":
            self.render_home_page()
        elif self.page == "Data Loading":
            self.render_data_loading_page()
        elif self.page == "Preprocessing":
            self.render_preprocessing_page()
        elif self.page == "Clustering Analysis":
            self.render_clustering_page()
        elif self.page == "Outcome Analysis":
            self.render_outcome_analysis_page()
        elif self.page == "Cluster Interpretation":
            self.render_interpretation_page()


# ===================================
# Main Application Entry Point
# ===================================

if __name__ == "__main__":
    app = StreamlitAppClaude
()
    app.run()
