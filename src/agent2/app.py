import streamlit as st
import os
from pathlib import Path
import pandas as pd
import numpy as np

from data.data_loader import DataLoader
from models.feature_engineering import FeatureEngineer
from models.clustering import ClusteringAnalyzer
from visualization.visualizer import Visualizer

# Default dataset path
DEFAULT_DATASET_PATH = "/Users/artinmajdi/Documents/GitHubs/Career/mimic_iv/dataset/mimic-iv-3.1"


class StreamlitApp:
	"""
	Streamlit application for analyzing provider order patterns in the MIMIC-IV 3.1 database.
	"""

	def __init__(self):
		self.data_loader         = None
		self.feature_engineer    = None
		self.clustering_analyzer = None
		self.visualizer          = None
		self.data_loaded         = False

	@staticmethod
	def get_dataset_path():
		"""Get the dataset path from environment variable or user input"""
		# Try to get path from environment variable
		dataset_path = os.getenv('MIMIC_DATASET_PATH')

		# If not set, use default
		if dataset_path is None:
			dataset_path = DEFAULT_DATASET_PATH

		return dataset_path


	@staticmethod
	def show_data_overview(data_loader):
		"""
		Display an overview of the dataset.
		"""
		st.header("Data Overview")

		if data_loader is None:
			st.info("Please load the dataset first by clicking the 'Load Dataset' button in the sidebar.")
			return

		# Load and display basic statistics
		patients = data_loader.load_patients()
		admissions = data_loader.load_admissions()
		diagnoses = data_loader.load_diagnoses()

		st.subheader("Dataset Statistics")
		col1, col2, col3 = st.columns(3)

		with col1:
			st.metric("Total Patients", len(patients))
		with col2:
			st.metric("Total Admissions", len(admissions))
		with col3:
			st.metric("Total Diagnoses", len(diagnoses))

		# Show T2DM patient statistics
		t2dm_patients = data_loader.get_t2dm_patients()
		st.subheader("Type 2 Diabetes Patient Statistics")
		st.metric("Number of T2DM Patients", len(t2dm_patients))


	@staticmethod
	def show_clustering_analysis():
		st.header("Clustering Analysis")
		if not st.session_state.data_loaded:
			st.info("Please load the dataset first by clicking the 'Load Dataset' button in the sidebar.")
			return
		st.write("This section allows you to perform clustering analysis on provider order patterns.")


	@staticmethod
	def show_visualization():
		st.header("Visualization")
		if not st.session_state.data_loaded:
			st.info("Please load the dataset first by clicking the 'Load Dataset' button in the sidebar.")
			return
		st.write("This section provides interactive visualizations of the analysis results.")


	@staticmethod
	def show_statistical_analysis():
		st.header("Statistical Analysis")
		if not st.session_state.data_loaded:
			st.info("Please load the dataset first by clicking the 'Load Dataset' button in the sidebar.")
			return
		st.write("This section provides statistical analysis of the clustering results.")


	def run(self):
		st.set_page_config(
			page_title="MIMIC-IV Provider Order Analysis",
			page_icon="🏥",
			layout="wide"
		)

		st.title("MIMIC-IV Provider Order Pattern Analysis")
		st.markdown("""
		This application analyzes provider order patterns in the MIMIC-IV 3.1 database to identify
		clusters associated with shorter length of stay for Type 2 Diabetes patients.
		""")

		# Initialize session state for data loading
		if 'data_loaded' not in st.session_state:
			st.session_state.data_loaded = False
		if 'data_loader' not in st.session_state:
			st.session_state.data_loader = None
		if 'feature_engineer' not in st.session_state:
			st.session_state.feature_engineer = None
		if 'clustering_analyzer' not in st.session_state:
			st.session_state.clustering_analyzer = None
		if 'visualizer' not in st.session_state:
			st.session_state.visualizer = None

		# Dataset path input
		st.sidebar.title("Dataset Configuration")
		dataset_path = st.sidebar.text_input(
			"MIMIC-IV Dataset Path",
			value=StreamlitApp.get_dataset_path(),
			help="Enter the path to your MIMIC-IV 3.1 dataset directory"
		)

		# Load Dataset button
		if st.sidebar.button("Load Dataset"):
			with st.spinner("Initializing data loader..."):
				try:
					st.session_state.data_loader = DataLoader(dataset_path)
					st.session_state.feature_engineer = FeatureEngineer()
					st.session_state.clustering_analyzer = ClusteringAnalyzer()
					st.session_state.visualizer = Visualizer()
					st.session_state.data_loaded = True
					st.success("Dataset loaded successfully!")
				except Exception as e:
					st.error(f"Error loading dataset: {str(e)}")
					st.session_state.data_loaded = False

		# Sidebar
		st.sidebar.title("Analysis Parameters")

		# Only show analysis parameters if data is loaded
		if st.session_state.data_loaded:
			clustering_method = st.sidebar.selectbox(
				"Clustering Method",
				["kmeans", "dbscan", "hierarchical", "lda", "autoencoder_kmeans"],
				help="""Select clustering method:
				- kmeans: Standard K-means clustering
				- dbscan: Density-based clustering
				- hierarchical: Hierarchical clustering
				- lda: Latent Dirichlet Allocation for topic modeling
				- autoencoder_kmeans: Autoencoder for dimensionality reduction followed by K-means"""
			)

			n_clusters = st.sidebar.slider(
				"Number of Clusters",
				min_value=2,
				max_value=10,
				value=5,
				step=1,
				help="Number of clusters to identify (for kmeans, hierarchical, and autoencoder_kmeans)"
			)

			use_dim_reduction = st.sidebar.checkbox(
				"Use Dimensionality Reduction",
				value=True,
				help="Apply dimensionality reduction before clustering (except for LDA and Autoencoder which have built-in dimensionality reduction)"
			)

			# Additional parameters for specific methods
			if clustering_method == "lda":
				n_topics = st.sidebar.slider(
					"Number of Topics",
					min_value=2,
					max_value=20,
					value=5,
					step=1,
					help="Number of topics to identify in LDA"
				)
				n_iterations = st.sidebar.slider(
					"Number of Iterations",
					min_value=100,
					max_value=1000,
					value=500,
					step=100,
					help="Number of iterations for LDA training"
				)
			elif clustering_method == "autoencoder_kmeans":
				encoding_dim = st.sidebar.slider(
					"Encoding Dimension",
					min_value=2,
					max_value=50,
					value=10,
					step=1,
					help="Dimension of the encoded representation"
				)
				epochs = st.sidebar.slider(
					"Training Epochs",
					min_value=10,
					max_value=100,
					value=50,
					step=10,
					help="Number of training epochs for the autoencoder"
				)
			elif clustering_method == "dbscan":
				eps = st.sidebar.slider(
					"Epsilon",
					min_value=0.1,
					max_value=5.0,
					value=0.5,
					step=0.1,
					help="Maximum distance between two samples for them to be considered neighbors"
				)
				min_samples = st.sidebar.slider(
					"Minimum Samples",
					min_value=2,
					max_value=20,
					value=5,
					step=1,
					help="Number of samples in a neighborhood for a point to be considered a core point"
				)

			# Main content
			if st.sidebar.button("Run Analysis"):
				with st.spinner("Loading data..."):
					try:
						# Load data
						t2dm_patients = st.session_state.data_loader.get_t2dm_patients()
						patient_orders = st.session_state.data_loader.get_patient_orders(t2dm_patients)

						# Compute features
						features, sequences = st.session_state.feature_engineer.prepare_features(patient_orders)

						# Perform clustering
						clusters, metrics = st.session_state.clustering_analyzer.analyze_clusters(
							features,
							method=clustering_method,
							n_clusters=n_clusters,
							use_dim_reduction=use_dim_reduction
						)

						# Create visualizations
						figures = st.session_state.visualizer.create_dashboard(
							features,
							clusters,
							st.session_state.feature_engineer.create_order_frequency_matrix(patient_orders),
							st.session_state.data_loader.load_admissions(),
							sequences,
							metrics
						)

						# Display results
						st.header("Analysis Results")

						# Show metrics
						st.subheader("Clustering Metrics")
						metrics_df = pd.DataFrame({
							'Metric': list(metrics.keys()),
							'Value': list(metrics.values())
						})
						st.dataframe(metrics_df)

						# Show visualizations
						st.subheader("Visualizations")
						for fig in figures:
							st.plotly_chart(fig, use_container_width=True)

						# Show cluster characteristics
						st.subheader("Cluster Characteristics")
						cluster_stats = pd.DataFrame({
							'Cluster': clusters,
							'Patient Count': np.bincount(clusters[clusters >= 0])
						})
						st.dataframe(cluster_stats)
					except Exception as e:
						st.error(f"Error during analysis: {str(e)}")
						st.stop()
		else:
			st.info("Please load the dataset first by clicking the 'Load Dataset' button in the sidebar.")

		# Data Overview
		st.sidebar.title("Navigation")
		page = st.sidebar.radio(
			"Select a page",
			["Data Overview", "Clustering Analysis", "Visualization", "Statistical Analysis"]
		)

		if page == "Data Overview":
			StreamlitApp.show_data_overview(st.session_state.data_loader if st.session_state.data_loaded else None)
		elif page == "Clustering Analysis":
			StreamlitApp.show_clustering_analysis()
		elif page == "Visualization":
			StreamlitApp.show_visualization()
		elif page == "Statistical Analysis":
			StreamlitApp.show_statistical_analysis()


if __name__ == "__main__":
	app = StreamlitApp()
	app.run()
