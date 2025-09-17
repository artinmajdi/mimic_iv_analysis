# Standard library imports
import os
import logging

# Data processing imports
import numpy as np
import pandas as pd

# Visualization imports
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Machine learning imports
from scipy.cluster.hierarchy import dendrogram

from sklearn.metrics import adjusted_rand_score

# Streamlit import
import streamlit as st

# Local application imports
from mimic_iv_analysis.core.clustering import ClusteringAnalyzer
from mimic_iv_analysis.core.feature_engineering import FeatureEngineerUtils
from mimic_iv_analysis.visualization.visualizer_utils import display_dataframe_head

RANDOM_STATE = 42


class ClusteringAnalysisTab:
	""" Handles the UI and logic for the Clustering Analysis tab. """

	def __init__(self):
		self.clustering_analyzer = ClusteringAnalyzer()

	def _find_optimal_k(self, k_min, k_max, n_init, max_iter, data_for_clustering):

		if st.button("Find Optimal k (using Elbow & Silhouette)"):

			if k_max <= k_min:
				st.error("Maximum k must be greater than Minimum k.")
				return

			with st.spinner(f"Calculating Elbow and Silhouette scores for k={k_min} to {k_max}..."):

				k_metrics, optimal_k_silhouette = self.clustering_analyzer.find_optimal_k_kmeans_elbow_silhouette( data=data_for_clustering, k_range=range(k_min, k_max + 1), n_init=n_init, max_iter=max_iter )

				# Suggest optimal k based on silhouette (usually more reliable than elbow visually)
				st.session_state.optimal_k = int(optimal_k_silhouette) # Store best k

				st.success(f"Optimal k suggested by Silhouette Score: {st.session_state.optimal_k}")

				titles = {
					'inertia': {
						'title'  : "Elbow Method for Optimal k",
						'y_title': "Inertia (Within-cluster sum of squares)",
						'name'   : 'Inertia'
					},
					'silhouette': {
						'title'  : "Silhouette Score for Different k",
						'y_title': "Average Silhouette Score",
						'name'   : 'Silhouette Score'
					}
				}

				for metric in ['inertia', 'silhouette']:
					fig = go.Figure()
					fig.add_trace(
						go.Scatter(
							x    = k_metrics['k'],
							y    = k_metrics[metric],
							mode = 'lines+markers',
							name = titles[metric]['name']
						)
					)

					fig.update_layout(
						title       = titles[metric]['title'],
						xaxis_title = "Number of Clusters (k)",
						yaxis_title = titles[metric]['y_title']
					)

					annotation_text = "Suggested k" if metric == 'inertia' else "Optimal k"
					fig.add_vline(
						x               = st.session_state.optimal_k,
						line_dash       = "dash",
						line_color      = "red",
						annotation_text = f"{annotation_text} = {st.session_state.optimal_k}"
					)

					st.plotly_chart(fig, use_container_width=True)

				# Update n_clusters input to the found optimal k
				# st.rerun() # Rerun to update the number input widget value

	def _data_selection(self):

		st.markdown("<h3>Select Input Data for Clustering</h3>", unsafe_allow_html=True)

		# Option to use the current DataFrame or a feature matrix
		data_source_options = ["Current DataFrame", "Order Frequency Matrix", "Order Timing Features", "Upload Data"]
		# Determine default index based on available features
		default_data_source_index = 0
		if 'freq_matrix' in st.session_state and st.session_state.freq_matrix is not None:
			default_data_source_index = 1
		elif 'timing_features' in st.session_state and st.session_state.timing_features is not None:
			default_data_source_index = 2

		data_source = st.radio( "Select Data Source", data_source_options, index=default_data_source_index, horizontal=True )

		input_data = None
		input_data_ready = False # Flag to track if data is loaded and ready

		if data_source == "Current DataFrame":

			# Let user select columns from the current DataFrame
			if st.session_state.df is not None:

				# Get numeric columns only for clustering
				numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()

				if numeric_cols:
					default_selection = numeric_cols[:min(5, len(numeric_cols))]
					selected_cols = st.multiselect( "Select numeric columns for clustering", numeric_cols, default=default_selection )

					if selected_cols:
						input_data = st.session_state.df[selected_cols].copy()
						st.markdown(f"Selected data shape: {input_data.shape[0]} rows × {input_data.shape[1]} columns")
						display_dataframe_head(input_data)
						input_data_ready = True

					else:
						st.warning("Please select at least one numeric column.")

				else:
					st.warning("No numeric columns found in the current DataFrame. Please select another data source or load a table with numeric data.")
			else:
				st.warning("No DataFrame is currently loaded. Please load a dataset first.")

		elif data_source == "Order Frequency Matrix":
			# Use order frequency matrix if available
			if 'freq_matrix' in st.session_state and st.session_state.freq_matrix is not None:
				input_data = st.session_state.freq_matrix.copy() # Use copy
				st.markdown(f"Using order frequency matrix with shape: {input_data.shape[0]} patients × {input_data.shape[1]} order types")
				display_dataframe_head(input_data)
				input_data_ready = True
			else:
				st.warning("Order frequency matrix not found. Please generate it in the Feature Engineering tab first.")

		elif data_source == "Order Timing Features":
			# Use timing features if available
			if 'timing_features' in st.session_state and st.session_state.timing_features is not None:
				# Get numeric columns only
				numeric_cols = st.session_state.timing_features.select_dtypes(include=np.number).columns.tolist()

				if numeric_cols:
					selected_cols = st.multiselect(
						"Select timing features for clustering",
						numeric_cols,
						default=numeric_cols # Default to all numeric timing features
					)

					if selected_cols:
						input_data = st.session_state.timing_features[selected_cols].copy()
						st.markdown(f"Selected data shape: {input_data.shape[0]} rows × {input_data.shape[1]} columns")
						display_dataframe_head(input_data)
						input_data_ready = True
					else:
						st.warning("Please select at least one timing feature.")
				else:
					st.warning("No numeric columns found in the Order Timing Features. Please generate them first.")
			else:
				st.warning("Order timing features not found. Please generate them in the Feature Engineering tab first.")

		elif data_source == "Upload Data":
			# Allow user to upload a CSV or Parquet file
			uploaded_file = st.file_uploader("Upload CSV or Parquet file", type=["csv", "parquet"])

			if uploaded_file is not None:
				try:
					if uploaded_file.name.endswith('.csv'):
						input_data = pd.read_csv(uploaded_file)
					elif uploaded_file.name.endswith('.parquet'):
						input_data = pd.read_parquet(uploaded_file)

					# Basic validation: check for numeric data
					if input_data.select_dtypes(include=np.number).empty:
						st.error("Uploaded file contains no numeric columns suitable for clustering.")
						input_data = None
					else:
						st.markdown(f"Uploaded data shape: {input_data.shape[0]} rows × {input_data.shape[1]} columns")
						display_dataframe_head(input_data)

						# Select only numeric columns
						input_data = input_data.select_dtypes(include=np.number)
						st.info(f"Using {input_data.shape[1]} numeric columns for clustering.")
						input_data_ready = True
				except Exception as e:
					st.error(f"Error loading file: {str(e)}")
					logging.exception("Error loading uploaded clustering data")

		# Data preprocessing options (only if data is ready)
		if input_data_ready:
			st.markdown("<h4>Data Preprocessing</h4>", unsafe_allow_html=True)

			preprocess_col1, preprocess_col2 = st.columns(2)

			with preprocess_col1:
				preprocess_method = st.selectbox(
					"Preprocessing Method",
					["None", "Standard Scaling", "Min-Max Scaling", "Normalization"],
					index=1, # Default to Standard Scaling
					help="Select method to preprocess the data"
				)

			with preprocess_col2:
				handle_missing = st.selectbox(
					"Handle Missing Values",
					["Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Zero"],
					index=1, # Default to Fill with Mean
					help="Select method to handle missing values"
				)

			# Map selections to parameter values
			preprocess_method_map = {
				"None": None,
				"Standard Scaling": "standard",
				"Min-Max Scaling": "minmax",
				"Normalization": "normalize"
			}

			handle_missing_map = {
				"Drop Rows": "drop",
				"Fill with Mean": "mean",
				"Fill with Median": "median",
				"Fill with Zero": "zero"
			}

			# Button to process and store the data
			if st.button("Prepare Data for Clustering"):
				try:
					with st.spinner("Preprocessing data..."):
						# Check if Dask was used to load the data
						use_dask = st.session_state.get('use_dask', False)

						# Apply preprocessing
						processed_data = self.clustering_analyzer.preprocess_data(
							input_data, # Pass the loaded data
							method=preprocess_method_map[preprocess_method],
							handle_missing=handle_missing_map[handle_missing],
							use_dask=use_dask
						)

						# Check if data remains after preprocessing
						if processed_data.empty:
							st.error("Preprocessing resulted in an empty dataset. Check your missing value handling strategy (e.g., 'Drop Rows' might remove all data).")
						else:
							st.session_state.clustering_input_data = processed_data
							# Clear any previous reduced data
							st.session_state.reduced_data = None
							st.success(f"Data preprocessed and ready for clustering! Shape: {processed_data.shape}")

							# Show preview of processed data
							display_dataframe_head(processed_data)

				except AttributeError:
					st.error("Clustering Analyzer is not properly initialized or does not have a 'preprocess_data' method.")
				except Exception as e:
					st.error(f"Error preparing data: {str(e)}")
					logging.exception("Error in Prepare Data for Clustering")

		elif not input_data_ready and data_source != "Upload Data":
			# Show message if no data source could be loaded (excluding upload state)
			st.info("Select a data source and ensure it's available or upload a file.")


	def _dimensionality_reduction(self):
		st.markdown("<h3>Dimensionality Reduction</h3>", unsafe_allow_html=True)

		# Check if input data is available
		if 'clustering_input_data' in st.session_state and st.session_state.clustering_input_data is not None:
			input_data_for_reduction = st.session_state.clustering_input_data
			input_shape = input_data_for_reduction.shape

			st.markdown(f"""
			<div class='info-box'>
			Reduce the dimensionality of your data ({input_shape[0]} rows × {input_shape[1]} columns) to visualize and potentially improve clustering performance.
			</div>
			""", unsafe_allow_html=True)

			# Check if data has enough features for reduction
			if input_shape[1] <= 2:
				st.info("Data already has 2 or fewer dimensions. Dimensionality reduction is not applicable.")
			else:
				# Dimensionality reduction method selection
				reduction_col1, reduction_col2 = st.columns(2)

				with reduction_col1:
					reduction_method = st.selectbox(
						"Dimensionality Reduction Method",
						["PCA", "t-SNE", "UMAP"], # SVD often used differently, removed for clustering context
						index=0,
						help="Select method to reduce dimensions"
					)

				with reduction_col2:
					# Ensure n_components is less than original dimensions
					max_components = min(10, input_shape[1] -1) # At least reduce by 1
					n_components = st.number_input(
						"Number of Components",
						min_value=2,
						max_value=max_components,
						value=2,
						help=f"Number of dimensions to reduce to (max {max_components})"
					)

				# Method-specific parameters
				extra_params = {}
				if reduction_method == "t-SNE":
					tsne_col1, tsne_col2 = st.columns(2)
					with tsne_col1:
						perplexity = st.slider("Perplexity", min_value=5, max_value=50, value=min(30, input_shape[0]-1), help="Balance between local/global structure (must be < n_samples)")
					with tsne_col2:
						learning_rate = st.slider("Learning Rate", min_value=10, max_value=1000, value=200, step=10, help="Learning rate for t-SNE")
					n_iter = st.slider("Max Iterations", min_value=250, max_value=2000, value=1000, step=250, help="Maximum number of iterations")
					extra_params = {"perplexity": perplexity, "learning_rate": learning_rate, "n_iter": n_iter}

				elif reduction_method == "UMAP":
					try:
						umap_col1, umap_col2 = st.columns(2)
						with umap_col1:
							n_neighbors = st.slider("Number of Neighbors", min_value=2, max_value=min(100, input_shape[0]-1), value=min(15, input_shape[0]-1), help="Controls local/global embedding (must be < n_samples)")
						with umap_col2:
							min_dist = st.slider("Minimum Distance", min_value=0.0, max_value=0.99, value=0.1, step=0.05, help="Controls how tightly points are packed")
						metric = st.selectbox("Distance Metric", ["euclidean", "manhattan", "cosine", "correlation"], index=0, help="Metric used for distances")
						extra_params = {"n_neighbors": n_neighbors, "min_dist": min_dist, "metric": metric}
					except ImportError:
						st.warning("UMAP is not installed. Please install it (`pip install umap-learn`) to use this option.")
						reduction_method = "PCA" # Fallback
						st.info("Falling back to PCA.")
						extra_params = {}


				# Button to apply dimensionality reduction
				if st.button("Apply Dimensionality Reduction"):
					try:
						with st.spinner(f"Applying {reduction_method} dimensionality reduction..."):
							# Map method names
							method_map = {"PCA": "pca", "t-SNE": "tsne", "UMAP": "umap"}

							# Check if Dask was used to load the data
							use_dask = st.session_state.get('use_dask', False)

							# Apply reduction
							reduced_data = self.clustering_analyzer.apply_dimensionality_reduction(
								input_data_for_reduction,
								method=method_map[reduction_method],
								n_components=n_components,
								**extra_params
							)

							# Store reduced data
							st.session_state.reduced_data = reduced_data

							# Show success message
							st.success(f"Dimensionality reduction complete! Reduced from {input_shape[1]} to {n_components} dimensions.")

							# Show preview
							display_dataframe_head(reduced_data)

					except AttributeError:
						st.error("Clustering Analyzer is not properly initialized or does not have an 'apply_dimensionality_reduction' method.")
					except ValueError as e:
						st.error(f"Value Error during reduction: {e}. Check parameters (e.g., perplexity/n_neighbors vs sample size).")
					except Exception as e:
						st.error(f"Error applying dimensionality reduction: {str(e)}")
						logging.exception("Error in Apply Dimensionality Reduction")


				# Visualization of reduced data (if available and 2D/3D)
				if 'reduced_data' in st.session_state and st.session_state.reduced_data is not None:
					reduced_df = st.session_state.reduced_data
					reduced_shape = reduced_df.shape

					st.markdown("<h4>Visualization of Reduced Data</h4>", unsafe_allow_html=True)
					try:
						if reduced_shape[1] == 2:
							fig = px.scatter(
								reduced_df, x=reduced_df.columns[0], y=reduced_df.columns[1],
								title=f"2D Projection using {reduction_method}",
								opacity=0.7
							)
							st.plotly_chart(fig, use_container_width=True)
						elif reduced_shape[1] == 3:
							fig = px.scatter_3d(
								reduced_df, x=reduced_df.columns[0], y=reduced_df.columns[1], z=reduced_df.columns[2],
								title=f"3D Projection using {reduction_method}",
								opacity=0.7
							)
							st.plotly_chart(fig, use_container_width=True)
						else:
							st.info("Reduced data has more than 3 dimensions. Select 2 or 3 components for visualization.")
					except Exception as e:
						st.error(f"Error visualizing reduced data: {e}")


					# Option to save reduced data using FeatureEngineer's save method
					with st.expander("Save Reduced Data"):
						save_format_reduced = st.radio(
														label   = "Save Format ", # Key differentiation
														options = ["CSV", "Parquet"],
														horizontal=True,
														key     = "dimreduction_save_format"
													)
						if st.button("Save Reduced Data"):

							try:
								base_path_save = "."
								if 'current_file_path' in st.session_state and st.session_state.current_file_path:
									potential_path_save = os.path.dirname(st.session_state.current_file_path)
									if os.path.isdir(potential_path_save):
										base_path_save = potential_path_save

								filepath = FeatureEngineerUtils.save_features(
									features     = st.session_state.reduced_data,
									feature_type = f"{reduction_method.lower()}_reduced_data",
									base_path    = base_path_save,
									format       = save_format_reduced.lower()
								)
								st.success(f"Saved reduced data to {filepath}")

							except AttributeError:
								st.error("Feature Engineer is not properly initialized or does not have a 'save_features' method.")
							except Exception as e:
								st.error(f"Error saving reduced data: {str(e)}")
								logging.exception("Error saving reduced data")
		else:
			st.warning("No input data available for clustering. Please prepare data in the 'Data Selection' tab first.")

	def _run_k_means_clustering(self, n_clusters, data_for_clustering, n_init, max_iter):

		if st.button("Run K-means Clustering"):

			# try:
			with st.spinner(f"Running K-means clustering with k={n_clusters}..."):

				# Run K-means
				labels, kmeans_model = self.clustering_analyzer.run_kmeans_clustering( data=data_for_clustering, n_clusters=n_clusters, n_init=n_init, max_iter=max_iter )

				# Store labels in session state
				st.session_state.kmeans_labels = pd.Series(labels, index=data_for_clustering.index, name="kmeans_cluster")

				# Calculate metrics
				metrics = self.clustering_analyzer.evaluate_clustering( data=data_for_clustering, labels=st.session_state.kmeans_labels, method="kmeans" )

				# Store metrics centrally
				st.session_state.cluster_metrics['kmeans'] = metrics

				# Show success message with metrics
				metrics_text = "\n".join([f"- {k.replace('_', ' ').title()}: {v:.4f}" for k, v in metrics.items()])
				st.success(f"K-means clustering complete!\n{metrics_text}")

				# --- Visualization ---
				# Show cluster distribution
				cluster_counts = st.session_state.kmeans_labels.value_counts().sort_index()
				fig_dist = px.bar(
					x=cluster_counts.index, y=cluster_counts.values,
					labels={'x': 'Cluster', 'y': 'Number of Points'},
					title="Distribution of Points per Cluster (K-means)"
				)
				st.plotly_chart(fig_dist, use_container_width=True)

				# Visualize clusters if data is 2D or 3D
				if data_for_clustering.shape[1] in [2, 3]:
					vis_data = data_for_clustering.copy()
					vis_data['Cluster'] = st.session_state.kmeans_labels.astype(str) # Color needs categorical

					if data_for_clustering.shape[1] == 2:
						fig_scatter = px.scatter(
							vis_data, x=vis_data.columns[0], y=vis_data.columns[1], color='Cluster',
							title="K-means Clustering Results (2D)", color_discrete_sequence=px.colors.qualitative.G10
						)
						st.plotly_chart(fig_scatter, use_container_width=True)
					else: # 3D
						fig_scatter = px.scatter_3d(
							vis_data, x=vis_data.columns[0], y=vis_data.columns[1], z=vis_data.columns[2], color='Cluster',
							title="K-means Clustering Results (3D)", color_discrete_sequence=px.colors.qualitative.G10
						)
						st.plotly_chart(fig_scatter, use_container_width=True)

				# Show cluster centers
				if hasattr(kmeans_model, 'cluster_centers_'):
					st.markdown("<h4>Cluster Centers</h4>", unsafe_allow_html=True)
					centers = pd.DataFrame(
						kmeans_model.cluster_centers_,
						columns=data_for_clustering.columns,
						index=[f"Cluster {i}" for i in range(n_clusters)]
					)
					st.dataframe(centers.style.format("{:.3f}"), use_container_width=True)

			# except AttributeError:
			# 	st.error("Clustering Analyzer is not properly initialized or does not have the required methods.")
			# except Exception as e:
			# 	st.error(f"Error running K-means clustering: {str(e)}")
			# 	logging.exception("Error in Run K-means")

	def _save_results(self, data_for_clustering):
		if 'kmeans_labels' in st.session_state and st.session_state.kmeans_labels is not None:

			with st.expander("Save K-means Results"):
				save_col1, save_col2 = st.columns(2)

				with save_col1:
					if st.button("Save K-means Model"):

						try:
							base_path_save = "."
							if 'current_file_path' in st.session_state and st.session_state.current_file_path:

								potential_path_save = os.path.dirname(st.session_state.current_file_path)
								if os.path.isdir(potential_path_save):
									base_path_save = potential_path_save

							model_path = self.clustering_analyzer.save_model("kmeans", base_path_save)
							st.success(f"Saved K-means model to {model_path}")

						except AttributeError:
							st.error("Clustering Analyzer does not have a 'save_model' method or model is not available.")
						except Exception as e:
							st.error(f"Error saving K-means model: {str(e)}")

				with save_col2:
					if st.button("Save K-means Cluster Assignments"):

						try:
							# Create DataFrame with original index and cluster assignments
							assignments_df = pd.DataFrame(
								data  = {'cluster': st.session_state.kmeans_labels},
								index = data_for_clustering.index
							)

							base_path_save = "."
							if 'current_file_path' in st.session_state and st.session_state.current_file_path:
								potential_path_save = os.path.dirname(st.session_state.current_file_path)

								if os.path.isdir(potential_path_save):
									base_path_save = potential_path_save

							filepath = FeatureEngineerUtils.save_features(
								features     = assignments_df,
								feature_type = "kmeans_cluster_assignments",
								base_path    = base_path_save,
								format       = "csv"
							)

							st.success(f"Saved cluster assignments to {filepath}")

						except AttributeError:
							st.error("Feature Engineer does not have a 'save_features' method.")
						except Exception as e:
							st.error(f"Error saving K-means assignments: {str(e)}")

	def _kmeans_clustering(self):

		st.markdown("<h3>K-Means Clustering</h3>", unsafe_allow_html=True)

		# Check if input data is available
		if not ('clustering_input_data' in st.session_state and st.session_state.clustering_input_data is not None):
			st.warning("No input data available for clustering. Please prepare data in the 'Data Selection' tab first.")
			return

		st.markdown(""" <div class='info-box'> K-means clustering partitions data into k clusters, where each observation belongs to the cluster with the nearest mean. </div> """, unsafe_allow_html=True)

		# Determine which data to use
		data_for_clustering = st.session_state.clustering_input_data
		use_reduced_data    = False

		if 'reduced_data' in st.session_state and st.session_state.reduced_data is not None:
			use_reduced_data = st.checkbox(
				label     = "Use dimensionality-reduced data for clustering",
				value     = True, # Default to using reduced if available
				help      = "Use the reduced data instead of the original preprocessed data"
			)
			if use_reduced_data:
				data_for_clustering = st.session_state.reduced_data
				st.info(f"Using reduced data with shape: {data_for_clustering.shape}")
			else:
				st.info(f"Using preprocessed data with shape: {data_for_clustering.shape}")
		else:
			st.info(f"Using preprocessed data with shape: {data_for_clustering.shape}")

		# K-means parameters
		# Make sure n_clusters_default is an integer, not None
		optimal_k          = st.session_state.get('optimal_k')
		n_clusters_default = 5 if optimal_k is None else optimal_k

		n_clusters = st.number_input(
			label     = "Number of Clusters (k)",
			min_value = 2,
			max_value = max(20, n_clusters_default + 5), # Dynamic max based on optimal k
			value     = n_clusters_default,
			help      = "Number of clusters to form"
		)

		kmeans_params_col1, kmeans_params_col2 = st.columns(2)
		with kmeans_params_col1:
			max_iter = st.slider(
				label     = "Maximum Iterations",
				min_value = 100,
				max_value = 1000,
				value     = 300,
				step      = 100,
				help      = "Max iterations per run"
			)

		with kmeans_params_col2:
			n_init = st.slider(
				label     = "Number of Initializations",
				min_value = 1,
				max_value = 20,
				value     = 10,
				help      = "Number of runs with different seeds"
			)

		# --- Optimal k Section ---
		st.markdown("<h4>Find Optimal Number of Clusters (Elbow/Silhouette)</h4>", unsafe_allow_html=True)
		optimal_k_col1, optimal_k_col2 = st.columns(2)
		with optimal_k_col1:
			k_min = st.number_input(
				label     = "Minimum k",
				min_value = 2,
				max_value = 10,
				value     = 2
			)
		with optimal_k_col2:
			k_max = st.number_input(
				label     = "Maximum k",
				min_value = k_min + 1,
				max_value = 20,
				value     = 10
			)

		# Use optimal_k button
		self._find_optimal_k(k_min=k_min, k_max=k_max, n_init=n_init, max_iter=max_iter, data_for_clustering=data_for_clustering)

		# --- Run K-means Section ---
		st.markdown("<h4>Run K-means Clustering</h4>", unsafe_allow_html=True)
		self._run_k_means_clustering(n_clusters, data_for_clustering, n_init, max_iter)

		# --- Save Results Section ---
		self._save_results(data_for_clustering)

	def _hierarchical_clustering(self):
		st.markdown("<h3>Hierarchical Clustering</h3>", unsafe_allow_html=True)

		# Check if input data is available
		if 'clustering_input_data' in st.session_state and st.session_state.clustering_input_data is not None:
			st.markdown("""
			<div class='info-box'>
			Hierarchical clustering creates a tree of clusters (dendrogram) by progressively merging or splitting groups. It doesn't require specifying k beforehand but can be computationally intensive.
			</div>
			""", unsafe_allow_html=True)

			# Determine which data to use
			data_for_clustering = st.session_state.clustering_input_data
			use_reduced_data = False
			if 'reduced_data' in st.session_state and st.session_state.reduced_data is not None:
				use_reduced_data = st.checkbox(
					"Use dimensionality-reduced data for hierarchical clustering",
					value=True, # Default to using reduced if available
					help="Use the reduced data instead of the original preprocessed data"
				)
				if use_reduced_data:
					data_for_clustering = st.session_state.reduced_data
					st.info(f"Using reduced data with shape: {data_for_clustering.shape}")
				else:
					st.info(f"Using preprocessed data with shape: {data_for_clustering.shape}")
			else:
				st.info(f"Using preprocessed data with shape: {data_for_clustering.shape}")


			# Hierarchical clustering parameters
			hier_col1, hier_col2 = st.columns(2)
			with hier_col1:
				n_clusters_hier = st.number_input("Number of Clusters (for cutting dendrogram)", min_value=2, max_value=20, value=5, help="Number of clusters to extract after building the tree", key="hier_n_clusters")
			with hier_col2:
				linkage_method = st.selectbox("Linkage Method", ["ward", "complete", "average", "single"], index=0, help="Method for calculating distances between clusters")

			# Distance metric (affinity)
			distance_metric = st.selectbox("Distance Metric", ["euclidean", "manhattan", "cosine"], index=0, help="Metric for measuring distances between samples") # Removed correlation as less common for AgglomerativeClustering affinity

			# Ward linkage requires euclidean distance
			if linkage_method == "ward" and distance_metric != "euclidean":
				st.warning("Ward linkage requires Euclidean distance. Switching distance metric to 'euclidean'.")
				distance_metric = "euclidean"

			# Limit data size due to memory intensity
			max_samples_hier = 2000
			if len(data_for_clustering) > max_samples_hier:
				st.warning(f"Dataset size ({len(data_for_clustering)}) is large for hierarchical clustering. Using a random sample of {max_samples_hier} points to avoid memory issues.")
				data_for_clustering_hier = data_for_clustering.sample(max_samples_hier, random_state=RANDOM_STATE)
			else:
				data_for_clustering_hier = data_for_clustering


			# Button to run hierarchical clustering
			if st.button("Run Hierarchical Clustering"):
				if data_for_clustering_hier.empty:
					st.error("Cannot run clustering on empty data sample.")
				else:
					try:
						with st.spinner(f"Running hierarchical clustering ({linkage_method} linkage, {distance_metric} metric)..."):
							# Run hierarchical clustering
							labels, linkage_data = self.clustering_analyzer.run_hierarchical_clustering(
								data_for_clustering_hier,
								n_clusters=n_clusters_hier,
								linkage_method=linkage_method,
								distance_metric=distance_metric # Pass the selected metric
							)

							# Store labels in session state (using the index from the sampled data)
							st.session_state.hierarchical_labels = pd.Series(labels, index=data_for_clustering_hier.index, name="hierarchical_cluster")

							# Calculate metrics using the sampled data and labels
							metrics = self.clustering_analyzer.evaluate_clustering(
								data_for_clustering_hier,
								st.session_state.hierarchical_labels,
								"hierarchical" # Store metrics under 'hierarchical' key
							)
							# Store metrics centrally
							st.session_state.cluster_metrics['hierarchical'] = metrics

							# Show success message with metrics
							metrics_text = "\n".join([f"- {k.replace('_', ' ').title()}: {v:.4f}" for k, v in metrics.items()])
							st.success(f"Hierarchical clustering complete!\n{metrics_text}")

							# --- Visualization ---
							# Show cluster distribution (based on the sample)
							cluster_counts = st.session_state.hierarchical_labels.value_counts().sort_index()
							fig_dist = px.bar(
								x=cluster_counts.index, y=cluster_counts.values,
								labels={'x': 'Cluster', 'y': 'Number of Points'},
								title=f"Distribution of Points per Cluster (Hierarchical, Sample Size={len(data_for_clustering_hier)})"
							)
							st.plotly_chart(fig_dist, use_container_width=True)

							# Visualize clusters if data is 2D or 3D (using the sample)
							if data_for_clustering_hier.shape[1] in [2, 3]:
								vis_data = data_for_clustering_hier.copy()
								vis_data['Cluster'] = st.session_state.hierarchical_labels.astype(str)

								if data_for_clustering_hier.shape[1] == 2:
									fig_scatter = px.scatter(
										vis_data, x=vis_data.columns[0], y=vis_data.columns[1], color='Cluster',
										title=f"Hierarchical Clustering Results (2D, Sample Size={len(data_for_clustering_hier)})",
										color_discrete_sequence=px.colors.qualitative.G10
									)
									st.plotly_chart(fig_scatter, use_container_width=True)
								else: # 3D
									fig_scatter = px.scatter_3d(
										vis_data, x=vis_data.columns[0], y=vis_data.columns[1], z=vis_data.columns[2], color='Cluster',
										title=f"Hierarchical Clustering Results (3D, Sample Size={len(data_for_clustering_hier)})",
										color_discrete_sequence=px.colors.qualitative.G10
									)
									st.plotly_chart(fig_scatter, use_container_width=True)

							# Plot dendrogram
							st.markdown("<h4>Dendrogram</h4>", unsafe_allow_html=True)
							if linkage_data and 'linkage_matrix' in linkage_data:
								try:
									fig_dendro, ax = plt.subplots(figsize=(12, 7))
									dendrogram(
										linkage_data['linkage_matrix'],
										ax               = ax,
										truncate_mode    = 'lastp', # Show only the last p merged clusters
										p                = 12,      # Number of clusters to show at bottom
										show_leaf_counts = True,
										show_contracted  = True,
									)
									ax.set_title('Hierarchical Clustering Dendrogram (Truncated)')
									ax.set_xlabel('Cluster size (or sample index if leaf)')
									ax.set_ylabel('Distance')
									# Add cut line if n_clusters > 1
									if n_clusters_hier > 1 and len(linkage_data['linkage_matrix']) >= n_clusters_hier -1 :
										cut_distance = linkage_data['linkage_matrix'][-(n_clusters_hier-1), 2]
										ax.axhline(y=cut_distance, c='k', linestyle='--', label=f'Cut for {n_clusters_hier} clusters')
										ax.legend()
									st.pyplot(fig_dendro)
									plt.close(fig_dendro) # Close plot to free memory
								except Exception as e:
									st.error(f"Error plotting dendrogram: {e}")
							else:
								st.warning("Linkage data for dendrogram not available.")

					except AttributeError:
						st.error("Clustering Analyzer is not properly initialized or does not have the required methods.")
					except Exception as e:
						st.error(f"Error running hierarchical clustering: {str(e)}")
						logging.exception("Error in Run Hierarchical Clustering")


			# --- Save Results Section ---
			if 'hierarchical_labels' in st.session_state and st.session_state.hierarchical_labels is not None:
				with st.expander("Save Hierarchical Clustering Results"):
					st.info("Note: Saved assignments are based on the sampled data used for clustering.")
					save_col1, save_col2 = st.columns(2)
					# Hierarchical doesn't save a "model" in the same way as K-means/DBSCAN, usually just labels/linkage.
					# Skipping model save button here.
					with save_col2:
						if st.button("Save Hierarchical Cluster Assignments"):
							try:
								# Create DataFrame with sampled index and cluster assignments
								assignments_df = pd.DataFrame({'cluster': st.session_state.hierarchical_labels}, index=st.session_state.hierarchical_labels.index)

								base_path_save = "."
								if 'current_file_path' in st.session_state and st.session_state.current_file_path:

									potential_path_save = os.path.dirname(st.session_state.current_file_path)
									if os.path.isdir(potential_path_save):
										base_path_save = potential_path_save

								filepath = FeatureEngineerUtils.save_features( features=assignments_df, feature_type="hierarchical_cluster_assignments", base_path=base_path_save, format="csv" )

								st.success(f"Saved cluster assignments (for sample) to {filepath}")
							except AttributeError:
								st.error("Feature Engineer does not have a 'save_features' method.")
							except Exception as e:
								st.error(f"Error saving hierarchical assignments: {str(e)}")
		else:
			st.warning("No input data available for clustering. Please prepare data in the 'Data Selection' tab first.")

	def _dbscan_clustering(self):

		def dbscan_parameters() -> tuple[float, int, str, int]:
			"""
			Get the parameters for DBSCAN clustering.

			Returns:
				tuple[float, int, str, int]: A tuple containing the epsilon, minimum samples, metric, and k distance.
			"""

			eps_default = st.session_state.get('optimal_eps', 0.5) # Use optimal_eps if found, else 0.5


			# --- Epsilon ---
			eps = st.number_input(
				label   = "Epsilon (ε)",
				min_value = 0.01,
				max_value = max(eps_default if eps_default is not None else 0.5, 10.0),
				value     = eps_default,
				step      = 0.05,
				help      = "Maximum distance between samples to be neighbors",
				format    = "%.4f"
			)


			# --- Minimum Samples ---
			dbscan_col1, dbscan_col2 = st.columns(2)

			with dbscan_col1:
				min_samples = st.number_input(
					label     = "Minimum Samples",
					min_value = 2,
					max_value = 100,
					value     = max(5, int(0.01*len(data_for_clustering))),
					help      = "Number of samples in a neighborhood for a core point"
				) # Dynamic default suggestion

			with dbscan_col2:
				metric_dbscan = st.selectbox(
					label   = "Distance Metric",
					options = ["euclidean", "manhattan", "cosine", "l1", "l2"],
					index   = 0,
					help    = "Metric for distances",
					key     = "dbscan_metric"
				)

			# --- Number of Samples ---
			n_samples = st.number_input(
				label     = "Number of Samples",
				min_value = 2,
				max_value = 10000,
				value     = 1000,
				help      = "Number of samples to use for estimation" )


			# --- Optimal Epsilon (k-distance plot) ---
			st.markdown("<h4>Find Optimal Epsilon (ε) using k-distance plot</h4>", unsafe_allow_html=True)

			# Suggest k based on data dimensionality or min_samples. Heuristic based on literature or min_samples
			if data_for_clustering.shape[1] <= 2:
				k_dist_default = min_samples
			else:
				k_dist_default = min(max(min_samples, 2 * data_for_clustering.shape[1] - 1), len(data_for_clustering)-1)


			k_dist = st.slider(
				label   = "k for k-distance graph",
				min_value = 2,
				max_value = min(50, len(data_for_clustering)-1),
				value     = k_dist_default,
				help      = "Number of neighbors to consider (k = MinPts is common)"
			)

			return eps, min_samples, metric_dbscan, k_dist, n_samples


		def calculate_k_distance_plot(k_dist: int, n_samples: int, data_for_clustering: pd.DataFrame):
			"""
			Calculate the k-distance plot to find the optimal epsilon.

			Args:
				k_dist             : The number of neighbors to consider (k = MinPts is common).
				n_samples          : The number of samples to use for estimation.
				metric_dbscan      : The distance metric to use.
				data_for_clustering: The data to use for clustering.
			"""

			if st.button("Calculate k-distance plot to find 'knee'"):

				if k_dist >= len(data_for_clustering):
					st.error(f"k ({k_dist}) must be smaller than the number of samples ({len(data_for_clustering)}).")
					return None

				try:
					with st.spinner(f"Calculating {k_dist}-distance graph..."):

						# Find optimal epsilon using the analyzer method
						suggested_eps, k_distances_sorted = self.clustering_analyzer.find_optimal_eps_for_dbscan( data=data_for_clustering, k_dist=k_dist, n_samples=n_samples )

						# Store the suggested eps
						st.session_state.optimal_eps = suggested_eps

						# Show result
						st.success(f"Suggested epsilon (ε) based on the 'knee': {suggested_eps:.4f}")

						# Plot k-distance graph
						fig_kdist = go.Figure()
						fig_kdist.add_trace(go.Scatter(
														x    = list(range(len(k_distances_sorted))),
														y    = k_distances_sorted,
														mode = 'lines',
														name = f'{k_dist}-distance'
													))

						# Try to find the knee point mathematically (e.g., using Kneedle algorithm or max difference)
						# Simple max difference approach:
						diffs          = np.diff(k_distances_sorted, 1)
						knee_point_idx = np.argmax(diffs) + 1 # Index in the original sorted array
						knee_eps       = k_distances_sorted[knee_point_idx]

						fig_kdist.add_trace(go.Scatter(
														x      = [knee_point_idx],
														y      = [knee_eps],
														mode   = 'markers',
														marker = dict(size=10, color='red'),
														name   = f'Suggested ε ≈ {knee_eps:.4f}'
													))

						fig_kdist.update_layout(
												title       = f"{k_dist}-Distance Graph (Sorted)",
												xaxis_title = "Points (sorted by distance)",
												yaxis_title = f"{k_dist}-th Nearest Neighbor Distance",
												hovermode   = "x"
											)
						st.plotly_chart(fig_kdist, use_container_width=True)

						# Update eps input to the found optimal eps
						st.rerun() # Rerun to update the number input


				except AttributeError:
					st.error("Clustering Analyzer is not properly initialized or does not have a 'find_optimal_eps_for_dbscan' method.")
				except Exception as e:
					st.error(f"Error finding optimal epsilon: {str(e)}")
					logging.exception("Error in Find Optimal Epsilon (DBSCAN)")


		def run_dbscan_clustering(eps: float, min_samples: int, metric_dbscan: str, data_for_clustering: pd.DataFrame):
			"""
			Run DBSCAN clustering and calculate evaluation metrics.

			Args:
				eps: The epsilon value for DBSCAN.
				min_samples: The minimum number of samples in a neighborhood for a core point.
				metric_dbscan: The distance metric to use.
				data_for_clustering: The data to use for clustering.
			"""
			st.markdown("<h4>Run DBSCAN Clustering</h4>", unsafe_allow_html=True)

			if st.button("Run DBSCAN Clustering"):

				try:
					with st.spinner(f"Running DBSCAN clustering with ε={eps:.4f}, MinPts={min_samples}..."):

						# Run DBSCAN
						labels, dbscan_model = self.clustering_analyzer.run_dbscan_clustering( data=data_for_clustering, eps=eps, min_samples=min_samples, metric=metric_dbscan )

						# Store labels in session state
						st.session_state.dbscan_labels = pd.Series(labels, index=data_for_clustering.index, name="dbscan_cluster")

						# Count number of clusters and noise points
						unique_labels = set(labels)
						n_clusters_ = len(unique_labels) - (1 if -1 in labels else 0)
						n_noise_ = list(labels).count(-1)

						# Calculate metrics (only if more than one cluster found, excluding noise)
						metrics = {}
						if n_clusters_ > 1:
							# Filter out noise points for metric calculation
							non_noise_mask = st.session_state.dbscan_labels != -1
							if non_noise_mask.sum() > 1: # Need at least 2 points in clusters
								metrics = self.clustering_analyzer.evaluate_clustering(
									data_for_clustering[non_noise_mask],
									st.session_state.dbscan_labels[non_noise_mask],
									"dbscan" # Store metrics under 'dbscan' key
								)
								# Store metrics centrally
								st.session_state.cluster_metrics['dbscan'] = metrics
							else:
								st.warning("Not enough points in clusters (excluding noise) to calculate evaluation metrics.")
						else:
							st.warning("DBSCAN resulted in 0 or 1 cluster (excluding noise). Evaluation metrics require at least 2 clusters.")
							st.session_state.cluster_metrics.pop('dbscan', None) # Remove old metrics if they exist


						# Show success message
						st.success(f"""
						DBSCAN clustering complete!
						- Number of clusters found: {n_clusters_}
						- Number of noise points: {n_noise_} ({n_noise_/len(labels)*100:.2f}%)
						""")
						if metrics:
							metrics_text = "\n".join([f"- {k.replace('_', ' ').title()}: {v:.4f}" for k, v in metrics.items()])
							st.markdown(f"**Evaluation Metrics (excluding noise):**\n{metrics_text}")


						# --- Visualization ---
						# Show cluster distribution
						cluster_counts = st.session_state.dbscan_labels.value_counts().sort_index()
						# Map -1 to "Noise" for display
						display_labels = cluster_counts.index.map(lambda x: "Noise" if x == -1 else f"Cluster {x}")
						fig_dist = px.bar(
							x=display_labels, y=cluster_counts.values,
							labels={'x': 'Cluster / Noise', 'y': 'Number of Points'},
							title="Distribution of Points per Cluster (DBSCAN)"
						)
						st.plotly_chart(fig_dist, use_container_width=True)

						# Visualize clusters if data is 2D or 3D
						if data_for_clustering.shape[1] in [2, 3]:
							vis_data = data_for_clustering.copy()
							# Map labels to strings for coloring, handle noise
							vis_data['Cluster'] = st.session_state.dbscan_labels.apply(lambda x: 'Noise' if x == -1 else f'Cluster {x}').astype(str)

							if data_for_clustering.shape[1] == 2:
								fig_scatter = px.scatter(
									vis_data, x=vis_data.columns[0], y=vis_data.columns[1], color='Cluster',
									title="DBSCAN Clustering Results (2D)",
									color_discrete_map={"Noise": "grey"}, # Explicitly color noise
									category_orders={"Cluster": sorted(vis_data['Cluster'].unique(), key=lambda x: int(x.split()[-1]) if x != 'Noise' else -1)} # Sort clusters correctly
								)
								st.plotly_chart(fig_scatter, use_container_width=True)
							else: # 3D
								fig_scatter = px.scatter_3d(
									vis_data, x=vis_data.columns[0], y=vis_data.columns[1], z=vis_data.columns[2], color='Cluster',
									title="DBSCAN Clustering Results (3D)",
									color_discrete_map={"Noise": "grey"},
									category_orders={"Cluster": sorted(vis_data['Cluster'].unique(), key=lambda x: int(x.split()[-1]) if x != 'Noise' else -1)}
								)
								st.plotly_chart(fig_scatter, use_container_width=True)

				except AttributeError:
					st.error("Clustering Analyzer is not properly initialized or does not have the required methods.")
				except Exception as e:
					st.error(f"Error running DBSCAN clustering: {str(e)}")
					logging.exception("Error in Run DBSCAN")

		def save_dbscan_results():
			if not('dbscan_labels' in st.session_state and st.session_state.dbscan_labels is not None):
				st.warning("No DBSCAN results available. Please run DBSCAN clustering first.")
				return

			with st.expander("Save DBSCAN Results"):
				save_col1, save_col2 = st.columns(2)
				with save_col1:
					if st.button("Save DBSCAN Model"):
						try:
							base_path_save = "."
							if 'current_file_path' in st.session_state and st.session_state.current_file_path:
								potential_path_save = os.path.dirname(st.session_state.current_file_path)
								if os.path.isdir(potential_path_save):
									base_path_save = potential_path_save

							model_path = self.clustering_analyzer.save_model("dbscan", base_path_save)
							st.success(f"Saved DBSCAN model to {model_path}")
						except AttributeError:
							st.error("Clustering Analyzer does not have a 'save_model' method or model is not available.")
						except Exception as e:
							st.error(f"Error saving DBSCAN model: {str(e)}")
				with save_col2:
					if st.button("Save DBSCAN Cluster Assignments"):
						try:
							# Create DataFrame with original index and cluster assignments
							assignments_df = pd.DataFrame({'cluster': st.session_state.dbscan_labels}, index=data_for_clustering.index)

							base_path_save = "."
							if 'current_file_path' in st.session_state and st.session_state.current_file_path:
								potential_path_save = os.path.dirname(st.session_state.current_file_path)
								if os.path.isdir(potential_path_save):
									base_path_save = potential_path_save

							filepath = FeatureEngineerUtils.save_features(
								features    = assignments_df,
								feature_type= "dbscan_cluster_assignments",
								base_path   = base_path_save,
								format      = "csv"
							)

							st.success(f"Saved cluster assignments to {filepath}")
						except AttributeError:
							st.error("Feature Engineer does not have a 'save_features' method.")
						except Exception as e:
							st.error(f"Error saving DBSCAN assignments: {str(e)}")




		st.markdown("<h3>DBSCAN Clustering</h3>", unsafe_allow_html=True)

		# Check if input data is available
		if not ('clustering_input_data' in st.session_state and st.session_state.clustering_input_data is not None):
			st.warning("No input data available for clustering. Please prepare data in the 'Data Selection' tab first.")
			return

		st.markdown("""
		<div class='info-box'>
		DBSCAN (Density-Based Spatial Clustering of Applications with Noise) finds clusters of arbitrary shapes by grouping points that are closely packed together, marking outliers as noise (-1). It requires tuning `epsilon` (neighborhood distance) and `min_samples`.
		</div>
		""", unsafe_allow_html=True)

		# Determine which data to use
		data_for_clustering = st.session_state.clustering_input_data
		use_reduced_data    = False

		if 'reduced_data' in st.session_state and st.session_state.reduced_data is not None:
			use_reduced_data = st.checkbox( "Use dimensionality-reduced data for DBSCAN", value=True, help="Use the reduced data instead of the original preprocessed data" )

			if use_reduced_data:
				data_for_clustering = st.session_state.reduced_data
				st.info(f"Using reduced data with shape: {data_for_clustering.shape}")

			else:
				st.info(f"Using preprocessed data with shape: {data_for_clustering.shape}")


		# --- DBSCAN parameters ---
		eps, min_samples, metric_dbscan, k_dist,n_samples = dbscan_parameters()

		# --- Calculate k-distance plot ---
		calculate_k_distance_plot(k_dist=k_dist, n_samples=n_samples, data_for_clustering=data_for_clustering)


		# --- Run DBSCAN Section ---
		run_dbscan_clustering(eps=eps, min_samples=min_samples, metric_dbscan=metric_dbscan, data_for_clustering=data_for_clustering)

		# --- Save Results Section ---
		save_dbscan_results()

	def _lda_topic_modeling(self):
		st.markdown("<h3>LDA Topic Modeling</h3>", unsafe_allow_html=True)

		st.markdown("""
		<div class='info-box'>
		Latent Dirichlet Allocation (LDA) is a generative probabilistic model often used for topic modeling on text data. Here, it can identify underlying themes or patterns in order sequences or text columns.
		</div>
		""", unsafe_allow_html=True)

		# --- Data Selection for LDA ---
		lda_data_source = st.radio(
									label   = "Select Data Source for LDA",
									options = ["Order Sequences", "Text Column in Current Table"],
									index   = 0,
									help    = "Choose the input data for LDA."
								)

		documents = None
		doc_ids = None # To store corresponding patient IDs or indices

		if lda_data_source == "Order Sequences":
			if 'order_sequences' in st.session_state and st.session_state.order_sequences:

				# Convert sequences to space-separated strings
				documents = [" ".join(map(str, seq)) for seq in st.session_state.order_sequences.values()]
				doc_ids   = list(st.session_state.order_sequences.keys())

				st.info(f"Using {len(documents)} patient order sequences as documents.")
				st.text_area("Sample Document (Sequence):", documents[0] if documents else "", height=100)

			else:
				st.warning("Order sequences not found or empty. Please generate them in the 'Feature Engineering' tab first.")

		elif lda_data_source == "Text Column in Current Table":

			if st.session_state.df is not None:

				text_columns = st.session_state.df.select_dtypes(include=['object', 'string']).columns.tolist()

				if text_columns:

					text_col = st.selectbox(
						label   = "Select Text Column",
						options = text_columns,
						help    = "Choose the column containing text documents."
					)

					# Convert to list of strings, handle NaNs
					documents = st.session_state.df[text_col].fillna("").astype(str).tolist()
					doc_ids   = st.session_state.df.index # Use DataFrame index

					st.info(f"Using text from column '{text_col}' ({len(documents)} documents).")
					st.text_area("Sample Document (Text Column):", documents[0] if documents else "", height=100)

				else:
					st.warning("No text (object/string) columns found in the current table.")

			else:
				st.warning("No table loaded. Please load data first.")

		# --- LDA Parameters and Execution ---
		if documents:

			st.markdown("<h4>LDA Parameters</h4>", unsafe_allow_html=True)

			lda_col1, lda_col2 = st.columns(2)
			with lda_col1:
				n_topics = st.number_input("Number of Topics (k)", min_value=2, max_value=30, value=5, help="Number of topics to extract")

			with lda_col2:
				max_iter_lda = st.slider("Maximum Iterations (LDA)", min_value=10, max_value=100, value=20, step=5, help="Max iterations for LDA fitting") # Reduced default for faster online learning

			lda_col3, lda_col4 = st.columns(2)
			with lda_col3:
				vectorizer_type = st.selectbox("Vectorizer", ["CountVectorizer", "TfidfVectorizer"], index=0, help="Method to convert text to features")

			with lda_col4:
				max_features_lda = st.number_input("Max Features (Vocabulary Size)", min_value=100, max_value=10000, value=1000, step=100, help="Limit vocabulary size")

			# LDA implementation often uses batch learning by default in sklearn
			# learning_method = st.selectbox("Learning Method", ["batch", "online"], index=0, help="LDA parameter estimation method")

			if st.button("Run LDA Topic Modeling"):
				try:
					with st.spinner(f"Running LDA with {n_topics} topics..."):
						# Map vectorizer type
						vectorizer_map = {"CountVectorizer": "count", "TfidfVectorizer": "tfidf"}

						# Run LDA using the analyzer
						lda_model, doc_topic_matrix, topic_term_matrix = self.clustering_analyzer.run_lda_topic_modeling(
							documents       = documents,
							n_topics        = n_topics,
							vectorizer_type = vectorizer_map[vectorizer_type],
							max_features    = max_features_lda,
							max_iter        = max_iter_lda,
							# learning_method = learning_method # Sklearn LDA defaults usually fine
						)

						# Get feature names from the stored vectorizer
						feature_names = self.clustering_analyzer.models['lda']['vectorizer'].get_feature_names_out()

						# Store results in session state
						st.session_state.lda_results = {
							'doc_topic_matrix' : pd.DataFrame(doc_topic_matrix, index=doc_ids, columns=[f"Topic_{i}" for i in range(n_topics)]),
							'topic_term_matrix': pd.DataFrame(topic_term_matrix, index=[f"Topic_{i}" for i in range(n_topics)], columns=feature_names),
							'model'            : lda_model # Store the model itself if needed later
						}
						st.success(f"LDA topic modeling complete with {n_topics} topics!")

				except AttributeError:
					st.error("Clustering Analyzer is not properly initialized or does not have an 'run_lda_topic_modeling' method.")
				except Exception as e:
					st.error(f"Error running LDA topic modeling: {str(e)}")
					logging.exception("Error in Run LDA")


			# --- Display LDA Results ---
			if not ('lda_results' in st.session_state and st.session_state.lda_results):
				return

			st.markdown("<h4>LDA Results Visualization</h4>", unsafe_allow_html=True)
			try:
				lda_res = st.session_state.lda_results
				doc_topic_df = lda_res['doc_topic_matrix']
				topic_term_df = lda_res['topic_term_matrix']

				# Display top terms per topic
				st.markdown("<h5>Top Terms per Topic</h5>", unsafe_allow_html=True)
				top_terms = self.clustering_analyzer.get_top_terms_per_topic(topic_term_df, n_terms=10)
				st.dataframe(top_terms, use_container_width=True)

				# Display document-topic distribution heatmap (sample)
				st.markdown("<h5>Document-Topic Distribution (Sample Heatmap)</h5>", unsafe_allow_html=True)
				sample_size_lda  = min(30, doc_topic_df.shape[0])
				doc_topic_sample = doc_topic_df.iloc[:sample_size_lda]

				fig_heatmap = px.imshow(
						img     = doc_topic_sample,
						aspect  = "auto",
						labels  = dict(x="Topic", y="Document Index/ID", color="Probability"),
						title   = f"Document-Topic Probabilities (Sample of {sample_size_lda})",
						color_continuous_scale = "Viridis"
					)

				st.plotly_chart(fig_heatmap, use_container_width=True)

				# Topic distribution overview (dominant topic per document)
				st.markdown("<h5>Overall Topic Distribution (Dominant Topic per Document)</h5>", unsafe_allow_html=True)

				dominant_topics = doc_topic_df.idxmax(axis=1).value_counts().sort_index()

				# Create a proper DataFrame for the bar chart
				topic_dist_df = pd.DataFrame({
					'Topic'              : dominant_topics.index,
					'Number_of_Documents': dominant_topics.values
				})

				fig_dist = px.bar(
									data_frame = topic_dist_df,
									x          = 'Topic',
									y          = 'Number_of_Documents',
									labels     = {'Topic': 'Topic', 'Number_of_Documents': 'Number of Documents'},
									title      = "Number of Documents Primarily Assigned to Each Topic"
								)
				st.plotly_chart(fig_dist, use_container_width=True)

				# Optional: Topic Similarity (if useful)
				# st.markdown("<h5>Topic Similarity (Cosine Similarity of Topic-Term Vectors)</h5>", unsafe_allow_html=True)
				# topic_similarity = cosine_similarity(topic_term_df.values)
				# fig_sim = px.imshow(topic_similarity, x=topic_term_df.index, y=topic_term_df.index,
				#                     labels=dict(color="Cosine Similarity"), title="Topic Similarity Matrix",
				#                     color_continuous_scale="Blues")
				# st.plotly_chart(fig_sim, use_container_width=True)

			except AttributeError:
				st.error("Clustering Analyzer does not have a 'get_top_terms_per_topic' method.")

			except Exception as e:
				st.error(f"Error displaying LDA results: {e}")
				logging.exception("Error displaying LDA results")


			# --- Save LDA Results ---
			with st.expander("Save LDA Results"):
				save_col1, save_col2 = st.columns(2)
				with save_col1:
					# LDA model saving can be tricky (requires vectorizer too). Often results are saved.
					# Skipping direct model save button for simplicity.
					pass

				with save_col2:
					if st.button("Save LDA Document-Topic Distributions"):
						try:
							assignments_df = st.session_state.lda_results['doc_topic_matrix']
							base_path_save = "."
							if 'current_file_path' in st.session_state and st.session_state.current_file_path:

									potential_path_save = os.path.dirname(st.session_state.current_file_path)

									if os.path.isdir(potential_path_save):
										base_path_save = potential_path_save

							filepath = FeatureEngineerUtils.save_features(
												features     = assignments_df,
												feature_type = "lda_doc_topic_distributions",
												base_path    = base_path_save,
												format       = "csv" )

							st.success(f"Saved document-topic distributions to {filepath}")

						except AttributeError:
							st.error("Feature Engineer does not have a 'save_features' method.")
						except Exception as e:
							st.error(f"Error saving LDA distributions: {str(e)}")

	def _evaluation_metrics(self):

		st.markdown("<h3>Clustering Evaluation Metrics Comparison</h3>", unsafe_allow_html=True)

		st.markdown("""
		<div class='info-box'>
		Compare the performance of different clustering algorithms run in this session using standard internal evaluation metrics (which do not require ground truth labels).
		</div>
		""", unsafe_allow_html=True)

		# Check which results are available
		available_results = {}
		data_used_for_metrics = {} # Store the data used for each algorithm's metrics

		if 'kmeans_labels' in st.session_state and st.session_state.kmeans_labels is not None:
			available_results['K-means'] = st.session_state.kmeans_labels

			# Try to determine data used based on checkbox state during run - this is fragile
			# A better approach would be to store the data alongside the labels when run
			if st.session_state.get('kmeans_used_reduced', False) and 'reduced_data' in st.session_state:
				data_used_for_metrics['K-means'] = st.session_state.reduced_data

			elif 'clustering_input_data' in st.session_state:
				data_used_for_metrics['K-means'] = st.session_state.clustering_input_data

		if 'hierarchical_labels' in st.session_state and st.session_state.hierarchical_labels is not None:

			available_results['Hierarchical'] = st.session_state.hierarchical_labels

			# Hierarchical often uses sampled data - need the sample used
			# This assumes the sample is stored correctly, which might not be the case
			# If not stored, we cannot reliably recalculate metrics here
			if 'hierarchical_data_sample' in st.session_state: # Need to ensure this is saved during run
				data_used_for_metrics['Hierarchical'] = st.session_state.hierarchical_data_sample

			# Fallback - cannot reliably evaluate if sample not stored

		if 'dbscan_labels' in st.session_state and st.session_state.dbscan_labels is not None:

			available_results['DBSCAN'] = st.session_state.dbscan_labels

			if st.session_state.get('dbscan_used_reduced', False) and 'reduced_data' in st.session_state:
				data_used_for_metrics['DBSCAN'] = st.session_state.reduced_data

			elif 'clustering_input_data' in st.session_state:
				data_used_for_metrics['DBSCAN'] = st.session_state.clustering_input_data


		if not available_results:
			st.warning("No clustering results available in the current session. Run at least one clustering algorithm first.")
			return

		st.markdown("#### Summary of Metrics")
		metrics_summary = []

		for name, labels in available_results.items():
			# Retrieve stored metrics if available
			metrics    = st.session_state.cluster_metrics.get(name.lower(), {})
			n_clusters = len(set(labels)) - (1 if name == 'DBSCAN' and -1 in set(labels) else 0)
			n_noise    = list(labels).count(-1) if name == 'DBSCAN' else 0

			metrics_summary.append({
				'Algorithm'              : name,
				'Num Clusters'           : n_clusters,
				'Noise Points (%)'       : f"{n_noise / len(labels) * 100:.1f}%" if name == 'DBSCAN' else "N/A",
				'Silhouette Score'       : metrics.get('silhouette_score', None),
				'Davies-Bouldin Index'   : metrics.get('davies_bouldin_score', None),
				'Calinski-Harabasz Index': metrics.get('calinski_harabasz_score', None),
			})

		metrics_df = pd.DataFrame(metrics_summary)

		# Format numeric columns nicely
		float_cols = ['Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Index']
		format_dict = {col: "{:.4f}" for col in float_cols}
		st.dataframe(metrics_df.style.format(format_dict, na_rep="N/A"), use_container_width=True)

		st.markdown("""
		*   **Silhouette Score:** Higher is better (range -1 to 1). Measures how similar an object is to its own cluster compared to other clusters.
		*   **Davies-Bouldin Index:** Lower is better (min 0). Measures the average similarity ratio of each cluster with its most similar cluster.
		*   **Calinski-Harabasz Index:** Higher is better. Ratio of between-cluster dispersion to within-cluster dispersion.
		""")

		# --- Metrics Comparison Plot ---
		st.markdown("<h4>Metrics Comparison Visualization</h4>", unsafe_allow_html=True)
		metrics_to_plot = [col for col in float_cols if metrics_df[col].notna().any()] # Only plot metrics with values

		if metrics_to_plot:
			# Use melt for easier plotting with Plotly
			plot_df = metrics_df.melt(id_vars=['Algorithm'], value_vars=metrics_to_plot, var_name='Metric', value_name='Score')
			plot_df = plot_df.dropna() # Remove rows where metrics couldn't be calculated

			if not plot_df.empty:

				fig_comp = px.bar(plot_df, x='Metric', y='Score', color='Algorithm', barmode='group', title="Comparison of Clustering Evaluation Metrics")
				st.plotly_chart(fig_comp, use_container_width=True)

			else:
				st.info("No valid metrics available to plot.")

		else:
			st.info("No evaluation metrics were calculated or available for comparison.")


		# --- Cluster Agreement (if >1 result) ---
		if len(available_results) >= 2:
			st.markdown("<h4>Cluster Assignment Agreement (Adjusted Rand Index)</h4>", unsafe_allow_html=True)
			st.info("Compares the similarity of cluster assignments between pairs of algorithms (ignores noise points). Score close to 1 means high agreement, close to 0 means random agreement.")

			algo_names = list(available_results.keys())
			agreement_scores = pd.DataFrame(index=algo_names, columns=algo_names, dtype=float)

			for i in range(len(algo_names)):
				for j in range(i, len(algo_names)):
					algo1_name = algo_names[i]
					algo2_name = algo_names[j]

					if i == j:
						agreement_scores.loc[algo1_name, algo2_name] = 1.0
					else:
						labels1 = available_results[algo1_name]
						labels2 = available_results[algo2_name]

						# Ensure labels are aligned by index (important if sampling occurred)
						common_index = labels1.index.intersection(labels2.index)
						if len(common_index) < 2:
							ari = np.nan # Cannot compare if indices don't overlap sufficiently
						else:
							l1_common = labels1.loc[common_index]
							l2_common = labels2.loc[common_index]

							# Filter noise points (-1) for ARI calculation
							mask1 = l1_common != -1
							mask2 = l2_common != -1
							valid_mask = mask1 & mask2

							if valid_mask.sum() < 2:
								ari = np.nan # Not enough non-noise points to compare
							else:
								ari = adjusted_rand_score(l1_common[valid_mask], l2_common[valid_mask])

						agreement_scores.loc[algo1_name, algo2_name] = ari
						agreement_scores.loc[algo2_name, algo1_name] = ari # Symmetric matrix

			# Display heatmap
			fig_ari = px.imshow(agreement_scores,
								labels=dict(color="Adjusted Rand Index"),
								title="Pairwise Cluster Agreement (ARI)",
								color_continuous_scale='Blues', range_color=[0,1], # ARI typically 0-1, can be negative
								text_auto=".3f") # Show scores on heatmap
			st.plotly_chart(fig_ari, use_container_width=True)

		# --- Load/Compare Models (Placeholder/Optional) ---
		# This section is complex as it requires applying saved models to *current* data,
		# which might need the exact same preprocessing steps.
		# st.markdown("<h4>Load and Compare Saved Models</h4>", unsafe_allow_html=True)
		# st.info("Functionality to load previously saved models and compare them is under development.")

	def render(self):
		""" Renders the content of the Clustering Analysis tab. """

		st.markdown("<h2 class='sub-header'>Clustering Analysis</h2>", unsafe_allow_html=True)

		# Introductory text
		st.info("This section enables advanced clustering analysis on MIMIC-IV order data to discover patterns and patient groupings. You can apply different clustering algorithms and analyze the resulting clusters to gain insights.")

		# Clustering subtabs
		clustering_tabs = st.tabs([
			"📋 Data Selection",
			"📊 Dimensionality Reduction",
			"🔄 K-Means Clustering",
			"🌴 Hierarchical Clustering",
			"🔍 DBSCAN Clustering",
			"📝 LDA Topic Modeling",
			"📈 Evaluation Metrics"
		])

		with clustering_tabs[0]:
			self._data_selection()

		with clustering_tabs[1]:
			self._dimensionality_reduction()

		with clustering_tabs[2]:
			self._kmeans_clustering()

		with clustering_tabs[3]:
			self._hierarchical_clustering()

		with clustering_tabs[4]:
			self._dbscan_clustering()

		with clustering_tabs[5]:
			self._lda_topic_modeling()

		with clustering_tabs[6]:
			self._evaluation_metrics()
