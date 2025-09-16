import datetime
import logging
import os
import pickle
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import umap
from scipy.cluster.hierarchy import linkage
import scipy.spatial.distance as ssd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import LatentDirichletAllocation, PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize

# Import Dask for large data handling
import dask.dataframe as dd

# Streamlit import
import streamlit as st

# import go
import plotly.graph_objects as go

RANDOM_STATE = 42

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ClusteringAnalyzer:
	"""Handles clustering analysis for MIMIC-IV data."""

	def __init__(self):
		"""Initialize the clustering analysis class."""
		self.random_state = 42
		self.models = {}
		self.preprocessed_data = {}
		self.cluster_results = {}
		self.cluster_metrics = {}
		self._persisted_resources = {}


	def preprocess_data(self, data: Union[pd.DataFrame, dd.DataFrame], method: str = 'standard', handle_missing: str = 'drop', use_dask: bool = False) -> pd.DataFrame:
		"""
		Preprocess data for clustering analysis.

		Args:
			data: Input DataFrame to preprocess (can be pandas DataFrame or Dask DataFrame)
			method: Preprocessing method ('standard', 'minmax', 'normalize')
			handle_missing: How to handle missing values ('drop', 'mean', 'median', 'mode')
			use_dask: If True, data is treated as a Dask DataFrame and computed when needed

		Returns:
			Preprocessed DataFrame (always returns a pandas DataFrame)
		"""
		logging.info(f"Preprocessing data with method={method}, handle_missing={handle_missing}, use_dask={use_dask}")

		# Convert Dask DataFrame to pandas if necessary
		if use_dask and hasattr(data, 'compute'):
			with st.spinner('Computing data for preprocessing from Dask DataFrame...'):
				# For clustering, we need the full DataFrame
				logging.info("Converting Dask DataFrame to pandas for clustering analysis")
				df = data.compute()
				logging.info(f"Dask DataFrame converted to pandas DataFrame of shape {df.shape}")
		else:
			# Work with a copy of the data
			df = data.copy()

		# Handle missing values
		logging.info(f"Handling missing values with method: {handle_missing}")
		if handle_missing == 'drop':
			df = df.dropna()
		elif handle_missing == 'mean':
			df = df.fillna(df.mean())
		elif handle_missing == 'median':
			df = df.fillna(df.median())
		elif handle_missing == 'mode':
			df = df.fillna(df.mode().iloc[0])
		else:
			raise ValueError(f"Invalid missing value handling method: {handle_missing}")

		# Apply preprocessing based on method
		logging.info(f"Applying scaling method: {method}")
		if method == 'standard':
			scaler = StandardScaler()
			df_scaled = pd.DataFrame( scaler.fit_transform(df), columns=df.columns, index=df.index )
		elif method == 'minmax':
			scaler = MinMaxScaler()
			df_scaled = pd.DataFrame( scaler.fit_transform(df), columns=df.columns, index=df.index )
		elif method == 'normalize':
			df_scaled = pd.DataFrame( normalize(df, axis=1), columns=df.columns, index=df.index )
		else:
			raise ValueError(f"Invalid preprocessing method: {method}")

		# Store preprocessed data
		self.preprocessed_data = {
			'original'      : data if not use_dask else "Dask DataFrame (original)", # Don't store the entire computed Dask DataFrame
			'preprocessed'  : df_scaled,
			'method'        : method,
			'handle_missing': handle_missing,
			'use_dask'      : use_dask
		}

		logging.info(f"Preprocessing complete. Output shape: {df_scaled.shape}")
		return df_scaled


	def apply_dimensionality_reduction(self, data: pd.DataFrame, method: str = 'pca', n_components: int = 2, **kwargs) -> pd.DataFrame:
		"""
		Apply dimensionality reduction to input data.

		Args:
			data: Input DataFrame to reduce
			method: Reduction method ('pca', 'tsne', 'umap', 'svd')
			n_components: Number of dimensions to reduce to
			**kwargs: Additional parameters for the dimensionality reduction method

		Returns:
			DataFrame with reduced dimensions
		"""
		# Apply dimensionality reduction
		if method == 'pca':
			reducer = PCA(n_components=n_components, random_state=self.random_state)
			reduced_data = reducer.fit_transform(data)
			explained_variance = reducer.explained_variance_ratio_.sum()
			logging.info(f"PCA explained variance: {explained_variance:.4f}")

		elif method == 'tsne':
			# Default parameters for t-SNE
			tsne_params = {
				'perplexity'   : 30.0,
				'learning_rate': 200.0,
				'n_iter'       : 1000,
				'random_state' : self.random_state
			}
			# Update with any provided parameters
			tsne_params.update(kwargs)

			reducer = TSNE(n_components=n_components, **tsne_params)
			reduced_data = reducer.fit_transform(data)

		elif method == 'umap':
			# Default parameters for UMAP
			umap_params = {
				'n_neighbors' : 15,
				'min_dist'    : 0.1,
				'metric'      : 'euclidean',
				'random_state': self.random_state
			}
			# Update with any provided parameters
			umap_params.update(kwargs)

			reducer = umap.UMAP(n_components=n_components, **umap_params)
			reduced_data = reducer.fit_transform(data)

		elif method == 'svd':
			reducer = TruncatedSVD(n_components=n_components, random_state=self.random_state)
			reduced_data = reducer.fit_transform(data)
			explained_variance = reducer.explained_variance_ratio_.sum()
			logging.info(f"SVD explained variance: {explained_variance:.4f}")

		else:
			raise ValueError(f"Invalid dimensionality reduction method: {method}")

		# Convert to DataFrame
		col_names = [f"{method}{i+1}" for i in range(n_components)]
		reduced_df = pd.DataFrame( reduced_data, columns=col_names, index=data.index )

		# Store the reducer model
		self.models[f'reducer_{method}'] = reducer

		return reduced_df


	def run_kmeans_clustering(self, data: pd.DataFrame, n_clusters: int = 5, **kwargs) -> Tuple[pd.Series, KMeans]:
		"""
		Run K-means clustering on data.

		Args:
			data: Input DataFrame to cluster
			n_clusters: Number of clusters to form
			**kwargs: Additional parameters for KMeans

		Returns:
			Tuple of (cluster labels, KMeans model)
		"""
		# Default parameters
		kmeans_params = {
			'n_init'      : 10,
			'max_iter'    : 300,
			'random_state': self.random_state
		}
		# Update with any provided parameters
		kmeans_params.update(kwargs)

		# Run K-means
		kmeans = KMeans(n_clusters=n_clusters, **kmeans_params)
		labels = kmeans.fit_predict(data)

		# Convert labels to Series
		labels_series = pd.Series(labels, index=data.index, name='cluster')

		# Store model and results
		self.models['kmeans'] = kmeans
		self.cluster_results['kmeans'] = labels_series

		return labels_series, kmeans


	def run_hierarchical_clustering(self, data: pd.DataFrame, n_clusters: int = 5, linkage_method: str = 'ward', distance_metric: str = 'euclidean', **kwargs) -> Tuple[pd.Series, Dict]:
		"""
		Run hierarchical clustering on data.

		Args:
			data: Input DataFrame to cluster
			n_clusters: Number of clusters to form
			linkage_method: Linkage method ('ward', 'complete', 'average', 'single')
			distance_metric: Distance metric (e.g., 'euclidean', 'manhattan')
			**kwargs: Additional parameters for AgglomerativeClustering

		Returns:
			Tuple of (cluster labels, linkage data)
		"""
		# Compute linkage matrix for dendrogram
		if distance_metric == 'euclidean' and linkage_method == 'ward':
			# Use scipy's linkage function directly
			linkage_matrix = linkage(data, method=linkage_method, metric=distance_metric)
		else:
			# Calculate distance matrix first
			if linkage_method == 'ward' and distance_metric != 'euclidean':
				logging.warning("Ward linkage requires Euclidean distance. Switching to Euclidean.")
				distance_metric = 'euclidean'

			distance_matrix = ssd.pdist(data, metric=distance_metric)
			linkage_matrix = linkage(distance_matrix, method=linkage_method)

		# Run hierarchical clustering
		hierarchical = AgglomerativeClustering( n_clusters=n_clusters, linkage=linkage_method, metric=distance_metric if linkage_method != 'ward' else 'euclidean', **kwargs )
		labels = hierarchical.fit_predict(data)

		# Convert labels to Series
		labels_series = pd.Series(labels, index=data.index, name='cluster')

		# Store model and results
		linkage_data = {
			'linkage_matrix' : linkage_matrix,
			'linkage_method' : linkage_method,
			'distance_metric': distance_metric
		}
		self.models['hierarchical']          = hierarchical
		self.models['hierarchical_linkage']  = linkage_data
		self.cluster_results['hierarchical'] = labels_series

		return labels_series, linkage_data


	def run_dbscan_clustering(self, data: pd.DataFrame, eps: float = 0.5, min_samples: int = 5, **kwargs) -> Tuple[pd.Series, DBSCAN]:
		"""
		Run DBSCAN clustering on data.

		Args:
			data: Input DataFrame to cluster
			eps: The maximum distance between two samples to be considered neighbors
			min_samples: The number of samples in a neighborhood for a point to be considered a core point
			**kwargs: Additional parameters for DBSCAN

		Returns:
			Tuple of (cluster labels, DBSCAN model)
		"""
		# Run DBSCAN
		dbscan = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
		labels = dbscan.fit_predict(data)

		# Convert labels to Series
		labels_series = pd.Series(labels, index=data.index, name='cluster')

		# Store model and results
		self.models['dbscan'] = dbscan
		self.cluster_results['dbscan'] = labels_series

		return labels_series, dbscan


	def run_lda_topic_modeling(self, documents: List[str], n_topics: int = 5, vectorizer_type: str = 'count', max_features: int = 1000, **kwargs) -> Tuple[LatentDirichletAllocation, pd.DataFrame, pd.DataFrame]:
		"""
		Run LDA topic modeling on text data.

		Args:
			documents: List of document texts
			n_topics: Number of topics to extract
			vectorizer_type: Type of vectorizer ('count' or 'tfidf')
			max_features: Maximum number of features for vectorization
			**kwargs: Additional parameters for LDA

		Returns:
			Tuple of (LDA model, document-topic matrix, topic-term matrix)
		"""
		# Vectorize documents
		if vectorizer_type == 'count':
			vectorizer = CountVectorizer(max_features=max_features)
		elif vectorizer_type == 'tfidf':
			vectorizer = TfidfVectorizer(max_features=max_features)
		else:
			raise ValueError(f"Invalid vectorizer type: {vectorizer_type}")

		# Create document-term matrix
		dtm = vectorizer.fit_transform(documents)

		# Get feature names
		feature_names = vectorizer.get_feature_names_out()

		# Set default LDA parameters
		lda_params = {
			'n_components': n_topics,
			'random_state': self.random_state,
			'max_iter': 10,
			'learning_method': 'online'
		}
		# Update with provided parameters
		lda_params.update(kwargs)

		# Run LDA
		lda_model = LatentDirichletAllocation(**lda_params)
		document_topics = lda_model.fit_transform(dtm)

		# Create document-topic matrix
		doc_topic_cols = [f"Topic{i+1}" for i in range(n_topics)]
		doc_topic_matrix = pd.DataFrame(document_topics, columns=doc_topic_cols)

		# Create topic-term matrix
		topic_term_matrix = pd.DataFrame(
			lda_model.components_,
			columns=feature_names
		)

		# Store model and results
		self.models['lda'] = {
			'model': lda_model,
			'vectorizer': vectorizer
		}
		self.cluster_results['lda'] = doc_topic_matrix

		return lda_model, doc_topic_matrix, topic_term_matrix


	def get_top_terms_per_topic(self, topic_term_matrix: pd.DataFrame, n_terms: int = 10) -> pd.DataFrame:
		"""
		Extract top terms for each topic from LDA results.

		Args:
			topic_term_matrix: Topic-term matrix from LDA
			n_terms: Number of top terms to extract per topic

		Returns:
			DataFrame with top terms per topic
		"""
		top_terms = {}

		for topic_idx, topic in enumerate(topic_term_matrix.values):
			# Get indices of top terms
			top_term_indices = topic.argsort()[-n_terms:][::-1]
			# Get term names
			terms = [topic_term_matrix.columns[i] for i in top_term_indices]
			# Get term weights
			weights = [topic[i] for i in top_term_indices]

			# Store in dictionary
			top_terms[f"Topic{topic_idx+1}"] = {
				'terms': terms,
				'weights': weights
			}

		# Convert to DataFrame for easier visualization
		# Create column names that match the number of terms
		column_names = [f"Term_{i+1}" for i in range(n_terms)]
		result = pd.DataFrame(columns=column_names)

		for topic, data in top_terms.items():
			# Ensure we have exactly n_terms by padding with empty strings if needed
			terms_padded = data['terms'] + [''] * (n_terms - len(data['terms']))
			result.loc[topic] = terms_padded[:n_terms]

		return result


	def evaluate_clustering(self, data: pd.DataFrame, labels: pd.Series, method: str) -> Dict[str, float]:
		"""
		Evaluate clustering results using various metrics.

		Args:
			data: Data used for clustering
			labels: Cluster labels
			method: Clustering method name

		Returns:
			Dictionary of metric names and values
		"""
		# Skip evaluation if all samples are assigned to the same cluster
		if len(np.unique(labels)) <= 1:
			return {
				'silhouette_score': np.nan,
				'davies_bouldin_score': np.nan,
				'calinski_harabasz_score': np.nan
			}

		# Initialize metrics dictionary
		metrics = {}

		# Calculate silhouette score
		try:
			metrics['silhouette_score'] = silhouette_score(data, labels)
		except:
			metrics['silhouette_score'] = np.nan

		# Calculate Davies-Bouldin index
		try:
			metrics['davies_bouldin_score'] = davies_bouldin_score(data, labels)
		except:
			metrics['davies_bouldin_score'] = np.nan

		# Calculate Calinski-Harabasz index
		try:
			metrics['calinski_harabasz_score'] = calinski_harabasz_score(data, labels)
		except:
			metrics['calinski_harabasz_score'] = np.nan

		# Store metrics
		self.cluster_metrics[method] = metrics

		return metrics


	def find_optimal_k_for_kmeans(self, data: pd.DataFrame, k_range: range = range(2, 11), metric: str = 'silhouette', **kwargs) -> Tuple[int, Dict[str, List[float]]]:
		"""
		Find the optimal number of clusters for K-means.

		Args:
			data: Input data
			k_range: Range of k values to try
			metric: Metric to optimize ('silhouette', 'davies_bouldin', 'calinski_harabasz', 'inertia')
			**kwargs: Additional parameters for KMeans

		Returns:
			Tuple of (optimal k, metrics for all k)
		"""
		# Track persisted intermediates for this method
		persisted_intermediates = {}
		
		try:
			# Persist data if it's a Dask DataFrame for repeated access
			if hasattr(data, 'persist'):
				data_persisted = data.persist()
				persisted_intermediates['data'] = data_persisted
				logging.info("Persisted input data for optimal k search")
			else:
				data_persisted = data
			
			# Initialize metrics
			results = {
				'k': list(k_range),
				'silhouette': [],
				'davies_bouldin': [],
				'calinski_harabasz': [],
				'inertia': []
			}

			# Compute metrics for each k
			for k in k_range:
				# Run K-means
				kmeans = KMeans(n_clusters=k, random_state=self.random_state, **kwargs)
				labels = kmeans.fit_predict(data_persisted)

				# Store inertia
				results['inertia'].append(kmeans.inertia_)

				# Calculate other metrics if more than one cluster
				if k > 1:
					results['silhouette'].append(silhouette_score(data_persisted, labels))
					results['davies_bouldin'].append(davies_bouldin_score(data_persisted, labels))
					results['calinski_harabasz'].append(calinski_harabasz_score(data_persisted, labels))
				else:
					results['silhouette'].append(np.nan)
					results['davies_bouldin'].append(np.nan)
					results['calinski_harabasz'].append(np.nan)

			# Find optimal k based on selected metric
			if metric == 'silhouette':
				# Higher is better
				optimal_idx = np.nanargmax(results['silhouette'])
			elif metric == 'davies_bouldin':
				# Lower is better
				optimal_idx = np.nanargmin(results['davies_bouldin'])
			elif metric == 'calinski_harabasz':
				# Higher is better
				optimal_idx = np.nanargmax(results['calinski_harabasz'])
			elif metric == 'inertia':
				# Use elbow method for inertia
				# Calculate the rate of change of inertia
				inertia = np.array(results['inertia'])
				rate_of_change = np.diff(inertia) / inertia[:-1]

				# Find the point where rate of change starts to diminish
				# (add 1 because diff reduces length by 1)
				optimal_idx = np.argmax(rate_of_change) + 1

				# Ensure optimal_idx is within bounds
				if optimal_idx >= len(k_range):
					optimal_idx = len(k_range) - 1
			else:
				raise ValueError(f"Invalid metric: {metric}")

			# Get optimal k
			optimal_k = k_range[optimal_idx]
			
			# Update persisted resources tracking
			self._persisted_resources.update(persisted_intermediates)

			return optimal_k, results
			
		except Exception as e:
			logging.error(f"Error in find_optimal_k_for_kmeans: {e}")
			# Clean up persisted intermediates on error
			for key, persisted_obj in persisted_intermediates.items():
				try:
					if hasattr(persisted_obj, '__dask_graph__'):
						del persisted_obj
				except Exception as cleanup_error:
					logging.warning(f"Error cleaning up persisted resource {key}: {cleanup_error}")
			raise


	def find_optimal_eps_for_dbscan(self, data: pd.DataFrame, k_dist: int = 5, n_samples: int = 1000) -> float:
		"""
		Find the optimal epsilon value for DBSCAN using k-distance graph.

		Args:
			data: Input data
			k_dist: k value for k-distance
			n_samples: Number of samples to use for estimation

		Returns:
			Suggested epsilon value
		"""
		# Sample data if it's too large
		if len(data) > n_samples:
			data_sample = data.sample(n_samples, random_state=self.random_state)
		else:
			data_sample = data

		# Calculate distances
		from sklearn.neighbors import NearestNeighbors
		neighbors = NearestNeighbors(n_neighbors=k_dist).fit(data_sample)
		distances, _ = neighbors.kneighbors(data_sample)

		# Sort distances to the kth nearest neighbor
		k_distances = np.sort(distances[:, k_dist-1])

		# Calculate "slope"
		slopes = np.diff(k_distances)

		# Find the point of maximum slope
		max_slope_idx = np.argmax(slopes) + 1

		# Get the suggested epsilon value
		suggested_eps = k_distances[max_slope_idx]

		return suggested_eps, k_distances


	def save_model(self, model_name: str, path: str) -> str:
		"""
		Save a trained model to disk.

		Args:
			model_name: Name of the model to save
			path: Directory path to save to

		Returns:
			Path to saved model file
		"""
		if model_name not in self.models:
			raise ValueError(f"Model {model_name} not found")

		# Create directory if it doesn't exist
		models_dir = os.path.join(path, 'models')
		os.makedirs(models_dir, exist_ok=True)

		# Create timestamp for filename
		timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

		# Save model
		model_path = os.path.join(models_dir, f"{model_name}_{timestamp}.pkl")
		with open(model_path, 'wb') as f:
			pickle.dump(self.models[model_name], f)

		return model_path


	def load_model(self, model_path: str, model_name: str) -> Any:
		"""
		Load a trained model from disk.

		Args:
			model_path: Path to the saved model file
			model_name: Name to assign to the loaded model

		Returns:
			The loaded model
		"""
		# Load model
		with open(model_path, 'rb') as f:
			model = pickle.load(f)

		# Store model
		self.models[model_name] = model

		return model


	def get_cluster_summary(self, data: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
		"""
		Generate summary statistics for each cluster.

		Args:
			data: Original data
			labels: Cluster labels

		Returns:
			DataFrame with cluster statistics
		"""
		# Combine data with cluster labels
		data_with_clusters = data.copy()
		data_with_clusters['cluster'] = labels

		# Initialize summary DataFrame
		summary = pd.DataFrame()

		# Calculate statistics for each cluster
		for cluster_id in sorted(labels.unique()):
			# Get data for this cluster
			cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster_id]

			# Calculate basic statistics
			stats = {
				'count': len(cluster_data),
				'percentage': len(cluster_data) / len(data) * 100
			}

			# Add statistics for each feature
			for col in data.columns:
				stats[f"{col}_mean"] = cluster_data[col].mean()
				stats[f"{col}_std"] = cluster_data[col].std()
				stats[f"{col}_min"] = cluster_data[col].min()
				stats[f"{col}_max"] = cluster_data[col].max()

			# Add to summary
			summary[f"Cluster {cluster_id}"] = pd.Series(stats)

		return summary.T

	def find_optimal_k_kmeans_elbow_silhouette(self, data: pd.DataFrame, k_range=range(2, 11), n_init=10, max_iter=300) -> pd.DataFrame:
		"""
		Find optimal k for KMeans using both elbow method and silhouette score.

		Args:
			data: Input data for clustering
			k_range: Range of k values to try
			n_init: Number of initializations for KMeans
			max_iter: Maximum iterations for KMeans

		Returns:
			DataFrame of metrics with inertia and silhouette scores
		"""
		# Track persisted intermediates for this method
		persisted_intermediates = {}
		
		try:
			# Persist data if it's a Dask DataFrame for repeated access
			if hasattr(data, 'persist'):
				data_persisted = data.persist()
				persisted_intermediates['data'] = data_persisted
				logging.info("Persisted input data for optimal k search")
			else:
				data_persisted = data
			
			metrics = {'k': list(k_range), 'inertia': [], 'silhouette': []}

			for k in k_range:
				kmeans = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=self.random_state)
				labels = kmeans.fit_predict(data_persisted)

				# Record inertia (within-cluster sum of squares)
				metrics['inertia'].append(kmeans.inertia_)

				# Calculate silhouette score for k > 1
				if k > 1:
					metrics['silhouette'].append(silhouette_score(data_persisted, labels))
				else:
					metrics['silhouette'].append(0)

			# Find optimal k using silhouette score (higher is better)
			optimal_k_silhouette = k_range[np.argmax(metrics['silhouette'])]

			# Update persisted resources tracking
			self._persisted_resources.update(persisted_intermediates)
			
			# Return optimal k based on silhouette score and all metrics
			return pd.DataFrame(metrics), optimal_k_silhouette
			
		except Exception as e:
			logging.error(f"Error in find_optimal_k_kmeans_elbow_silhouette: {e}")
			# Clean up persisted intermediates on error
			for key, persisted_obj in persisted_intermediates.items():
				try:
					if hasattr(persisted_obj, '__dask_graph__'):
						del persisted_obj
				except Exception as cleanup_error:
					logging.warning(f"Error cleaning up persisted resource {key}: {cleanup_error}")
			raise


class ClusterInterpreter:
	"""Handles advanced analytics and visualization for cluster analysis."""

	def __init__(self):
		self.random_state = 42
		self.analysis_results = {}

	def calculate_length_of_stay(self, df: pd.DataFrame, admission_col: str, discharge_col: str, patient_id_col: str = None) -> pd.Series:
		"""Calculate length of stay in days for each patient."""

		if admission_col not in df.columns or discharge_col not in df.columns:
			raise ValueError(f"Columns {admission_col} or {discharge_col} not found in DataFrame")

		# Ensure datetime format
		if df[admission_col].dtype != 'datetime64[ns]':
			df[admission_col] = pd.to_datetime(df[admission_col])
		if df[discharge_col].dtype != 'datetime64[ns]':
			df[discharge_col] = pd.to_datetime(df[discharge_col])

		# Calculate length of stay in days
		los = (df[discharge_col] - df[admission_col]).dt.total_seconds() / (24 * 60 * 60)

		# Create a Series with patient ID as index if provided
		if patient_id_col and patient_id_col in df.columns:
			los = pd.Series(los.values, index=df[patient_id_col], name='length_of_stay')
		else:
			los = pd.Series(los.values, name='length_of_stay')

		return los

	def compare_los_across_clusters(self, los_data: pd.Series, cluster_labels: pd.Series) -> pd.DataFrame:
		"""Compare length of stay statistics across clusters."""
		# Combine LOS and cluster labels
		combined = pd.DataFrame({
			'length_of_stay': los_data,
			'cluster': cluster_labels
		})

		# Calculate stats by cluster
		stats = combined.groupby('cluster')['length_of_stay'].agg([
			'count', 'mean', 'std', 'min', 'max',
			lambda x: x.quantile(0.25).round(2),
			lambda x: x.quantile(0.5).round(2),
			lambda x: x.quantile(0.75).round(2)
		]).reset_index()

		# Rename columns
		stats.columns = ['Cluster', 'Count', 'Mean LOS', 'Std Dev', 'Min LOS', 'Max LOS', '25th Perc', 'Median', '75th Perc']

		# Store results
		self.analysis_results['los_comparison'] = stats

		return stats

	def statistical_testing(self,
						  data: pd.DataFrame,
						  feature_cols: List[str],
						  cluster_col: str = 'cluster',
						  method: str = 'anova') -> pd.DataFrame:
		"""Perform statistical tests to compare features between clusters."""
		# Initialize results
		results = []

		# Perform statistical tests for each feature
		for feature in feature_cols:
			if method == 'anova':
				# Group data by cluster
				groups = [data[data[cluster_col] == cluster][feature].dropna().values
						 for cluster in data[cluster_col].unique() if cluster != -1]  # Exclude noise points

				# Only perform test if we have at least 2 clusters with data
				if len(groups) >= 2 and all(len(g) > 0 for g in groups):
					try:
						# Perform one-way ANOVA
						from scipy import stats
						f_stat, p_value = stats.f_oneway(*groups)
						significant = p_value < 0.05

						results.append({
							'Feature': feature,
							'Test': 'ANOVA',
							'Statistic': f_stat,
							'P-Value': p_value,
							'Significant': significant
						})
					except Exception as e:
						results.append({
							'Feature': feature,
							'Test': 'ANOVA',
							'Statistic': None,
							'P-Value': None,
							'Significant': False,
							'Error': str(e)
						})
			elif method == 'kruskal':
				# Non-parametric Kruskal-Wallis H-test
				groups = [data[data[cluster_col] == cluster][feature].dropna().values
							for cluster in data[cluster_col].unique() if cluster != -1]

				if len(groups) >= 2 and all(len(g) > 0 for g in groups):
					try:
						from scipy import stats
						h_stat, p_value = stats.kruskal(*groups)
						significant = p_value < 0.05

						results.append({
							'Feature'    : feature,
							'Test'       : 'Kruskal-Wallis',
							'Statistic'  : h_stat,
							'P-Value'    : p_value,
							'Significant': significant
						})
					except Exception as e:
						results.append({
							'Feature'    : feature,
							'Test'       : 'Kruskal-Wallis',
							'Statistic'  : None,
							'P-Value'    : None,
							'Significant': False,
							'Error'      : str(e)
						})

		# Create DataFrame of results
		results_df = pd.DataFrame(results)

		# Apply multiple testing correction if we have results
		if len(results_df) > 0 and 'P-Value' in results_df.columns:
			# Bonferroni correction
			results_df['Adjusted P-Value'] = results_df['P-Value'] * len(results_df)
			results_df['Adjusted P-Value'] = results_df['Adjusted P-Value'].clip(upper=1.0)
			results_df['Significant (Adjusted)'] = results_df['Adjusted P-Value'] < 0.05

		# Store results
		self.analysis_results['statistical_tests'] = results_df

		return results_df

	def calculate_feature_importance(self, data: pd.DataFrame, cluster_col: str = 'cluster') -> pd.DataFrame:
		"""Calculate feature importance for distinguishing between clusters."""
		from sklearn.ensemble import RandomForestClassifier
		from sklearn.preprocessing import StandardScaler
		import numpy as np

		# Extract features and target
		X = data.drop(columns=[cluster_col])
		y = data[cluster_col]

		# Standardize features
		scaler = StandardScaler()
		X_scaled = scaler.fit_transform(X)

		# Train a Random Forest classifier
		rf = RandomForestClassifier(n_estimators=100, random_state=42)
		rf.fit(X_scaled, y)

		# Get feature importance
		importance = rf.feature_importances_

		# Create DataFrame with feature importance
		importance_df = pd.DataFrame({
			'Feature': X.columns,
			'Importance': importance
		}).sort_values('Importance', ascending=False)

		# Store results
		self.analysis_results['feature_importance'] = importance_df

		return importance_df

	def generate_cluster_characterization(self,
										data: pd.DataFrame,
										cluster_col: str = 'cluster',
										patient_id_col: str = None,
										important_features: List[str] = None) -> Dict:
		"""Generate comprehensive characterization for each cluster."""
		# Check if 'cluster' column exists
		if cluster_col not in data.columns:
			raise ValueError(f"Column '{cluster_col}' not found in DataFrame")

		# Get unique clusters
		clusters = data[cluster_col].unique()

		# If no important features specified, use all features except cluster
		if important_features is None:
			important_features = [col for col in data.columns if col != cluster_col]
			if patient_id_col in important_features:
				important_features.remove(patient_id_col)

		# Initialize results
		characterization = {}

		# For each cluster
		for cluster in clusters:
			# Skip noise points (if DBSCAN was used)
			if cluster == -1:
				continue

			# Get data for this cluster
			cluster_data = data[data[cluster_col] == cluster]

			# Calculate basic statistics
			stats = {
				'size': len(cluster_data),
				'percentage': (len(cluster_data) / len(data)) * 100,
				'features': {}
			}

			# Calculate statistics for each feature
			for feature in important_features:
				if feature in cluster_data.columns:
					# Skip non-numeric features
					if not np.issubdtype(cluster_data[feature].dtype, np.number):
						continue

					feature_stats = {
						'mean': cluster_data[feature].mean(),
						'std': cluster_data[feature].std(),
						'min': cluster_data[feature].min(),
						'max': cluster_data[feature].max(),
						'median': cluster_data[feature].median(),
						'25th': cluster_data[feature].quantile(0.25),
						'75th': cluster_data[feature].quantile(0.75)
					}

					# Compare to overall dataset
					if np.issubdtype(data[feature].dtype, np.number):
						overall_mean = data[feature].mean()
						if overall_mean != 0:
							feature_stats['diff_from_mean'] = (feature_stats['mean'] - overall_mean) / overall_mean * 100
						else:
							feature_stats['diff_from_mean'] = 0

					stats['features'][feature] = feature_stats

			# Store cluster stats
			characterization[f"Cluster {cluster}"] = stats

		# Store results
		self.analysis_results['cluster_characterization'] = characterization

		return characterization

	def generate_html_report(self,
						   title: str = "Cluster Analysis Report",
						   include_plots: bool = True) -> str:
		"""Generate an HTML report from analysis results."""
		# Check if we have analysis results
		if not self.analysis_results:
			raise ValueError("No analysis results available. Run analysis methods first.")

		# Initialize HTML content
		html = f"""
		<!DOCTYPE html>
		<html>
		<head>
			<title>{title}</title>
			<style>
				body {{ font-family: Arial, sans-serif; margin: 20px; }}
				h1 {{ color: #2c3e50; }}
				h2 {{ color: #3498db; margin-top: 30px; }}
				table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
				th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
				th {{ background-color: #f2f2f2; }}
				tr:nth-child(even) {{ background-color: #f9f9f9; }}
				.significant {{ background-color: #d4efdf; }}
				.not-significant {{ background-color: #fadbd8; }}
				.plot-container {{ margin: 20px 0; }}
			</style>
			<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
		</head>
		<body>
			<h1>{title}</h1>
			<p>Report generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
		"""

		# Add Length of Stay comparison if available
		if 'los_comparison' in self.analysis_results:
			los_df = self.analysis_results['los_comparison']
			html += """
			<h2>Length of Stay Analysis</h2>
			<table>
				<tr>
			"""
			# Add column headers
			for col in los_df.columns:
				html += f"<th>{col}</th>"
			html += "</tr>"

			# Add rows
			for _, row in los_df.iterrows():
				html += "<tr>"
				for col in los_df.columns:
					html += f"<td>{row[col]}</td>"
				html += "</tr>"
			html += "</table>"

			# Add LOS plot if requested
			if include_plots:
				html += """
				<div id="los-plot" class="plot-container" style="height: 400px;"></div>
				<script>
					var losData = [
				"""
				# Create data for each cluster
				for cluster in los_df['Cluster'].unique():
					cluster_data = los_df[los_df['Cluster'] == cluster]
					html += f"""
					{{
						type: 'box',
						name: 'Cluster {cluster}',
						y: [{cluster_data['Mean LOS'].values[0]}],
						boxpoints: false,
						jitter: 0.3,
						pointpos: -1.8,
						boxmean: true
					}},
					"""

				html += """
				];

				var losLayout = {
					title: 'Length of Stay by Cluster',
					yaxis: {title: 'Length of Stay (days)'}
				};

				Plotly.newPlot('los-plot', losData, losLayout);
				</script>
				"""

		# Add Statistical Tests if available
		if 'statistical_tests' in self.analysis_results:
			test_df = self.analysis_results['statistical_tests']
			html += """
			<h2>Statistical Tests</h2>
			<table>
				<tr>
			"""
			# Add column headers
			for col in test_df.columns:
				html += f"<th>{col}</th>"
			html += "</tr>"

			# Add rows
			for _, row in test_df.iterrows():
				# Highlight significant results
				sig_class = "significant" if row.get('Significant (Adjusted)', False) else "not-significant"
				html += f'<tr class="{sig_class}">'
				for col in test_df.columns:
					value = row[col]
					# Format p-values
					if 'P-Value' in col and value is not None:
						value = f"{value:.4f}"
					html += f"<td>{value}</td>"
				html += "</tr>"
			html += "</table>"

		# Add Feature Importance if available
		if 'feature_importance' in self.analysis_results:
			imp_df = self.analysis_results['feature_importance']
			html += """
			<h2>Feature Importance</h2>
			<table>
				<tr>
					<th>Feature</th>
					<th>Importance</th>
				</tr>
			"""

			# Add rows (show top 10 features)
			for _, row in imp_df.head(10).iterrows():
				html += f"""
				<tr>
					<td>{row['Feature']}</td>
					<td>{row['Importance']:.4f}</td>
				</tr>
				"""
			html += "</table>"

			# Add feature importance plot if requested
			if include_plots:
				html += """
				<div id="importance-plot" class="plot-container" style="height: 500px;"></div>
				<script>
					var impData = [{
						type: 'bar',
						x: [
				"""

				# Add feature names
				for _, row in imp_df.head(10).iterrows():
					html += f"'{row['Feature']}', "

				html += """
						],
						y: [
				"""

				# Add importance values
				for _, row in imp_df.head(10).iterrows():
					html += f"{row['Importance']:.4f}, "

				html += """
						],
						marker: {
							color: 'rgba(55, 128, 191, 0.7)',
							line: {
								color: 'rgba(55, 128, 191, 1.0)',
								width: 2
							}
						}
					}];

					var impLayout = {
						title: 'Feature Importance',
						xaxis: {title: 'Feature'},
						yaxis: {title: 'Importance'}
					};

					Plotly.newPlot('importance-plot', impData, impLayout);
				</script>
				"""

		# Add Cluster Characterization if available
		if 'cluster_characterization' in self.analysis_results:
			characterization = self.analysis_results['cluster_characterization']
			html += """
			<h2>Cluster Characterization</h2>
			"""

			# For each cluster
			for cluster, stats in characterization.items():
				html += f"""
				<h3>{cluster}</h3>
				<p><strong>Size:</strong> {stats['size']} patients ({stats['percentage']:.2f}% of dataset)</p>

				<h4>Key Features</h4>
				<table>
					<tr>
						<th>Feature</th>
						<th>Mean</th>
						<th>Median</th>
						<th>Std Dev</th>
						<th>Diff from Overall Mean (%)</th>
					</tr>
				"""

				# Add rows for each feature (show top 10 by diff from mean)
				features_sorted = sorted(
					stats['features'].items(),
					key=lambda x: abs(x[1].get('diff_from_mean', 0)),
					reverse=True
				)

				for feature, feature_stats in features_sorted[:10]:
					html += f"""
					<tr>
						<td>{feature}</td>
						<td>{feature_stats['mean']:.2f}</td>
						<td>{feature_stats['median']:.2f}</td>
						<td>{feature_stats['std']:.2f}</td>
						<td>{feature_stats.get('diff_from_mean', 0):.2f}%</td>
					</tr>
					"""

				html += "</table>"

		# Close HTML document
		html += """
		</body>
		</html>
		"""

		return html

	def visualize_order_patterns(self, order_sequences: Dict[Any, List[str]], cluster_labels: pd.Series, max_orders: int = 20, max_patients: int = 50) -> go.Figure:
		"""
		Create interactive visualizations of order patterns by cluster.

		Args:
			order_sequences: Dictionary mapping patient IDs to lists of orders
			cluster_labels: Series with patient IDs as index and cluster labels as values
			max_orders: Maximum number of order types to include (most frequent)
			max_patients: Maximum number of patients to visualize per cluster

		Returns:
			Plotly figure with order pattern visualization
		"""
		# Match patient IDs between sequences and cluster labels
		common_patients = set(order_sequences.keys()) & set(cluster_labels.index)
		if not common_patients:
			raise ValueError("No matching patient IDs between order sequences and cluster labels")

		# Get unique clusters
		clusters = cluster_labels.loc[common_patients].unique()

		# Count order frequencies
		all_orders = []
		for patient_id in common_patients:
			all_orders.extend(order_sequences[patient_id])

		# Get most frequent orders
		order_counts = pd.Series(all_orders).value_counts()
		top_orders = order_counts.head(max_orders).index.tolist()

		# Create subplots - one for each cluster
		n_clusters = len(clusters)
		fig = make_subplots(
			rows=n_clusters,
			cols=1,
			subplot_titles=[f"Cluster {c}" for c in clusters],
			vertical_spacing=0.05
		)

		# For each cluster
		for i, cluster in enumerate(clusters):
			# Get patients in this cluster
			cluster_patients = cluster_labels[cluster_labels == cluster].index
			cluster_patients = list(set(cluster_patients) & common_patients)

			# Sample patients if too many
			if len(cluster_patients) > max_patients:
				cluster_patients = random.sample(cluster_patients, max_patients)

			# Create heatmap data
			heatmap_data = []
			y_labels = []

			for j, patient_id in enumerate(cluster_patients):
				# Get sequence for this patient
				sequence = order_sequences[patient_id]

				# Filter to top orders
				filtered_sequence = [order for order in sequence if order in top_orders]

				# Create patient row
				patient_row = [0] * len(top_orders)

				# Count occurrences of each order
				for order in filtered_sequence:
					idx = top_orders.index(order)
					patient_row[idx] += 1

				# Add to heatmap data
				heatmap_data.append(patient_row)
				y_labels.append(f"Patient {patient_id}")

			# Create heatmap if we have data
			if heatmap_data:
				fig.add_trace(
					go.Heatmap(
						z=heatmap_data,
						x=top_orders,
						y=y_labels,
						colorscale='Viridis',
						name=f"Cluster {cluster}"
					),
					row=i+1,
					col=1
				)

		# Update layout
		fig.update_layout(
			height=300 * n_clusters,
			title="Order Patterns by Cluster",
			xaxis_title="Order Type",
			yaxis_title="Patient"
		)

		# Adjust x-axis for readability
		for i in range(1, n_clusters + 1):
			fig.update_xaxes(tickangle=45, row=i, col=1)

		# Store results
		self.analysis_results['order_patterns'] = {
			'figure': fig,
			'top_orders': top_orders
		}

		return fig
