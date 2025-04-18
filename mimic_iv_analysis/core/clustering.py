import datetime
import logging
import os
import pickle
from typing import Any, Dict, List, Tuple

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


RANDOM_STATE = 42

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MIMICClusteringAnalysis:
	"""Handles clustering analysis for MIMIC-IV data."""

	def __init__(self):
		"""Initialize the clustering analysis class."""
		self.random_state = 42
		self.models = {}
		self.preprocessed_data = {}
		self.cluster_results = {}
		self.cluster_metrics = {}

	def preprocess_data(self, data: pd.DataFrame, method: str = 'standard', handle_missing: str = 'drop') -> pd.DataFrame:
		"""
		Preprocess data for clustering analysis.

		Args:
			data: Input DataFrame to preprocess
			method: Preprocessing method ('standard', 'minmax', 'normalize')
			handle_missing: How to handle missing values ('drop', 'mean', 'median', 'mode')

		Returns:
			Preprocessed DataFrame
		"""
		# Work with a copy of the data
		df = data.copy()

		# Handle missing values
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
			'original'      : data,
			'preprocessed'  : df_scaled,
			'method'        : method,
			'handle_missing': handle_missing
		}

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
		result = pd.DataFrame(columns=range(1, n_terms+1))

		for topic, data in top_terms.items():
			result.loc[topic] = data['terms']

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
			labels = kmeans.fit_predict(data)

			# Store inertia
			results['inertia'].append(kmeans.inertia_)

			# Calculate other metrics if more than one cluster
			if k > 1:
				results['silhouette'].append(silhouette_score(data, labels))
				results['davies_bouldin'].append(davies_bouldin_score(data, labels))
				results['calinski_harabasz'].append(calinski_harabasz_score(data, labels))
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

		return optimal_k, results

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


