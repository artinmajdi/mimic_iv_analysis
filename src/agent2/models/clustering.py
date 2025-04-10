import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from typing import Tuple, Dict, List
import umap
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

class ClusteringAnalyzer:
    def __init__(self):
        self.models = {}
        self.reducer = None
        self.autoencoder = None

    def perform_lda(self, features: pd.DataFrame, n_components: int = 5, n_iter: int = 500) -> Tuple[np.ndarray, LatentDirichletAllocation]:
        """
        Perform Latent Dirichlet Allocation on order patterns
        """
        lda = LatentDirichletAllocation(
            n_components=n_components,
            n_iter=n_iter,
            random_state=42
        )
        lda_features = lda.fit_transform(features)
        self.models['lda'] = lda
        return lda_features, lda

    def build_autoencoder(self, input_dim: int, encoding_dim: int) -> Model:
        """
        Build an autoencoder model
        """
        # Encoder
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim * 2, activation='relu')(input_layer)
        encoded = Dense(encoding_dim, activation='relu')(encoded)

        # Decoder
        decoded = Dense(encoding_dim * 2, activation='relu')(encoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)

        # Autoencoder
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

        return autoencoder

    def perform_autoencoder_kmeans(self, features: pd.DataFrame, encoding_dim: int = 10, epochs: int = 50) -> Tuple[np.ndarray, Model, KMeans]:
        """
        Perform dimensionality reduction using autoencoder followed by K-means clustering
        """
        # Build and train autoencoder
        autoencoder = self.build_autoencoder(features.shape[1], encoding_dim)
        autoencoder.fit(
            features,
            features,
            epochs=epochs,
            batch_size=32,
            verbose=0
        )

        # Get encoded representation
        encoder = Model(autoencoder.input, autoencoder.layers[-3].output)
        encoded_features = encoder.predict(features)

        # Perform K-means on encoded features
        kmeans = KMeans(n_clusters=encoding_dim, random_state=42)
        clusters = kmeans.fit_predict(encoded_features)

        self.autoencoder = autoencoder
        self.models['kmeans'] = kmeans

        return clusters, autoencoder, kmeans

    def perform_tsne(self, features: pd.DataFrame, n_components: int = 2) -> np.ndarray:
        """
        Perform t-SNE dimensionality reduction
        """
        tsne = TSNE(
            n_components=n_components,
            random_state=42
        )
        tsne_features = tsne.fit_transform(features)
        self.reducer = tsne
        return tsne_features

    def perform_umap(self, features: pd.DataFrame, n_components: int = 2) -> np.ndarray:
        """
        Perform UMAP dimensionality reduction
        """
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=42
        )
        umap_features = reducer.fit_transform(features)
        self.reducer = reducer
        return umap_features

    def perform_kmeans(self, features: np.ndarray, n_clusters: int = 5) -> Tuple[np.ndarray, KMeans]:
        """
        Perform K-means clustering
        """
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42
        )
        clusters = kmeans.fit_predict(features)
        self.models['kmeans'] = kmeans
        return clusters, kmeans

    def perform_dbscan(self, features: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> Tuple[np.ndarray, DBSCAN]:
        """
        Perform DBSCAN clustering
        """
        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples
        )
        clusters = dbscan.fit_predict(features)
        self.models['dbscan'] = dbscan
        return clusters, dbscan

    def perform_hierarchical(self, features: np.ndarray, n_clusters: int = 5) -> Tuple[np.ndarray, AgglomerativeClustering]:
        """
        Perform hierarchical clustering
        """
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters
        )
        clusters = hierarchical.fit_predict(features)
        self.models['hierarchical'] = hierarchical
        return clusters, hierarchical

    def evaluate_clusters(self, features: np.ndarray, clusters: np.ndarray) -> Dict[str, float]:
        """
        Evaluate clustering results using various metrics
        """
        # Remove noise points (cluster label -1) for evaluation
        mask = clusters != -1
        if np.sum(mask) < 2:
            return {
                'silhouette_score': np.nan,
                'calinski_harabasz_score': np.nan
            }

        return {
            'silhouette_score': silhouette_score(features[mask], clusters[mask]),
            'calinski_harabasz_score': calinski_harabasz_score(features[mask], clusters[mask])
        }

    def analyze_clusters(self,
                        features: pd.DataFrame,
                        method: str = 'kmeans',
                        n_clusters: int = 5,
                        use_dim_reduction: bool = True,
                        **kwargs) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Perform complete clustering analysis
        """
        # Apply dimensionality reduction if requested and method doesn't have built-in reduction
        if use_dim_reduction and method not in ['lda', 'autoencoder_kmeans']:
            reduced_features = self.perform_umap(features)
        else:
            reduced_features = features.values

        # Perform clustering
        if method == 'kmeans':
            clusters, _ = self.perform_kmeans(reduced_features, n_clusters)
        elif method == 'dbscan':
            clusters, _ = self.perform_dbscan(reduced_features, **kwargs)
        elif method == 'hierarchical':
            clusters, _ = self.perform_hierarchical(reduced_features, n_clusters)
        elif method == 'lda':
            lda_features, _ = self.perform_lda(features, n_components=n_clusters, **kwargs)
            clusters = np.argmax(lda_features, axis=1)
        elif method == 'autoencoder_kmeans':
            clusters, _, _ = self.perform_autoencoder_kmeans(features, encoding_dim=n_clusters, **kwargs)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Evaluate clusters
        metrics = self.evaluate_clusters(reduced_features, clusters)

        return clusters, metrics
