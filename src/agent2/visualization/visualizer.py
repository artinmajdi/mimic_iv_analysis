import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List
import seaborn as sns
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set1

    def plot_cluster_scatter(self,
                           features: pd.DataFrame,
                           clusters: np.ndarray,
                           title: str = "Cluster Visualization") -> go.Figure:
        """
        Create interactive scatter plot of clusters
        """
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'x': features.iloc[:, 0],
            'y': features.iloc[:, 1],
            'cluster': clusters
        })

        # Create scatter plot
        fig = px.scatter(
            plot_df,
            x='x',
            y='y',
            color='cluster',
            title=title,
            color_discrete_sequence=self.color_palette
        )

        return fig

    def plot_order_heatmap(self,
                          order_matrix: pd.DataFrame,
                          clusters: np.ndarray) -> go.Figure:
        """
        Create heatmap of order frequencies by cluster
        """
        # Calculate mean order frequencies by cluster
        cluster_means = order_matrix.groupby(clusters).mean()

        # Create heatmap
        fig = px.imshow(
            cluster_means,
            title="Order Frequency by Cluster",
            color_continuous_scale='Viridis'
        )

        return fig

    def plot_length_of_stay(self,
                           admissions: pd.DataFrame,
                           clusters: np.ndarray) -> go.Figure:
        """
        Create box plot of length of stay by cluster
        """
        # Calculate length of stay
        admissions['length_of_stay'] = (
            pd.to_datetime(admissions['dischtime']) -
            pd.to_datetime(admissions['admittime'])
        ).dt.total_seconds() / (24 * 3600)

        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'cluster': clusters,
            'length_of_stay': admissions['length_of_stay']
        })

        # Create box plot
        fig = px.box(
            plot_df,
            x='cluster',
            y='length_of_stay',
            title="Length of Stay by Cluster",
            color='cluster',
            color_discrete_sequence=self.color_palette
        )

        return fig

    def plot_order_sequence(self,
                          sequences: Dict[str, List[str]],
                          clusters: np.ndarray,
                          n_samples: int = 5) -> go.Figure:
        """
        Create visualization of order sequences
        """
        # Sample sequences from each cluster
        sampled_sequences = []
        for cluster in np.unique(clusters):
            cluster_indices = np.where(clusters == cluster)[0]
            if len(cluster_indices) > 0:
                sampled_indices = np.random.choice(
                    cluster_indices,
                    min(n_samples, len(cluster_indices)),
                    replace=False
                )
                for idx in sampled_indices:
                    sampled_sequences.append({
                        'cluster': cluster,
                        'sequence': ' -> '.join(sequences[idx])
                    })

        # Create DataFrame for plotting
        plot_df = pd.DataFrame(sampled_sequences)

        # Create sequence visualization
        fig = px.scatter(
            plot_df,
            x='cluster',
            y='sequence',
            color='cluster',
            title="Order Sequences by Cluster",
            color_discrete_sequence=self.color_palette
        )

        return fig

    def plot_cluster_metrics(self, metrics: Dict[str, float]) -> go.Figure:
        """
        Create bar plot of clustering metrics
        """
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'metric': list(metrics.keys()),
            'value': list(metrics.values())
        })

        # Create bar plot
        fig = px.bar(
            plot_df,
            x='metric',
            y='value',
            title="Clustering Metrics",
            color='metric',
            color_discrete_sequence=self.color_palette
        )

        return fig

    def create_dashboard(self,
                        features: pd.DataFrame,
                        clusters: np.ndarray,
                        order_matrix: pd.DataFrame,
                        admissions: pd.DataFrame,
                        sequences: Dict[str, List[str]],
                        metrics: Dict[str, float]) -> List[go.Figure]:
        """
        Create complete dashboard of visualizations
        """
        figures = [
            self.plot_cluster_scatter(features, clusters),
            self.plot_order_heatmap(order_matrix, clusters),
            self.plot_length_of_stay(admissions, clusters),
            self.plot_order_sequence(sequences, clusters),
            self.plot_cluster_metrics(metrics)
        ]

        return figures
