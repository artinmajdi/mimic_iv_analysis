import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()

    def create_order_frequency_matrix(self, orders: pd.DataFrame) -> pd.DataFrame:
        """
        Create a matrix of order frequencies per patient
        """
        # Count unique order types per patient
        order_counts = orders.groupby(['subject_id', 'order_type']).size().unstack(fill_value=0)

        # Normalize the counts
        order_counts = pd.DataFrame(
            self.scaler.fit_transform(order_counts),
            columns=order_counts.columns,
            index=order_counts.index
        )

        return order_counts

    def create_temporal_features(self, orders: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from order timestamps
        """
        # Convert timestamps to datetime
        orders['ordertime'] = pd.to_datetime(orders['ordertime'])

        # Calculate time since admission
        admissions = orders.groupby('hadm_id')['ordertime'].min()
        orders['time_since_admission'] = orders.apply(
            lambda x: (x['ordertime'] - admissions[x['hadm_id']]).total_seconds() / 3600,
            axis=1
        )

        # Create temporal features
        temporal_features = orders.groupby('subject_id').agg({
            'time_since_admission': ['mean', 'std', 'min', 'max'],
            'order_type': 'nunique'
        })

        # Flatten column names
        temporal_features.columns = ['_'.join(col).strip() for col in temporal_features.columns.values]

        return temporal_features

    def create_sequence_features(self, orders: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Create order sequences for each patient
        """
        # Sort orders by time
        orders = orders.sort_values(['subject_id', 'ordertime'])

        # Group orders by patient and create sequences
        sequences = orders.groupby('subject_id')['order_type'].apply(list).to_dict()

        return sequences

    def create_order_type_distribution(self, orders: pd.DataFrame) -> pd.DataFrame:
        """
        Create distribution of order types per patient
        """
        # Calculate order type proportions
        order_dist = orders.groupby(['subject_id', 'order_type']).size().unstack(fill_value=0)
        order_dist = order_dist.div(order_dist.sum(axis=1), axis=0)

        return order_dist

    def combine_features(self,
                        frequency_matrix: pd.DataFrame,
                        temporal_features: pd.DataFrame,
                        order_dist: pd.DataFrame) -> pd.DataFrame:
        """
        Combine all features into a single feature matrix
        """
        # Join all feature matrices
        features = frequency_matrix.join(temporal_features, how='outer')
        features = features.join(order_dist, how='outer', rsuffix='_dist')

        # Fill NaN values with 0
        features = features.fillna(0)

        return features

    def prepare_features(self, orders: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """
        Prepare all features for clustering analysis
        """
        # Create individual feature matrices
        frequency_matrix = self.create_order_frequency_matrix(orders)
        temporal_features = self.create_temporal_features(orders)
        sequences = self.create_sequence_features(orders)
        order_dist = self.create_order_type_distribution(orders)

        # Combine features
        features = self.combine_features(
            frequency_matrix,
            temporal_features,
            order_dist
        )

        return features, sequences
