"""
K-Means clustering for mutual fund style classification
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

class StyleBoxClusterer:
    """Unsupervised clustering to identify investment styles"""
    
    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None
        self.features_with_clusters = None
        
    def train(self, X_scaled, features_df):
        """Train K-Means model and assign cluster labels"""
        self.model = KMeans(n_clusters=self.n_clusters, 
                           random_state=self.random_state, 
                           n_init=10)
        
        clusters = self.model.fit_predict(X_scaled)
        
        # Add cluster labels to features dataframe
        self.features_with_clusters = features_df.copy()
        self.features_with_clusters['Cluster'] = clusters
        
        # Simple style mapping
        self._simple_style_mapping()
        
        return self.features_with_clusters
    
    def _simple_style_mapping(self):
        """Simple mapping based on volatility and momentum rankings"""
        if self.features_with_clusters is None:
            return
        
        # Get cluster averages
        cluster_vol = self.features_with_clusters.groupby('Cluster')['volatility'].mean()
        cluster_mom = self.features_with_clusters.groupby('Cluster')['momentum'].mean()
        
        # Rank clusters
        vol_ranks = cluster_vol.rank().astype(int)
        mom_ranks = cluster_mom.rank().astype(int)
        
        # Define mappings
        cap_map = {1: 'Large Cap', 2: 'Mid Cap', 3: 'Small Cap', 4: 'Micro Cap'}
        style_map = {1: 'Value', 2: 'Blend', 3: 'Growth', 4: 'Momentum'}
        
        # Apply mapping
        style_labels = []
        for cluster in self.features_with_clusters['Cluster']:
            vol_rank = vol_ranks[cluster] if cluster in vol_ranks.index else 2
            mom_rank = mom_ranks[cluster] if cluster in mom_ranks.index else 2
            
            cap = cap_map.get(vol_rank, 'Mid Cap')
            style = style_map.get(mom_rank, 'Blend')
            style_labels.append(f"{cap} {style}")
        
        self.features_with_clusters['Style_Box'] = style_labels
    
    def save_model(self, path='models/kmeans_model.pkl'):
        """Save trained model"""
        if self.model is not None:
            joblib.dump(self.model, path)
            print(f"Model saved to {path}")
    
    def load_model(self, path='models/kmeans_model.pkl'):
        """Load trained model"""
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")