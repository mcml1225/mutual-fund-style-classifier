"""
Feature engineering for mutual fund classification
Creates metrics that capture growth/value characteristics and market cap
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple

class FeatureEngineer:
    """Extract financial features for clustering"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Args:
            data: DataFrame with columns ['Close', 'Volume', 'Ticker']
        """
        self.data = data.copy()  # Create a copy to avoid modifying original
        self.features_df = None
        
        # Ensure date is datetime index
        if 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            self.data.set_index('Date', inplace=True)
        
    def calculate_returns(self):
        """Calculate daily, monthly, and 6-month returns"""
        # Sort by date for each ticker
        self.data = self.data.sort_values(['Ticker', self.data.index.name or 'Date'])
        
        # Daily returns
        self.data['daily_return'] = self.data.groupby('Ticker')['Close'].pct_change()
        
        # 6-month cumulative return (Momentum feature) - ~126 trading days
        self.data['momentum_6m'] = self.data.groupby('Ticker')['Close'].transform(
            lambda x: x.pct_change(periods=min(126, len(x)-1)) if len(x) > 1 else 0
        )
        
        return self
    
    def calculate_volatility(self):
        """Calculate historical volatility (annualized)"""
        # Daily volatility for each ticker
        volatility_stats = self.data.groupby('Ticker')['daily_return'].agg([
            ('volatility_daily', lambda x: x.std() if len(x) > 1 else 0),
            ('avg_volume', lambda x: np.log(x.mean() + 1) if hasattr(x, 'mean') else 0)
        ])
        
        # Annualize volatility (sqrt(252 trading days))
        volatility_stats['volatility_annual'] = volatility_stats['volatility_daily'] * np.sqrt(252)
        
        # Merge back - use first() since these are constant per ticker
        for ticker in volatility_stats.index:
            mask = self.data['Ticker'] == ticker
            self.data.loc[mask, 'volatility_annual'] = volatility_stats.loc[ticker, 'volatility_annual']
            self.data.loc[mask, 'avg_volume'] = volatility_stats.loc[ticker, 'avg_volume']
        
        return self
    
    def calculate_sharpe_ratio(self, risk_free_rate=0.02):
        """Calculate Sharpe ratio as growth/value proxy"""
        # Risk-adjusted returns
        excess_returns = self.data.groupby('Ticker')['daily_return'].mean() - (risk_free_rate/252)
        daily_std = self.data.groupby('Ticker')['daily_return'].std()
        
        # Avoid division by zero
        sharpe_ratios = excess_returns / daily_std
        sharpe_ratios = sharpe_ratios.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Map back to data
        self.data['sharpe_ratio'] = self.data['Ticker'].map(sharpe_ratios)
        
        return self
    
    def create_features_matrix(self) -> pd.DataFrame:
        """
        Create final feature matrix for clustering
        
        Returns:
            DataFrame with features per ticker
        """
        # Get last values for each ticker (most recent data)
        latest_data = self.data.groupby('Ticker').last().reset_index()
        
        # Create features dataframe
        features = pd.DataFrame()
        features['Ticker'] = latest_data['Ticker']
        
        # Add features (handle missing values)
        features['volatility'] = latest_data.get('volatility_annual', 0).fillna(0)
        features['momentum'] = latest_data.get('momentum_6m', 0).fillna(0)
        features['sharpe_ratio'] = latest_data.get('sharpe_ratio', 0).fillna(0)
        features['liquidity'] = latest_data.get('avg_volume', 0).fillna(0)
        features['avg_return'] = self.data.groupby('Ticker')['daily_return'].mean().values
        features['avg_return'] = features['avg_return'].fillna(0)
        
        # Add category if exists (for evaluation only)
        if 'Category' in latest_data.columns:
            features['Category'] = latest_data['Category']
        else:
            features['Category'] = 'Unknown'
            
        if 'True_Label' in latest_data.columns:
            features['True_Label'] = latest_data['True_Label']
        else:
            # Create label from ticker if possible
            features['True_Label'] = features['Ticker'].apply(
                lambda x: 'Large_Cap' if x in ['SPY', 'QQQ', 'IVV'] 
                else 'Mid_Cap' if x in ['MDY', 'IJH'] 
                else 'Small_Cap' if x in ['IWM', 'IJR'] 
                else 'Unknown'
            )
        
        # Replace any infinite values
        features = features.replace([np.inf, -np.inf], 0)
        
        # Handle any remaining NaN
        features = features.fillna(0)
        
        self.features_df = features
        return features
    
    def scale_features(self) -> Tuple[np.ndarray, StandardScaler]:
        """Standardize features for clustering"""
        feature_columns = ['volatility', 'momentum', 'sharpe_ratio', 'liquidity']
        
        # Ensure all features exist
        available_features = [col for col in feature_columns if col in self.features_df.columns]
        
        if not available_features:
            raise ValueError("No feature columns available for scaling")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.features_df[available_features])
        
        return X_scaled, scaler


if __name__ == "__main__":
    # Test the feature engineering
    from data_acquisition import MutualFundDataCollector
    
    print("Testing Feature Engineer...")
    collector = MutualFundDataCollector()
    raw_data = collector.fetch_data('SPY', period='1mo')
    
    if raw_data is not None:
        raw_data['Ticker'] = 'SPY'
        engineer = FeatureEngineer(raw_data)
        engineer.calculate_returns().calculate_volatility().calculate_sharpe_ratio()
        features = engineer.create_features_matrix()
        
        print(f"Features shape: {features.shape}")
        print(features[['Ticker', 'volatility', 'momentum', 'sharpe_ratio']].head())
        print("✅ Feature engineering test passed!")
    else:
        print("❌ Failed to fetch test data")