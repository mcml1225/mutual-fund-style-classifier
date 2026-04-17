"""
Feature engineering for mutual fund classification
Fixed version - handles timezone issues
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
        self.data = data.copy()
        self.features_df = None
        
        # Handle Date column properly with timezone fix
        if 'Date' in self.data.columns:
            # Convert to datetime with UTC to avoid timezone issues
            self.data['Date'] = pd.to_datetime(self.data['Date'], utc=True, errors='coerce')
            
            # Drop rows with invalid dates
            self.data = self.data.dropna(subset=['Date'])
            
            # Set as index for easier manipulation
            self.data.set_index('Date', inplace=True)
        
        # Ensure we have valid data
        if self.data.empty:
            raise ValueError("No valid data after processing dates")
        
    def calculate_returns(self):
        """Calculate daily returns and momentum"""
        # Reset index temporarily for groupby if needed
        if self.data.index.name == 'Date':
            self.data = self.data.reset_index()
        
        # Sort by Ticker and Date safely
        self.data = self.data.sort_values(['Ticker', 'Date'])
        
        # Calculate daily returns
        self.data['daily_return'] = self.data.groupby('Ticker')['Close'].pct_change()
        self.data['daily_return'] = self.data['daily_return'].fillna(0)
        
        # Calculate 5-day momentum (short term)
        self.data['momentum_5d'] = self.data.groupby('Ticker')['Close'].pct_change(5)
        self.data['momentum_5d'] = self.data['momentum_5d'].fillna(0)
        
        # Calculate 20-day momentum (medium term)
        self.data['momentum_20d'] = self.data.groupby('Ticker')['Close'].pct_change(20)
        self.data['momentum_20d'] = self.data['momentum_20d'].fillna(0)
        
        return self
    
    def calculate_volatility(self):
        """Calculate historical volatility (annualized)"""
        # Calculate rolling volatility (20-day window)
        self.data['volatility_daily'] = self.data.groupby('Ticker')['daily_return'].transform(
            lambda x: x.rolling(window=20, min_periods=1).std()
        )
        
        # Annualize volatility (sqrt(252 trading days))
        self.data['volatility'] = self.data['volatility_daily'] * np.sqrt(252)
        self.data['volatility'] = self.data['volatility'].fillna(0)
        
        # Calculate liquidity (log volume)
        self.data['liquidity'] = np.log1p(self.data['Volume'])
        self.data['liquidity'] = self.data['liquidity'].fillna(0)
        
        return self
    
    def calculate_sharpe_ratio(self, risk_free_rate=0.02):
        """Calculate Sharpe ratio"""
        # Excess returns
        self.data['excess_return'] = self.data['daily_return'] - (risk_free_rate / 252)
        
        # Rolling Sharpe ratio (20-day window)
        self.data['sharpe_ratio'] = self.data.groupby('Ticker')['excess_return'].transform(
            lambda x: x.rolling(window=20, min_periods=1).mean() / 
                     (x.rolling(window=20, min_periods=1).std() + 1e-6)
        )
        self.data['sharpe_ratio'] = self.data['sharpe_ratio'].fillna(0)
        
        return self
    
    def create_features_matrix(self) -> pd.DataFrame:
        """
        Create final feature matrix for clustering (one row per ticker)
        """
        # Get the most recent values for each ticker
        latest_data = self.data.groupby('Ticker').last().reset_index()
        
        # Create features dataframe
        features = pd.DataFrame()
        features['Ticker'] = latest_data['Ticker']
        
        # Extract features
        features['volatility'] = latest_data['volatility'].fillna(0)
        features['momentum'] = latest_data['momentum_20d'].fillna(0)
        features['momentum_5d'] = latest_data['momentum_5d'].fillna(0)
        features['sharpe_ratio'] = latest_data['sharpe_ratio'].fillna(0)
        features['liquidity'] = latest_data['liquidity'].fillna(0)
        
        # Calculate average daily return
        avg_returns = self.data.groupby('Ticker')['daily_return'].mean()
        features['avg_return'] = features['Ticker'].map(avg_returns).fillna(0)
        
        # Add category information if available
        if 'Category' in latest_data.columns:
            features['Category'] = latest_data['Category']
        else:
            features['Category'] = 'Unknown'
            
        if 'True_Label' in latest_data.columns:
            features['True_Label'] = latest_data['True_Label']
        else:
            # Infer label from ticker
            def infer_label(ticker):
                ticker_upper = ticker.upper()
                large_cap = ['SPY', 'QQQ', 'IVV', 'VOO', 'VUG', 'IWY', 'SPYG', 'VTV', 'IWD']
                mid_cap = ['MDY', 'IJH', 'VO', 'VOT', 'MDYG', 'VOE', 'IJJ']
                small_cap = ['IWM', 'IJR', 'VB', 'VBK', 'SLYG', 'VBR', 'IJS']
                
                if ticker_upper in large_cap:
                    return 'Large_Cap'
                elif ticker_upper in mid_cap:
                    return 'Mid_Cap'
                elif ticker_upper in small_cap:
                    return 'Small_Cap'
                else:
                    return 'Unknown'
            
            features['True_Label'] = features['Ticker'].apply(infer_label)
        
        # Ensure all numeric columns are float
        numeric_cols = ['volatility', 'momentum', 'momentum_5d', 'sharpe_ratio', 'liquidity', 'avg_return']
        for col in numeric_cols:
            features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)
        
        self.features_df = features
        return features
    
    def scale_features(self) -> Tuple[np.ndarray, StandardScaler]:
        """Standardize features for clustering"""
        feature_columns = ['volatility', 'momentum', 'sharpe_ratio', 'liquidity']
        
        # Select only numeric columns that exist
        available_features = [col for col in feature_columns if col in self.features_df.columns]
        
        if not available_features:
            raise ValueError("No feature columns available for scaling")
        
        X = self.features_df[available_features].copy()
        X = X.astype(float)
        X = X.fillna(0)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, scaler


if __name__ == "__main__":
    print("Feature Engineering module ready")
    print("Fixed timezone handling")