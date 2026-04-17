"""
Data acquisition for mutual funds/ETFs
Includes comprehensive ticker list and data cleaning
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MutualFundDataCollector:
    """Collect historical data for mutual funds and ETFs"""
    
    def __init__(self):
        # Comprehensive tickers covering all Morningstar Style Box categories
        self.tickers = {
            'Large_Cap_Growth': ['SPYG', 'VUG', 'IWY', 'SPY', 'QQQ', 'IVW', 'SCHG'],
            'Large_Cap_Value': ['SPYV', 'VTV', 'IWD', 'IVE', 'SCHV', 'DGRO'],
            'Large_Cap_Blend': ['IVV', 'VOO', 'SPLG', 'SPY', 'VV'],
            'Mid_Cap_Growth': ['MDYG', 'VOT', 'IJK', 'MDY', 'SCHM', 'IWP'],
            'Mid_Cap_Value': ['MDYV', 'VOE', 'IJJ', 'SCHV', 'IWS'],
            'Mid_Cap_Blend': ['IVOO', 'VO', 'IJH', 'SCHM', 'WRD'],
            'Small_Cap_Growth': ['SLYG', 'VBK', 'IJT', 'IWM', 'SCHA', 'VIOG'],
            'Small_Cap_Value': ['SLYV', 'VBR', 'IJS', 'SCHA', 'VIOV'],
            'Small_Cap_Blend': ['IJR', 'VB', 'IWC', 'SCHA', 'VTWO']
        }
        
        # Flat list of all tickers for easy access
        self.all_tickers = []
        for category, ticker_list in self.tickers.items():
            self.all_tickers.extend(ticker_list)
        self.all_tickers = list(set(self.all_tickers))  # Remove duplicates
        
        logger.info(f"Initialized with {len(self.all_tickers)} unique tickers across 9 categories")
    
    def clean_dataframe(self, df):
        """Remove unnecessary index columns from dataframe"""
        if df is None or df.empty:
            return df
        
        # Remove unnamed index columns
        unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col) or col == 'index']
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
            logger.info(f"Removed columns: {unnamed_cols}")
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    def fetch_data(self, ticker, period='3mo'):
        """
        Fetch historical data for a single ticker
        
        Args:
            ticker (str): Stock symbol
            period (str): '1mo', '3mo', '6mo', '1y', '2y', '5y'
        
        Returns:
            pd.DataFrame: Historical OHLCV data with cleaned structure
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            
            if data is not None and not data.empty:
                data = data.reset_index()
                data['Ticker'] = ticker
                
                # Clean the data
                data = self.clean_dataframe(data)
                
                logger.info(f"Successfully fetched {len(data)} rows for {ticker}")
                return data
            else:
                logger.warning(f"No data returned for {ticker}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return None
    
    def get_category_for_ticker(self, ticker):
        """Get the category for a given ticker"""
        for category, ticker_list in self.tickers.items():
            if ticker in ticker_list:
                return category
        return 'Unknown'
    
    def collect_all_funds(self, period='3mo', max_tickers=None):
        """
        Collect data for all tickers
        
        Args:
            period (str): Time period for historical data
            max_tickers (int): Maximum number of tickers to fetch (None for all)
        
        Returns:
            pd.DataFrame: Combined data for all tickers
        """
        tickers_to_fetch = self.all_tickers
        if max_tickers:
            tickers_to_fetch = tickers_to_fetch[:max_tickers]
        
        all_data = []
        
        for i, ticker in enumerate(tickers_to_fetch):
            logger.info(f"Processing {ticker} ({i+1}/{len(tickers_to_fetch)})")
            
            df = self.fetch_data(ticker, period=period)
            
            if df is not None and not df.empty:
                # Add category information
                df['Category'] = self.get_category_for_ticker(ticker)
                df['True_Label'] = df['Category']
                all_data.append(df)
        
        if all_data:
            # Combine all data
            master_df = pd.concat(all_data, ignore_index=True)
            master_df = self.clean_dataframe(master_df)
            
            logger.info(f"Successfully collected {len(master_df)} records for {master_df['Ticker'].nunique()} tickers")
            return master_df
        else:
            logger.error("No data collected")
            return pd.DataFrame()
    
    def save_to_csv(self, df, filename='data/raw/mutual_funds_data.csv'):
        """Save collected data to CSV"""
        if df is None or df.empty:
            logger.error("No data to save")
            return False
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Clean before saving
        df = self.clean_dataframe(df)
        
        # Save without index
        df.to_csv(filename, index=False)
        logger.info(f"Saved {len(df)} records to {filename}")
        return True
    
    def load_from_csv(self, filename='data/raw/mutual_funds_data.csv'):
        """Load data from CSV file"""
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df = self.clean_dataframe(df)
            logger.info(f"Loaded {len(df)} records from {filename}")
            return df
        else:
            logger.warning(f"File {filename} not found")
            return pd.DataFrame()
    
    def get_ticker_summary(self):
        """Get summary of tickers by category"""
        summary = {}
        for category, tickers in self.tickers.items():
            summary[category] = {
                'count': len(tickers),
                'tickers': tickers
            }
        return summary


if __name__ == "__main__":
    # Test the data collector
    print("=" * 50)
    print("Testing MutualFundDataCollector")
    print("=" * 50)
    
    collector = MutualFundDataCollector()
    
    # Print summary
    print("\n📊 Ticker Summary:")
    summary = collector.get_ticker_summary()
    for category, info in summary.items():
        print(f"  {category}: {info['count']} tickers")
    
    print(f"\n📈 Total unique tickers: {len(collector.all_tickers)}")
    
    # Test fetching a single ticker
    print("\n🔍 Testing single ticker fetch...")
    test_data = collector.fetch_data('SPY', period='1mo')
    if test_data is not None:
        print(f"  ✅ Fetched {len(test_data)} rows for SPY")
        print(f"  📋 Columns: {list(test_data.columns)}")
    
    # Test collecting all funds (limited to 5 for quick test)
    print("\n🌐 Testing multi-ticker fetch (5 tickers)...")
    all_data = collector.collect_all_funds(period='1mo', max_tickers=5)
    if not all_data.empty:
        print(f"  ✅ Collected {len(all_data)} total records")
        print(f"  📊 Unique tickers: {all_data['Ticker'].nunique()}")
        print(f"  📋 Sample data:\n{all_data[['Ticker', 'Close', 'Volume']].head()}")
        
        # Save test data
        collector.save_to_csv(all_data, 'data/raw/test_data.csv')
        print("  💾 Saved test data to CSV")
    
    print("\n✅ Data acquisition module ready!")