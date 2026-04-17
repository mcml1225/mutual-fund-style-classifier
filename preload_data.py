"""
Preload all ticker data - Run this once before starting the app
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_acquisition import MutualFundDataCollector

def preload_all_data():
    """Fetch data for ALL tickers and save to CSV"""
    
    print("=" * 60)
    print("🔄 PRELOADING ALL TICKER DATA")
    print("=" * 60)
    
    collector = MutualFundDataCollector()
    
    # Show what we're going to fetch
    print("\n📊 Categories to fetch:")
    summary = collector.get_ticker_summary()
    for category, info in summary.items():
        print(f"  • {category}: {info['count']} tickers")
    
    total_tickers = len(collector.all_tickers)
    print(f"\n📈 Total tickers to fetch: {total_tickers}")
    
    # Fetch all data (no max_tickers limit)
    print("\n🌐 Fetching data from Yahoo Finance...")
    print("   This may take 1-2 minutes...")
    
    df = collector.collect_all_funds(period='3mo', max_tickers=None)
    
    if not df.empty:
        # Save to CSV
        collector.save_to_csv(df)
        
        print("\n" + "=" * 60)
        print("✅ DATA PRELOAD COMPLETE!")
        print("=" * 60)
        print(f"📊 Records saved: {len(df)}")
        print(f"📈 Unique tickers: {df['Ticker'].nunique()}")
        print(f"📋 Tickers: {list(df['Ticker'].unique())}")
        print("\n🚀 You can now run: streamlit run app/main.py")
    else:
        print("\n❌ Failed to fetch data. Check your internet connection.")

if __name__ == "__main__":
    preload_all_data()