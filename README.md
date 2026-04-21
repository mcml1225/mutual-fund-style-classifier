# Mutual Fund Style Classifier

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-red.svg)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org)
[![Yahoo Finance](https://img.shields.io/badge/Yahoo%20Finance-API-purple.svg)](https://finance.yahoo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 1. PROJECT OVERVIEW

An unsupervised machine learning system that automatically classifies mutual funds and ETFs into the Morningstar Style Box categories, identifying whether they are Growth, Value, Small Cap, or Large Cap using only historical price data.

**Key Features:**

- Automatic Clustering: K-Means algorithm classifies funds without pre-labeled data
- Style Box Visualization: Interactive 3x3 heatmap of the Morningstar Style Box
- Risk-Return Analysis: Visualize volatility versus performance relationships
- Sharpe Ratio: Measure risk-adjusted returns for each fund
- Live Data: Real-time data integration with Yahoo Finance API

---

## 2. PROJECT STRUCTURE
mutual-fund-style-classifier/
│
├── app/
│ └── main.py # Streamlit web application
│
├── src/
│ ├── data_acquisition.py # Yahoo Finance data downloader
│ ├── feature_engineering.py # Feature extraction (volatility, momentum, Sharpe)
│ └── clustering_model.py # K-Means clustering model
│
├── data/
│ └── raw/ # CSV data storage (git-ignored)
│
├── models/ # Trained models (git-ignored)
│
├── preload_data_simple.py # One-time data download script
├── requirements.txt # Python dependencies
├── .gitignore # Git ignore rules
└── README.md # This file

text

---

## 3. INSTALLATION

**Prerequisites:**
- Python 3.8 or higher
- Git
- Internet connection (for fetching market data)

**Step-by-step installation:**


### 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/mutual-fund-style-classifier.git
cd mutual-fund-style-classifier

### 2. Create virtual environment
python -m venv venv

### 3. Activate virtual environment
### On Windows:
venv\Scripts\activate
### On Mac/Linux:
source venv/bin/activate

### 4. Install dependencies
pip install -r requirements.txt

### 5. Preload market data (one-time setup)
python preload_data_simple.py

### 6. Run the Streamlit app
streamlit run app/main.py

##  4. VISUALIZATIONS
Visualization	Description	Investment Insight
Morningstar Style Box	3x3 matrix of Cap Size vs Investment Style	Large Cap = stable, Small Cap = high growth potential
Risk-Return Map	Volatility vs Average Return	Higher Sharpe Ratio = better risk-adjusted returns
Cluster Distribution	Automatic fund groupings	Funds with similar characteristics cluster together
Style Distribution	Bar chart of style categories	See which investment styles dominate  portfolio
##  5. TECHNOLOGY STACK
Component	Technology	Purpose
Frontend	Streamlit	Interactive web dashboard
Machine Learning	Scikit-learn (K-Means)	Unsupervised clustering
Data Source	Yahoo Finance API	Real-time market data
Visualization	Plotly	Interactive charts and heatmaps
Data Processing	Pandas, NumPy	Feature engineering
##  6. METHODOLOGY
Features Used for Classification:

Feature	What It Measures	Investment Meaning
Volatility	Price fluctuation	Small Cap > Mid Cap > Large Cap
Momentum	Price trend direction	Growth > Blend > Value
Sharpe Ratio	Risk-adjusted return	Higher = better investment efficiency
Liquidity	Trading volume	Large funds = more liquid
How K-Means Clustering Works:

Feature Extraction: Calculate volatility, momentum, Sharpe ratio, and liquidity for each fund

Normalization: Scale features to have equal weight in clustering

Clustering: Group funds into k clusters based on feature similarity

Style Mapping: Map clusters to Morningstar Style Box based on:

Volatility ranking -> Market Cap (Low = Large Cap, High = Small Cap)

Momentum ranking -> Style (Low = Value, High = Growth)

##  7. EXAMPLE RESULTS
After running the application, you will see:

text
Morningstar Style Box:

+-------------+---------+---------+---------+
|             | Value   | Blend   | Growth  |
+-------------+---------+---------+---------+
| Large Cap   | SPYV    | IVV     | SPYG    |
| Mid Cap     | MDYV    | IJH     | MDYG    |
| Small Cap   | SLYV    | IJR     | SLYG    |
+-------------+---------+---------+---------+
##  8. TROUBLESHOOTING
Issue	Solution
ModuleNotFoundError	Run pip install -r requirements.txt
No data in app	Run python preload_data_simple.py first
Timezone errors	Latest code handles UTC conversion automatically
Port already in use	Run streamlit run app/main.py --server.port 8502
## 9. LICENSE
This project is licensed under the MIT License - see the LICENSE file for details.

## 10. AUTHOR
mcml1225

GitHub: @mcml1225


## 11. ACKNOWLEDGMENTS
Morningstar, Inc. for the Style Box methodology

Yahoo Finance for providing free market data API

Streamlit for the web framework

Scikit-learn for the ML implementation
