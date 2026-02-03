# Market Sentiment Analysis - Trader Performance vs Market Sentiment
## Data Science Intern Assignment - Primetrade.ai

### Project Overview
This project analyzes the relationship between Bitcoin market sentiment (Fear/Greed) and trader behavior/performance on Hyperliquid. The analysis reveals key patterns that can inform trading strategies.

### Setup Instructions

#### Prerequisites
- Python 3.7+
- Required packages: pandas, numpy, matplotlib, seaborn

#### Installation
```bash
# Install required packages
pip install pandas numpy matplotlib seaborn jupyter
```

#### Running the Analysis
```bash
# Option 1: Run Jupyter Notebook
jupyter notebook analysis.ipynb

# Option 2: Run Python script
python simple_analysis.py
```

### Project Structure
```
Market-Sentiment-Analysis/
├── README.md                           # This file
├── simple_analysis.py                  # Main analysis script
├── ANALYSIS SUMMARY.md                 # Summary of findings
├── fear_greed_index.csv                # Sentiment data
├── historical_data.csv                 # Trading data
├── output/                             # Generated charts and insights
│   ├── chart1_performance_by_sentiment.png
│   ├── chart2_behavior_by_sentiment.png
│   ├── chart3_segmentation_analysis.png
│   ├── chart4_segment_sentiment_heatmap.png
│   └── insights.txt
└── trader_metrics.csv                  # Processed metrics
```

### Data Sources
1. **Bitcoin Market Sentiment**: Fear/Greed classification by date
2. **Hyperliquid Trader Data**: Historical trades with PnL, leverage, size, etc.

### Key Findings
See [ANALYSIS_SUMMARY.md](ANALYSIS_SUMMARY.md) for detailed insights and strategy recommendations.

### Methodology
1. **Data Preparation**: Cleaned and aligned datasets by date
2. **Metric Creation**: Calculated daily PnL, win rate, trade frequency, leverage, long/short ratios
3. **Segmentation**: Divided traders into meaningful groups (leverage, frequency, consistency)
4. **Analysis**: Statistical comparison of performance and behavior across sentiment periods
5. **Visualization**: Created charts to illustrate patterns


