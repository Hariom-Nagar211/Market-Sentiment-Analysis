 pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Data Preparation
 load_data(file_path):
    data = pdread_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data

def calculate_returns(data):
    data['Daily_Return'] = data['Close'].pct_change()
    return data

# Sentiment Analysis
def identify_sentiment(data, fear_threshold, greed_threshold):
    data['Sentiment'] = np.where(data['Greed_Fear_Index'] < fear_threshold, 'Fear', 
                                 np.where(data['Greed_Fear_Index'] > greed_threshold, 'Greed', 'Neutral'))
    return data

# Performance Metrics
def calculate_metrics(data):
    metrics = {
        'Average_Daily_Return': data['Daily_Return'].mean(),
        'Max_Drawdown': (data['Close'] / data['Close'].cummax() - 1).min(),
        'Sharpe_Ratio': (data['Daily_Return'].mean() / data['Daily_Return'].std()) * np.sqrt(252)
    }
    return metrics

# Segmentation Analysis
def segment_traders(data, frequency_threshold, consistency_threshold):
    data['Trading_Frequency'] = pd.qcut(data['Trades'], q=3, labels=False)
    data['Consistency'] = pd.qcut(data['Win_Rate'], q=3, labels=False)
    return data

# Visualization
def plot_performance(data):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Sentiment', y='Daily_Return', data=data)
    plt.title('Average Daily Return by Sentiment')
    plt.show()

def plot_heatmap(data):
    heatmap_data = pd.crosstab(data['Trading_Frequency'], data['Consistency'])
    sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu')
    plt.title('Heatmap of Trading Frequency and Consistency')
    plt.show()

# Insights Extraction
def extract_insights(data):
    insights = {
        'Best_Period': data.groupby('Sentiment')['Daily_Return'].mean().idxmax(),
        'Frequency_Insight': data.groupby(['Trading_Frequency', 'Sentiment'])['Daily_Return'].mean()
    }
    return insights

# Main Function
def main():
    file_path = 'trader_data.csv'
    fear_threshold = -0.1
    greed_threshold = 0.1
    frequency_threshold = 3
    consistency_threshold = 3

    data = load_data(file_path)
    data = calculate_returns(data)
    data = identify_sentiment(data, fear_threshold, greed_threshold)
    metrics = calculate_metrics(data)
    data = segment_traders(data, frequency_threshold, consistency_threshold)
    
    plot_performance(data)
    plot_heatmap(data)
    
    insights = extract_insights(data)
    print(insights)

if __name__ == "__main__":
    main()
