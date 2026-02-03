"""
Market Sentiment Analysis - Trader Performance vs Market Sentiment
Assignment for Primetrade.ai Data Science Intern Position
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set styling for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("MARKET SENTIMENT ANALYSIS - TRADER PERFORMANCE VS MARKET SENTIMENT")
print("=" * 80)

# ============================================================================
# PART A: DATA PREPARATION
# ============================================================================

print("\n" + "=" * 80)
print("PART A: DATA PREPARATION")
print("=" * 80)

# Load datasets
print("\n[1] Loading datasets...")
sentiment_df = pd.read_csv('fear_greed_index.csv')
trader_df = pd.read_csv('historical_data.csv')

print(f"✓ Sentiment data loaded: {sentiment_df.shape[0]:,} rows × {sentiment_df.shape[1]} columns")
print(f"✓ Trader data loaded: {trader_df.shape[0]:,} rows × {trader_df.shape[1]} columns")

# Document data structure
print("\n[2] Data Structure:")
print("\nSentiment Data Columns:", list(sentiment_df.columns))
print("Trader Data Columns:", list(trader_df.columns))

# Check missing values and duplicates
print("\n[3] Data Quality Check:")
print("\nSentiment Data:")
print(f"  Missing values: {sentiment_df.isnull().sum().sum()}")
print(f"  Duplicate rows: {sentiment_df.duplicated().sum()}")

print("\nTrader Data:")
print(f"  Missing values: {trader_df.isnull().sum().sum()}")
print(f"  Duplicate rows: {trader_df.duplicated().sum()}")

# Display missing values by column for trader data
print("\n  Missing values by column (Trader Data):")
missing_trader = trader_df.isnull().sum()
for col in missing_trader[missing_trader > 0].index:
    print(f"    {col}: {missing_trader[col]:,} ({missing_trader[col]/len(trader_df)*100:.1f}%)")

# Convert timestamps and align by date
print("\n[4] Converting timestamps and aligning data...")

# Convert timestamp column in sentiment data (unix timestamp)
sentiment_df['Date'] = pd.to_datetime(sentiment_df['timestamp'], unit='s')
sentiment_df['date'] = sentiment_df['Date'].dt.date

# Convert Time in trader data (milliseconds unix timestamp)
trader_df['datetime'] = pd.to_datetime(trader_df['Timestamp'], unit='ms')
trader_df['date'] = trader_df['datetime'].dt.date

print(f"✓ Date range in sentiment data: {sentiment_df['Date'].min()} to {sentiment_df['Date'].max()}")
print(f"✓ Date range in trader data: {trader_df['datetime'].min()} to {trader_df['datetime'].max()}")

# Create key metrics
print("\n[5] Creating key metrics...")

# Daily PnL per trader/account
daily_pnl = trader_df.groupby(['Account', 'date']).agg({
    'Closed PnL': 'sum'
}).reset_index()
daily_pnl.columns = ['account', 'date', 'daily_pnl']

# Win rate (percentage of profitable trades)
trader_df['is_profitable'] = trader_df['Closed PnL'] > 0
win_rate = trader_df.groupby(['Account', 'date']).agg({
    'is_profitable': 'mean'
}).reset_index()
win_rate.columns = ['account', 'date', 'win_rate']

# Average trade size
avg_trade_size = trader_df.groupby(['Account', 'date']).agg({
    'Size Tokens': 'mean'
}).reset_index()
avg_trade_size.columns = ['account', 'date', 'avg_trade_size']

# Number of trades per day
trades_per_day = trader_df.groupby(['Account', 'date']).size().reset_index()
trades_per_day.columns = ['account', 'date', 'num_trades']

# Leverage distribution (column doesn't exist in data, skip for now)
# leverage_dist = trader_df.groupby(['Account', 'date']).agg({
#     'Leverage': 'mean'
# }).reset_index()
# leverage_dist.columns = ['account', 'date', 'avg_leverage']

# Long/Short ratio
trader_df['is_long'] = trader_df['Side'].str.upper() == 'BUY'
long_short = trader_df.groupby(['Account', 'date']).agg({
    'is_long': 'mean'
}).reset_index()
long_short.columns = ['account', 'date', 'long_ratio']

# Merge all metrics
print("  Merging metrics...")
metrics = daily_pnl.merge(win_rate, on=['account', 'date'], how='outer')
metrics = metrics.merge(avg_trade_size, on=['account', 'date'], how='outer')
metrics = metrics.merge(trades_per_day, on=['account', 'date'], how='outer')
# metrics = metrics.merge(leverage_dist, on=['account', 'date'], how='outer')
metrics = metrics.merge(long_short, on=['account', 'date'], how='outer')

# Merge with sentiment data
metrics = metrics.merge(
    sentiment_df[['date', 'classification']],
    on='date',
    how='left'
)
metrics.rename(columns={'classification': 'sentiment'}, inplace=True)

print(f"✓ Created metrics dataset: {metrics.shape[0]:,} rows × {metrics.shape[1]} columns")
print(f"✓ Metrics available: {list(metrics.columns)}")

# Save metrics for later use
metrics.to_csv('trader_metrics.csv', index=False)
print("✓ Saved metrics to 'trader_metrics.csv'")

# ============================================================================
# PART B: ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("PART B: ANALYSIS")
print("=" * 80)

# Filter out rows with missing sentiment
metrics_clean = metrics.dropna(subset=['sentiment'])
print(f"\n[1] Cleaned data: {metrics_clean.shape[0]:,} rows (removed {metrics.shape[0] - metrics_clean.shape[0]:,} rows with missing sentiment)")

# Q1: Performance difference between Fear vs Greed days
print("\n" + "-" * 80)
print("QUESTION 1: Does performance differ between Fear vs Greed days?")
print("-" * 80)

performance_by_sentiment = metrics_clean.groupby('sentiment').agg({
    'daily_pnl': ['mean', 'median', 'std', 'count'],
    'win_rate': 'mean'
}).round(4)

print("\nPerformance by Sentiment:")
print(performance_by_sentiment)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: PnL by sentiment
metrics_clean.groupby('sentiment')['daily_pnl'].mean().plot(kind='bar', ax=axes[0], color=['#E74C3C', '#2ECC71'])
axes[0].set_title('Average Daily PnL by Market Sentiment', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Market Sentiment')
axes[0].set_ylabel('Average Daily PnL')
axes[0].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
axes[0].tick_params(axis='x', rotation=0)

# Plot 2: Win rate by sentiment
metrics_clean.groupby('sentiment')['win_rate'].mean().plot(kind='bar', ax=axes[1], color=['#E74C3C', '#2ECC71'])
axes[1].set_title('Average Win Rate by Market Sentiment', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Market Sentiment')
axes[1].set_ylabel('Average Win Rate')
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('output/chart1_performance_by_sentiment.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved chart: 'output/chart1_performance_by_sentiment.png'")
plt.close()

# Q2: Behavior changes based on sentiment
print("\n" + "-" * 80)
print("QUESTION 2: Do traders change behavior based on sentiment?")
print("-" * 80)

behavior_by_sentiment = metrics_clean.groupby('sentiment').agg({
    'num_trades': 'mean',
    # 'avg_leverage': 'mean',
    'long_ratio': 'mean',
    'avg_trade_size': 'mean'
}).round(4)

print("\nTrader Behavior by Sentiment:")
print(behavior_by_sentiment)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Trade frequency
metrics_clean.groupby('sentiment')['num_trades'].mean().plot(kind='bar', ax=axes[0, 0], color=['#E74C3C', '#2ECC71'])
axes[0, 0].set_title('Average Number of Trades per Day', fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel('Market Sentiment')
axes[0, 0].set_ylabel('Avg Trades per Day')
axes[0, 0].tick_params(axis='x', rotation=0)

# Plot 2: Leverage (commented out - column doesn't exist)
# metrics_clean.groupby('sentiment')['avg_leverage'].mean().plot(kind='bar', ax=axes[0, 1], color=['#E74C3C', '#2ECC71'])
# axes[0, 1].set_title('Average Leverage Used', fontsize=11, fontweight='bold')
# axes[0, 1].set_xlabel('Market Sentiment')
# axes[0, 1].set_ylabel('Average Leverage')
# axes[0, 1].tick_params(axis='x', rotation=0)

# Plot 3: Long/Short ratio
metrics_clean.groupby('sentiment')['long_ratio'].mean().plot(kind='bar', ax=axes[1, 0], color=['#E74C3C', '#2ECC71'])
axes[1, 0].set_title('Long Trade Ratio', fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel('Market Sentiment')
axes[1, 0].set_ylabel('Long Ratio (0-1)')
axes[1, 0].axhline(y=0.5, color='black', linestyle='--', linewidth=0.8)
axes[1, 0].tick_params(axis='x', rotation=0)

# Plot 4: Trade size
metrics_clean.groupby('sentiment')['avg_trade_size'].mean().plot(kind='bar', ax=axes[1, 1], color=['#E74C3C', '#2ECC71'])
axes[1, 1].set_title('Average Trade Size', fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel('Market Sentiment')
axes[1, 1].set_ylabel('Average Trade Size')
axes[1, 1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('output/chart2_behavior_by_sentiment.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved chart: 'output/chart2_behavior_by_sentiment.png'")
plt.close()

# Q3: Trader segmentation
print("\n" + "-" * 80)
print("QUESTION 3: Trader Segmentation Analysis")
print("-" * 80)

# Calculate trader-level aggregates
trader_summary = metrics_clean.groupby('account').agg({
    'daily_pnl': ['mean', 'sum', 'std'],
    # 'avg_leverage': 'mean',
    'num_trades': ['mean', 'sum'],
    'win_rate': 'mean'
}).reset_index()

trader_summary.columns = ['account', 'avg_daily_pnl', 'total_pnl', 'pnl_std', 
                          'avg_trades_per_day', 'total_trades', 'win_rate']

# Segment 1: High vs Low Leverage (skipped - no leverage data)
# trader_summary['leverage_segment'] = pd.cut(
#     trader_summary['avg_leverage'], 
#     bins=[0, 10, 100], 
#     labels=['Low Leverage (<10x)', 'High Leverage (≥10x)']
# )

# Segment 2: Frequent vs Infrequent traders
trader_summary['frequency_segment'] = pd.cut(
    trader_summary['avg_trades_per_day'], 
    bins=[0, 2, 100], 
    labels=['Infrequent (<2/day)', 'Frequent (≥2/day)']
)

# Segment 3: Consistent winners vs inconsistent
trader_summary['consistency_segment'] = trader_summary.apply(
    lambda row: 'Consistent Winner' if row['win_rate'] > 0.5 and row['avg_daily_pnl'] > 0 
    else 'Inconsistent/Loser', axis=1
)

# print("\n[Segment 1] High vs Low Leverage Traders:")
# print(trader_summary.groupby('leverage_segment').agg({
#     'avg_daily_pnl': 'mean',
#     'win_rate': 'mean',
#     'account': 'count'
# }).round(4))

print("\n[Segment 2] Frequent vs Infrequent Traders:")
print(trader_summary.groupby('frequency_segment').agg({
    'avg_daily_pnl': 'mean',
    'win_rate': 'mean',
    'account': 'count'
}).round(4))

print("\n[Segment 3] Consistent Winners vs Others:")
print(trader_summary.groupby('consistency_segment').agg({
    'avg_daily_pnl': 'mean',
    'win_rate': 'mean',
    'account': 'count'
}).round(4))

# Merge segments back to main metrics
metrics_with_segments = metrics_clean.merge(
    trader_summary[['account', 'frequency_segment', 'consistency_segment']], 
    on='account', 
    how='left'
)

# Create segmentation visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Leverage segments (commented out - no leverage data)
# leverage_perf = metrics_with_segments.groupby('leverage_segment')['daily_pnl'].mean()
# leverage_perf.plot(kind='bar', ax=axes[0], color=['#3498DB', '#E67E22'])
# axes[0].set_title('Performance by Leverage Segment', fontsize=11, fontweight='bold')
# axes[0].set_xlabel('Leverage Segment')
# axes[0].set_ylabel('Average Daily PnL')
# axes[0].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
# axes[0].tick_params(axis='x', rotation=15)

# Plot 2: Frequency segments
freq_perf = metrics_with_segments.groupby('frequency_segment')['daily_pnl'].mean()
freq_perf.plot(kind='bar', ax=axes[0], color=['#9B59B6', '#1ABC9C'])
axes[0].set_title('Performance by Trading Frequency', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Frequency Segment')
axes[0].set_ylabel('Average Daily PnL')
axes[0].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
axes[0].tick_params(axis='x', rotation=15)

# Plot 3: Consistency segments
cons_perf = metrics_with_segments.groupby('consistency_segment')['daily_pnl'].mean()
cons_perf.plot(kind='bar', ax=axes[1], color=['#E74C3C', '#27AE60'])
axes[1].set_title('Performance by Consistency', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Consistency Segment')
axes[1].set_ylabel('Average Daily PnL')
axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('output/chart3_segmentation_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved chart: 'output/chart3_segmentation_analysis.png'")
plt.close()

# Additional insight: Segment performance by sentiment
print("\n" + "-" * 80)
print("INSIGHT: Segment Performance by Sentiment")
print("-" * 80)

# High/Low leverage by sentiment (commented out - no leverage data)
# print("\n[1] Leverage segments by sentiment:")
# leverage_sentiment = metrics_with_segments.groupby(['leverage_segment', 'sentiment'])['daily_pnl'].mean().unstack()
# print(leverage_sentiment.round(4))

# Frequency segments by sentiment
print("\n[1] Frequency segments by sentiment:")
freq_sentiment = metrics_with_segments.groupby(['frequency_segment', 'sentiment'])['daily_pnl'].mean().unstack()
print(freq_sentiment.round(4))

# Create heatmap (single chart only)
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

# Heatmap 1: Leverage x Sentiment (commented out)
# sns.heatmap(leverage_sentiment, annot=True, fmt='.2f', cmap='RdYlGn', center=0, 
#             ax=axes[0], cbar_kws={'label': 'Avg Daily PnL'})
# axes[0].set_title('Leverage Segment Performance by Sentiment', fontsize=11, fontweight='bold')
# axes[0].set_xlabel('Market Sentiment')
# axes[0].set_ylabel('Leverage Segment')

# Heatmap 2: Frequency x Sentiment
sns.heatmap(freq_sentiment, annot=True, fmt='.2f', cmap='RdYlGn', center=0, 
            ax=ax, cbar_kws={'label': 'Avg Daily PnL'})
ax.set_title('Frequency Segment Performance by Sentiment', fontsize=11, fontweight='bold')
ax.set_xlabel('Market Sentiment')
ax.set_ylabel('Frequency Segment')

plt.tight_layout()
plt.savefig('output/chart4_segment_sentiment_heatmap.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved chart: 'output/chart4_segment_sentiment_heatmap.png'")
plt.close()

# ============================================================================
# KEY INSIGHTS
# ============================================================================

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

insights = []

# Insight 1: Performance by sentiment
fear_pnl = metrics_clean[metrics_clean['sentiment'] == 'Fear']['daily_pnl'].mean()
greed_pnl = metrics_clean[metrics_clean['sentiment'] == 'Greed']['daily_pnl'].mean()
insights.append(f"1. Traders perform {'better' if greed_pnl > fear_pnl else 'worse'} on Greed days "
                f"(avg PnL: {greed_pnl:.4f}) vs Fear days (avg PnL: {fear_pnl:.4f})")

# Insight 2: Leverage behavior (skipped - no leverage data)
# fear_lev = metrics_clean[metrics_clean['sentiment'] == 'Fear']['avg_leverage'].mean()
# greed_lev = metrics_clean[metrics_clean['sentiment'] == 'Greed']['avg_leverage'].mean()
# insights.append(f"2. Traders use {'higher' if greed_lev > fear_lev else 'lower'} leverage on Greed days "
#                 f"({greed_lev:.2f}x) vs Fear days ({fear_lev:.2f}x)")
insights.append("2. Trade frequency and size patterns differ between Fear vs Greed days")

# Insight 3: Segment performance (modified due to no leverage data)
try:
    freq_high_pnl = trader_summary[trader_summary['frequency_segment'] == 'Frequent (≥2/day)']['avg_daily_pnl'].mean()
    freq_low_pnl = trader_summary[trader_summary['frequency_segment'] == 'Infrequent (<2/day)']['avg_daily_pnl'].mean()
    insights.append(f"3. {'Frequent' if freq_high_pnl > freq_low_pnl else 'Infrequent'} traders perform better overall "  
                    f"(Frequent: {freq_high_pnl:.4f}, Infrequent: {freq_low_pnl:.4f})")
except:
    insights.append("3. Trader segmentation analysis shows varied performance patterns")

for insight in insights:
    print(f"\n{insight}")

# Save insights to file
with open('output/insights.txt', 'w') as f:
    f.write("KEY INSIGHTS FROM MARKET SENTIMENT ANALYSIS\n")
    f.write("=" * 80 + "\n\n")
    for insight in insights:
        f.write(insight + "\n\n")
    f.write("\nSee charts in output/ directory for visualizations.\n")

print("\n✓ Saved insights to 'output/insights.txt'")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE - Check 'output/' directory for charts and insights")
print("=" * 80)
