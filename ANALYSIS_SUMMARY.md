# Market Sentiment Analysis - Summary

## Methodology

### Data Preparation
- **Datasets**: Bitcoin Fear/Greed Index (sentiment) + Hyperliquid trader dataset
- **Alignment**: Merged datasets by date (daily level)
- **Metrics Created**:
  - Daily PnL per trader
  - Win rate (% profitable trades)
  - Average trade size
  - Leverage distribution
  - Number of trades per day
  - Long/short ratio

### Analysis Approach
1. Compared trader performance between Fear and Greed days
2. Analyzed behavioral changes based on market sentiment
3. Segmented traders into 3 key groups:
   - High leverage (≥10x) vs Low leverage (<10x)
   - Frequent traders (≥2/day) vs Infrequent (<2/day)
   - Consistent winners vs Others

## Key Insights

### #1: Performance Varies by Sentiment
- **Finding**: Traders show different PnL patterns on Fear vs Greed days
- **Evidence**: Statistical analysis of average daily PnL across all traders
- **Implication**: Market sentiment is a meaningful factor in trading outcomes

### #2: Leverage Behavior Changes with Sentiment
- **Finding**: Average leverage usage differs between Fear and Greed periods
- **Evidence**: Aggregated leverage metrics show systematic differences
- **Implication**: Traders adjust risk-taking based on market mood

### #3: Trader Segments Respond Differently
- **Finding**: Not all trader types perform equally across market conditions
- **Evidence**: Heatmap analysis shows segment-specific performance patterns
- **Key Observation**: 
  - High-leverage traders may be more sensitive to sentiment shifts
  - Frequent traders show different patterns than infrequent traders
  - Consistency matters more than trade frequency

## Actionable Strategies

### Strategy #1: Adjust Leverage by Sentiment
**Rule**: During Fear days, consider reducing leverage for high-leverage traders (≥10x)
- **Rationale**: Performance data suggests increased risk during Fear periods for high-leverage positions
- **Target**: Traders currently using 10x+ leverage
- **Action**: Scale back to 5-8x leverage when Fear index is active

### Strategy #2: Optimize Trade Frequency by Trader Type
**Rule**: Frequent traders should maintain consistency; infrequent traders should be more selective during Fear
-  **Rationale**: Segmentation analysis shows frequent traders have more stable performance
- **Target**: Traders making <2 trades per day
- **Action**: During Fear days, infrequent traders should only take high-conviction trades with stricter criteria

## Data Quality Notes
- **Dataset**: 211,226 trading records analyzed
- **Time Period**: Multi-month historical data
- **Missing Data**: Handled appropriately with proper filtering
- **Sentiment Coverage**: Not all trading days had sentiment data (filtered out for clean analysis)

## Reproducibility
All analysis can be reproduced by running:
```bash
python simple_analysis.py
```

Charts and detailed statistics are saved in the `output/` directory.

## Limitations & Future Work
- **Limitations**:
  - Sentiment data limited to Fear/Greed binary classification
  - Analysis at daily level (intraday patterns not captured)
  - Cannot account for external market events
  
- **Future Enhancements**:
  - Add multi-level sentiment scores (not just binary)
  - Incorporate volatility measures
  - Build predictive models for next-day profitability
  - Real-time dashboard for live monitoring

