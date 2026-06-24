
E-commerce EDA Report

Phase 1 - Data Audit

- First checked the dataset shape, column types and missing values.
- A few duplicate rows were present in the data.
- Generated summary statistics for both numerical and categorical columns.
- Some missing values were found in a few fields, so data cleaning is required before modeling.

Phase 2 - Skewness and Distribution

- Calculated skewness for all numerical features.
- Plotted the distribution of final_price using a histogram.
- The distribution is positively skewed, which means most products have lower prices while a small number have very high prices.

Phase 3 - Categorical Analysis

- Counted unique values in all categorical columns.
- Identified columns with high cardinality.
- Columns such as city can create too many categories, so direct one-hot encoding may not be the best option.

Phase 4 - Return Analysis

- Calculated the overall return percentage.
- Analyzed the most common return reasons.
- Compared return rates across categories and shipping methods.
- Some categories showed higher return behavior than others.

Phase 5 - Feature Engineering

- Filled missing numerical values using median.
- Filled missing categorical values using the most frequent value.
- Applied standard scaling on numerical features.
- Used one-hot encoding for nominal categorical features.
- Used ordinal encoding for loyalty tiers:
  Bronze < Silver < Gold < Platinum.
- Combined all preprocessing steps using a ColumnTransformer pipeline.

Phase 6 - Correlation and Feature Importance

- Generated a correlation heatmap for numerical features.
- Observed a few features with noticeable correlation.
- Calculated mutual information scores to understand feature importance.
- Features like price, category, shipping method and delivery time appeared more useful for predicting returns.

Deliverables

- Data audit summary
- Distribution and skewness analysis
- Categorical feature analysis
- Return pattern insights
- Feature engineering pipeline
- Correlation and feature importance analysis

Conclusion

The dataset required basic cleaning and preprocessing before model building. High-cardinality categorical features need special handling, and return prediction appears to be influenced by factors such as pricing, category, shipping method and delivery duration.
