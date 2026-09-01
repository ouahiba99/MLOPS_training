# Notebook 04 — EDA Findings

This summary was generated from the training split only.

## Target
- **Finding:** The training late rate is 9.02%.
- **Implication:** The target is imbalanced, so accuracy alone is not sufficient. Use precision, recall, F1 and especially PR-AUC / Average Precision when evaluating models.

## Missingness
- **Finding:** order_delivered_carrier_date has the largest absolute late-rate difference between missing and present observations (90.98 percentage points).
- **Implication:** Investigate whether missingness reflects an operational process state. A missingness indicator is appropriate only when that state is known at prediction time.

## Numerical
- **Finding:** payment_count has the strongest absolute skewness (22.42).
- **Implication:** Investigate extreme values rather than automatically deleting them. Tree-based models are generally less sensitive to monotonic skew.

## Categorical
- **Finding:** customer_city has 3,747 unique categories.
- **Implication:** Avoid blindly one-hot encoding very high-cardinality variables. Use rare-category grouping or another leakage-safe strategy.

## Temporal
- **Finding:** Order purchase timestamps show calendar structure that can be examined by month, weekday, hour, weekend and holiday.
- **Implication:** Engineer calendar features from purchase-time information rather than passing raw timestamps directly to the model.

## Holidays
- **Finding:** Holiday and non-holiday late rates were compared using the order purchase date.
- **Implication:** If the difference is material and stable, a holiday indicator can be considered as a prediction-time feature.

## Leakage
- **Finding:** 5 columns were flagged as post-outcome or derived from post-outcome information.
- **Implication:** Exclude these variables from Notebook 5 model features. They may be used here only to demonstrate why they are leakage.

## Evaluation Strategy
- **Finding:** Temporal behavior was examined for distribution shift, but this notebook does not change the train/validation/test split.
- **Implication:** Notebook 5/6 must use the split strategy documented in Notebook 3 and keep validation/test data out of EDA-driven feature decisions.
