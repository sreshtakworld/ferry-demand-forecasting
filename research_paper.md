# Short-Term Ferry Ticket Demand Forecasting and Predictive Decision Support System

## Abstract

Ferry operators need timely estimates of passenger ticket demand to support operational planning. Historical analysis helps identify past demand patterns, but predictive models can provide estimates of expected ticket sales and redemptions over short future time horizons. This project presents a machine-learning-based forecasting and predictive decision support system using historical Toronto Island ferry ticket data.

The system uses timestamp-based calendar features and historical lag features to predict sales and redemptions over the next 30, 60, and 120 minutes. Six Random Forest regression models are trained and evaluated using chronological training and testing partitions. The resulting models are integrated into a Streamlit dashboard that displays forecasts, model evaluation metrics, recent historical records, and demand-category guidance.

The experimental results demonstrate the feasibility of building a short-term forecasting prototype from historical ticket records. However, forecast errors increase with the prediction horizon, and extreme demand peaks may be underestimated. The system is intended to support human decision-making rather than replace operational judgment. Its current predictions are based on historical data ending December 21, 2025, and are not live forecasts.

**Keywords:** Ferry demand forecasting, Random Forest regression, machine learning, time-series features, predictive analytics, decision support system.

## 1. Introduction

Transportation operators must make decisions using changing demand conditions. For ferry services, anticipating ticket sales and redemptions can help operators understand expected activity and prepare for potential increases in demand.

Traditional historical reporting describes demand that has already occurred. Short-term forecasting extends this analysis by estimating demand over upcoming time intervals. Such estimates can support monitoring and planning, provided their limitations are understood.

This project develops a short-term ferry ticket demand forecasting system using historical ticket records. It predicts sales and redemptions for three future horizons: 30 minutes, 60 minutes, and 120 minutes. A web-based dashboard presents the results in an accessible format.

The primary objectives are:

1. Prepare historical ticket records for machine-learning analysis.
2. Construct timestamp-based and lag-based predictive features.
3. Train and evaluate six Random Forest regression models.
4. Present short-term predictions and evaluation results through a dashboard.
5. Communicate forecast limitations to support responsible interpretation.

## 2. Dataset Description

The dataset, `Toronto Island Ferry Tickets.csv`, contains timestamped ticket records with the following principal fields:

- `_id`: Record identifier.
- `Timestamp`: Date and time associated with the record.
- `Redemption Count`: Recorded ticket redemptions.
- `Sales Count`: Recorded ticket sales.

The dataset contains approximately 261,538 records, covering May 1, 2015, through December 21, 2025. Missing timestamp intervals are present. The forecasting pipeline does not treat missing intervals as zero-demand observations.

Data preparation includes parsing timestamps, ordering records chronologically, constructing prediction targets, and generating features. Forecast targets are retained only when every required future 15-minute timestamp is available.

## 3. Proposed Methodology

### 3.1 Forecasting targets

The system estimates the total sales and redemptions over the next 30, 60, and 120 minutes. The targets are constructed from consecutive 15-minute intervals following the current timestamp.

For a horizon containing \(k\) intervals, the target is:

\[
Y_t^{(k)}=\sum_{i=1}^{k} y_{t+i}
\]

where \(y_{t+i}\) represents the recorded count in a future 15-minute interval. The values of \(k\) are 2, 4, and 8 for the respective horizons.

A target is excluded if any required future timestamp is missing. This prevents incomplete windows from being interpreted as complete demand totals.

### 3.2 Feature engineering

The model uses calendar features and historical lag features.

**Calendar features:**
- Hour
- Minute
- Day of week
- Month
- Weekend indicator

**Historical lag features:**
- Sales and redemptions 15 minutes earlier
- Sales and redemptions 30 minutes earlier
- Sales and redemptions 60 minutes earlier
- Sales and redemptions 120 minutes earlier

Lag values are associated with exact historical timestamps. Missing historical values are not automatically interpreted as zero.

### 3.3 Random Forest regression

Random Forest regression is used because it can model nonlinear relationships between calendar patterns, recent ticket activity, and future demand.

Six separate models are trained: one for each combination of demand type and forecast horizon.

The principal model parameters are:

| Parameter | Value |
|---|---:|
| Number of trees | 100 |
| Maximum tree depth | 12 |
| Minimum samples per leaf | 5 |
| Random seed | 42 |
| Parallel jobs | -1 |

The data is divided chronologically, with 80% used for training and 20% for testing. This preserves temporal ordering rather than randomly mixing earlier and later records.

### 3.4 Evaluation metrics

Model performance is assessed using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

MAE measures the average absolute difference between predictions and observed values. RMSE gives greater weight to larger errors. Both metrics are reported in the units of the corresponding target counts.

### 3.5 Predictive decision support dashboard

The application is implemented using Python, pandas, scikit-learn, joblib, and Streamlit. It loads the prepared dataset and six saved models, generates predictions from the latest eligible historical feature row, and displays results for all three forecast horizons.

The dashboard also presents evaluation metrics, recent historical records, and demand-category guidance based on historical sales percentiles. The categories are indicative monitoring labels, not verified ferry capacity or staffing thresholds.

## 4. Experimental Results

The chronological test results are summarized below.

| Forecast target | Test records | MAE | RMSE |
|---|---:|---:|---:|
| Sales — next 30 minutes | 34,856 | 38.85 | 111.30 |
| Redemptions — next 30 minutes | 34,856 | 38.43 | 113.60 |
| Sales — next 60 minutes | 32,465 | 71.47 | 172.09 |
| Redemptions — next 60 minutes | 32,465 | 69.67 | 176.64 |
| Sales — next 120 minutes | 28,625 | 142.78 | 286.37 |
| Redemptions — next 120 minutes | 28,625 | 138.14 | 293.81 |

The results show that prediction errors increase as the forecasting horizon becomes longer. This is consistent with the greater uncertainty involved in estimating demand farther into the future.

The MAE values indicate the average absolute prediction error for each target. The RMSE values are larger, suggesting that some observations have substantially larger errors than the average absolute error alone would indicate. Extreme demand peaks are a particular limitation because the models may underestimate sudden surges.

These metrics should be interpreted in the context of the target count definitions and the historical test period. They do not establish that the system improves actual ferry operations or guarantees reliable predictions during unusual events.

## 5. Discussion

The prototype demonstrates an end-to-end workflow from historical data preparation to model evaluation and dashboard-based presentation. Separate models allow sales and redemptions to be forecast independently at different horizons.

The dashboard makes results accessible without requiring users to execute machine-learning scripts manually. Its explicit historical-data warning is important because the latest available dataset timestamp is December 21, 2025. Consequently, the displayed forecasts are based on the most recent eligible historical record, not current ticket activity.

The system can support exploratory analysis and demonstrate predictive decision support. Operational use would require more recent data, ongoing validation, comparison with baseline forecasting methods, and assessment of errors during peak periods.

## 6. Limitations and Future Work

The current prototype has several limitations:

1. **Historical data:** The dataset does not provide live ticket activity.
2. **Peak demand errors:** Sudden or extreme demand increases may be underestimated.
3. **External variables:** The current features do not explicitly incorporate weather, holidays, special events, service disruptions, or ferry capacity.
4. **Evaluation scope:** Reported metrics describe the specified chronological test partitions and do not guarantee future performance.
5. **Operational validation:** The demand categories are based on historical percentiles and have not been validated as capacity or staffing thresholds.

Future work may include incorporating weather and event data, comparing Random Forest with seasonal-naive and other forecasting baselines, testing alternative time-series models, evaluating performance across seasons and peak-demand periods, and implementing a reliable pipeline for regularly updated data.

## 7. Conclusion

This project presents a short-term ferry ticket demand forecasting and predictive decision support prototype using historical ticket records and Random Forest regression. Six models estimate sales and redemptions over 30-, 60-, and 120-minute horizons, and a Streamlit dashboard presents predictions, evaluation metrics, and recent historical observations.

The evaluation shows that the prototype can produce short-term estimates, while errors increase at longer horizons and extreme peaks remain challenging. The application is therefore best treated as a demonstration and exploratory decision support tool. Further validation with current data and operational context is necessary before relying on it for real-world planning.

## References

1. Toronto Island ferry ticket historical dataset, `Toronto Island Ferry Tickets.csv`, covering May 1, 2015, to December 21, 2025. Dataset provenance and redistribution terms should be verified and documented before formal publication.
2. Scikit-learn documentation, Random Forest regression and model evaluation: https://scikit-learn.org/stable/
3. Streamlit documentation: https://docs.streamlit.io/
4. pandas documentation: https://pandas.pydata.org/docs/

**Project demonstration:** https://ferry-demand-forecasting-sreshta.streamlit.app/

**Source code:** https://github.com/sreshtakworld/ferry-demand-forecasting
