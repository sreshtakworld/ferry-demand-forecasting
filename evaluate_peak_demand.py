
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

df = pd.read_csv("ferry_model_data.csv", parse_dates=["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

features = [
    "Hour", "Minute", "Day_of_Week", "Month", "Is_Weekend",
    "Sales_Lag_15m", "Redemptions_Lag_15m",
    "Sales_Lag_30m", "Redemptions_Lag_30m",
    "Sales_Lag_60m", "Redemptions_Lag_60m",
    "Sales_Lag_120m", "Redemptions_Lag_120m",
]

target = "Sales_Next_30m"

data = df.dropna(subset=features + [target]).copy()
split = int(len(data) * 0.8)

train = data.iloc[:split]
test = data.iloc[split:]

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)

model.fit(train[features], train[target])
predictions = model.predict(test[features])

results = test[["Timestamp", target]].copy()
results["Predicted"] = predictions
results["Absolute_Error"] = (
    results[target] - results["Predicted"]
).abs()

# Define peak demand using the 90th percentile of the training targets.
peak_threshold = train[target].quantile(0.90)

groups = {
    "All test periods": results,
    "Ordinary demand (actual below training 90th percentile)": (
        results[results[target] < peak_threshold]
    ),
    "Peak demand (actual at or above training 90th percentile)": (
        results[results[target] >= peak_threshold]
    ),
}

print(f"Training 90th-percentile peak threshold: {peak_threshold:.1f}")

for name, subset in groups.items():
    if subset.empty:
        continue

    mae = mean_absolute_error(subset[target], subset["Predicted"])
    rmse = sqrt(mean_squared_error(subset[target], subset["Predicted"]))

    print(f"\n{name}")
    print(f"Rows: {len(subset):,}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

results.to_csv("sales_30m_test_predictions.csv", index=False)
print("\nSaved individual test predictions to sales_30m_test_predictions.csv")

