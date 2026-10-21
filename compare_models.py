
import pandas as pd
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

INPUT_FILE = "ferry_model_data.csv"
OUTPUT_FILE = "fair_model_comparison.csv"

df = pd.read_csv(INPUT_FILE, parse_dates=["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

calendar_features = [
    "Hour", "Minute", "Day_of_Week", "Month", "Is_Weekend"
]
lag_features = [
    "Sales_Lag_15m", "Redemptions_Lag_15m",
    "Sales_Lag_30m", "Redemptions_Lag_30m",
    "Sales_Lag_60m", "Redemptions_Lag_60m",
    "Sales_Lag_120m", "Redemptions_Lag_120m",
]
features = calendar_features + lag_features

targets = [
    ("Sales_Next_30m", "Sales Count", 30),
    ("Redemptions_Next_30m", "Redemption Count", 30),
    ("Sales_Next_60m", "Sales Count", 60),
    ("Redemptions_Next_60m", "Redemption Count", 60),
    ("Sales_Next_120m", "Sales Count", 120),
    ("Redemptions_Next_120m", "Redemption Count", 120),
]

timestamp_to_position = pd.Series(
    df.index.to_numpy(), index=df["Timestamp"]
)

results = []

for target, source, minutes in targets:
    data = df.dropna(subset=features + [target]).copy()

    # Chronological split: older rows train, newer rows test.
    split = int(len(data) * 0.8)
    train = data.iloc[:split]
    test = data.iloc[split:]

    # Build recent-demand baseline from exact earlier timestamps.
    steps = minutes // 15
    baseline = pd.Series(0.0, index=test.index)
    baseline_valid = pd.Series(True, index=test.index)

    for step in range(1, steps + 1):
        past_times = test["Timestamp"] - pd.Timedelta(
            minutes=15 * step
        )
        positions = past_times.map(timestamp_to_position)
        exists = positions.notna()
        baseline_valid &= exists

        valid_test_indices = test.index[exists.to_numpy()]
        source_positions = positions[exists].astype(int)

        baseline.loc[valid_test_indices] += df.loc[
            source_positions, source
        ].to_numpy()

    comparable_test = test.loc[baseline_valid]

    if comparable_test.empty:
        print(f"Skipping {target}: no comparable test rows")
        continue

    X_train = train[features]
    y_train = train[target]
    X_test = comparable_test[features]
    y_test = comparable_test[target]
    baseline_predictions = baseline.loc[comparable_test.index]

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    baseline_mae = mean_absolute_error(y_test, baseline_predictions)
    rf_mae = mean_absolute_error(y_test, predictions)
    rf_rmse = sqrt(mean_squared_error(y_test, predictions))
    improvement = (
        (baseline_mae - rf_mae) / baseline_mae * 100
        if baseline_mae else 0.0
    )

    results.append({
        "Target": target,
        "Train_Rows": len(train),
        "Same_Test_Rows": len(comparable_test),
        "Recent_Baseline_MAE": baseline_mae,
        "RandomForest_MAE": rf_mae,
        "RandomForest_RMSE": rf_rmse,
        "MAE_Improvement_Percent": improvement,
    })

    print(f"\nTarget: {target}")
    print(f"Training rows: {len(train):,}")
    print(f"Same test rows: {len(comparable_test):,}")
    print(f"Recent baseline MAE: {baseline_mae:.2f}")
    print(f"Random Forest MAE: {rf_mae:.2f}")
    print(f"Random Forest RMSE: {rf_rmse:.2f}")
    print(f"MAE improvement: {improvement:.2f}%")

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILE, index=False)

print("\n--- FAIR COMPARISON SUMMARY ---")
print(results_df.to_string(index=False))
print(f"\nSaved results to {OUTPUT_FILE}")
