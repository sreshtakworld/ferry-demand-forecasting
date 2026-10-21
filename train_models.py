
import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.dummy import DummyRegressor
from math import sqrt

INPUT_FILE = "ferry_model_data.csv"
RESULT_FILE = "model_results.csv"
MODEL_DIR = "saved_models"

os.makedirs(MODEL_DIR, exist_ok=True)

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
    "Sales_Next_30m", "Redemptions_Next_30m",
    "Sales_Next_60m", "Redemptions_Next_60m",
    "Sales_Next_120m", "Redemptions_Next_120m",
]

results = []

for target in targets:
    data = df.dropna(subset=features + [target]).copy()

    split = int(len(data) * 0.8)
    train = data.iloc[:split]
    test = data.iloc[split:]

    if len(train) == 0 or len(test) == 0:
        print(f"Skipping {target}: insufficient data")
        continue

    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]

    baseline = DummyRegressor(strategy="median")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Save the trained model and the exact feature order.
    model_path = os.path.join(MODEL_DIR, f"{target}.joblib")
    joblib.dump({
        "model": model,
        "features": features,
        "target": target,
    }, model_path)

    baseline_mae = mean_absolute_error(y_test, baseline_predictions)
    model_mae = mean_absolute_error(y_test, predictions)
    model_rmse = sqrt(mean_squared_error(y_test, predictions))

    results.append({
        "Target": target,
        "Train_Rows": len(train),
        "Test_Rows": len(test),
        "Baseline_MAE": baseline_mae,
        "RandomForest_MAE": model_mae,
        "RandomForest_RMSE": model_rmse,
    })

    print(f"\nTarget: {target}")
    print(f"Training rows: {len(train):,}")
    print(f"Testing rows: {len(test):,}")
    print(f"Baseline MAE: {baseline_mae:.2f}")
    print(f"Random Forest MAE: {model_mae:.2f}")
    print(f"Random Forest RMSE: {model_rmse:.2f}")
    print(f"Saved model: {model_path}")

results_df = pd.DataFrame(results)
results_df.to_csv(RESULT_FILE, index=False)

print("\n--- SUMMARY ---")
print(results_df.to_string(index=False))
print(f"\nSaved results to {RESULT_FILE}")
print(f"Saved models in {MODEL_DIR}")
