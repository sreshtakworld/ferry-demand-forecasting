
import os
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

DATA_FILE = "ferry_model_data.csv"
MODEL_DIR = "saved_models"
OUTPUT_FILE = "historical_validation.csv"

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

df = pd.read_csv(DATA_FILE, parse_dates=["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

all_results = []

for target in targets:
    model_path = os.path.join(MODEL_DIR, f"{target}.joblib")
    saved = joblib.load(model_path)
    model = saved["model"]
    model_features = saved["features"]

    # Match the chronological 80/20 split used during training.
    data = df.dropna(subset=model_features + [target]).copy()
    split = int(len(data) * 0.8)
    test = data.iloc[split:].copy()

    test["Predicted"] = model.predict(test[model_features])
    test["Absolute_Error"] = (
        test[target] - test["Predicted"]
    ).abs()

    mae = test["Absolute_Error"].mean()
    rmse = sqrt(((test[target] - test["Predicted"]) ** 2).mean())

    all_results.append({
        "Target": target,
        "Test_Rows": len(test),
        "MAE": mae,
        "RMSE": rmse,
    })

    print(f"\n{target}")
    print(f"Test rows: {len(test):,}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    # Keep timestamps, actual values and predictions for inspection.
    test[[
        "Timestamp", target, "Predicted", "Absolute_Error"
    ]].to_csv(
        f"validation_{target}.csv",
        index=False
    )

summary = pd.DataFrame(all_results)
summary.to_csv(OUTPUT_FILE, index=False)

print("\n=== VALIDATION SUMMARY ===")
print(summary.to_string(index=False))
print(f"\nSaved summary to {OUTPUT_FILE}")

