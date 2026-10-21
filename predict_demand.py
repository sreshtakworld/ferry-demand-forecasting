
import os
import joblib
import pandas as pd

DATA_FILE = "ferry_model_data.csv"
MODEL_DIR = "saved_models"

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

def classify_sales(value):
    if value >= 332:
        return (
            "High historical demand band",
            "Verify recent ticket activity and check whether extra "
            "operational readiness is needed."
        )
    elif value >= 126:
        return (
            "Above-typical historical demand",
            "Monitor ticket activity and compare with recent demand."
        )
    else:
        return (
            "Typical-to-lower historical demand band",
            "Continue monitoring; do not assume low demand guarantees spare capacity."
        )

df = pd.read_csv(DATA_FILE, parse_dates=["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

valid_rows = df.dropna(subset=features)

if valid_rows.empty:
    raise ValueError("No row has all required features available.")

latest = valid_rows.iloc[-1]
X = latest[features].to_frame().T

print("\n=== FERRY DEMAND FORECAST & DECISION SUPPORT ===")
print(f"Historical reference: {latest['Timestamp']}")
print("WARNING: Historical demonstration only; data ends in December 2025.")
print("Demand bands are statistical, not verified ferry-capacity limits.")

for horizon in ["30m", "60m", "120m"]:
    print(f"\n--- Forecast: next {horizon} ---")

    sales_target = f"Sales_Next_{horizon}"
    sales_path = os.path.join(MODEL_DIR, f"{sales_target}.joblib")

    saved_sales = joblib.load(sales_path)
    sales_model = saved_sales["model"]
    sales_prediction = max(
        0, round(float(sales_model.predict(X[saved_sales["features"]])[0]))
    )

    redemption_target = f"Redemptions_Next_{horizon}"
    redemption_path = os.path.join(MODEL_DIR, f"{redemption_target}.joblib")

    saved_redemptions = joblib.load(redemption_path)
    redemption_model = saved_redemptions["model"]
    redemption_prediction = max(
        0,
        round(float(
            redemption_model.predict(X[saved_redemptions["features"]])[0]
        ))
    )

    print(f"Predicted sales count: {sales_prediction:,}")
    print(f"Predicted redemption count: {redemption_prediction:,}")

    if horizon == "30m":
        category, recommendation = classify_sales(sales_prediction)
        print(f"Demand category: {category}")
        print(f"Recommendation: {recommendation}")
