
import pandas as pd

INPUT_FILE = "ferry_forecast_targets.csv"
OUTPUT_FILE = "ferry_model_data.csv"

df = pd.read_csv(INPUT_FILE, parse_dates=["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

# Map each timestamp to its row position for fast lookups.
timestamp_to_position = pd.Series(
    df.index.to_numpy(),
    index=df["Timestamp"]
)

# Add features using only observations from the past.
for minutes in [15, 30, 60, 120]:
    past_times = df["Timestamp"] - pd.Timedelta(minutes=minutes)
    positions = past_times.map(timestamp_to_position)

    for source_column, feature_name in [
        ("Sales Count", f"Sales_Lag_{minutes}m"),
        ("Redemption Count", f"Redemptions_Lag_{minutes}m"),
    ]:
        values = pd.Series(float("nan"), index=df.index)
        valid = positions.notna()

        values.loc[valid] = df.loc[
            positions[valid].astype(int), source_column
        ].to_numpy()

        df[feature_name] = values

df.to_csv(OUTPUT_FILE, index=False)

print("Saved:", OUTPUT_FILE)
print("Total rows:", len(df))

print("\n--- NEW LAG FEATURES ---")
lag_columns = [
    "Sales_Lag_15m", "Redemptions_Lag_15m",
    "Sales_Lag_30m", "Redemptions_Lag_30m",
    "Sales_Lag_60m", "Redemptions_Lag_60m",
    "Sales_Lag_120m", "Redemptions_Lag_120m",
]

for column in lag_columns:
    print(f"{column}: {df[column].notna().sum():,} valid values")

print("\n--- SAMPLE ---")
print(
    df[["Timestamp", "Sales Count", "Redemption Count"]
       + lag_columns]
    .head(10)
    .to_string(index=False)
)
