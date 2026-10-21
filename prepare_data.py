
import pandas as pd

INPUT_FILE = "Toronto Island Ferry Tickets.csv"
OUTPUT_FILE = "ferry_prepared.csv"

# Load the original dataset
df = pd.read_csv(INPUT_FILE)

# Convert timestamps
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="raise")

# Sort chronologically
df = df.sort_values("Timestamp").reset_index(drop=True)

# Add calendar features
df["Hour"] = df["Timestamp"].dt.hour
df["Minute"] = df["Timestamp"].dt.minute
df["Day_of_Week"] = df["Timestamp"].dt.dayofweek
df["Month"] = df["Timestamp"].dt.month
df["Is_Weekend"] = (df["Day_of_Week"] >= 5).astype(int)

# Keep only relevant columns for forecasting
df = df[
    [
        "Timestamp",
        "Sales Count",
        "Redemption Count",
        "Hour",
        "Minute",
        "Day_of_Week",
        "Month",
        "Is_Weekend",
    ]
]

# Save a separate prepared dataset
df.to_csv(OUTPUT_FILE, index=False)

print("Prepared dataset saved:", OUTPUT_FILE)
print("Rows:", len(df))
print("Columns:", list(df.columns))
print("\nFirst 5 chronological records:")
print(df.head().to_string(index=False))
print("\nLast 5 chronological records:")
print(df.tail().to_string(index=False))
