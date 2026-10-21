
import pandas as pd

INPUT_FILE = "ferry_prepared.csv"
OUTPUT_FILE = "ferry_forecast_targets.csv"

df = pd.read_csv(INPUT_FILE, parse_dates=["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

# Create a fast lookup from timestamp to row position.
timestamp_to_position = pd.Series(
    range(len(df)), index=df["Timestamp"]
)

for minutes in [30, 60, 120]:
    steps = minutes // 15
    sales_sum = pd.Series(0.0, index=df.index)
    redemption_sum = pd.Series(0.0, index=df.index)
    valid = pd.Series(True, index=df.index)

    for step in range(1, steps + 1):
        future_times = df["Timestamp"] + pd.Timedelta(
            minutes=15 * step
        )

        positions = future_times.map(timestamp_to_position)
        exists = positions.notna()
        valid &= exists

        # Add counts only where the future timestamp exists.
        rows = positions[exists].astype(int)
        sales_sum.loc[exists] += (
            df.loc[rows, "Sales Count"].to_numpy()
        )
        redemption_sum.loc[exists] += (
            df.loc[rows, "Redemption Count"].to_numpy()
        )

    sales_sum.loc[~valid] = float("nan")
    redemption_sum.loc[~valid] = float("nan")

    df[f"Sales_Next_{minutes}m"] = sales_sum
    df[f"Redemptions_Next_{minutes}m"] = redemption_sum

df.to_csv(OUTPUT_FILE, index=False)

print("Saved:", OUTPUT_FILE)
print("Total rows:", len(df))

print("\n--- TARGET AVAILABILITY ---")
target_columns = [
    "Sales_Next_30m", "Redemptions_Next_30m",
    "Sales_Next_60m", "Redemptions_Next_60m",
    "Sales_Next_120m", "Redemptions_Next_120m",
]

for column in target_columns:
    count = df[column].notna().sum()
    print(
        f"{column}: {count:,} valid targets "
        f"({count / len(df) * 100:.2f}%)"
    )

print("\n--- EXAMPLE TARGETS ---")
print(
    df[["Timestamp", "Sales Count", "Redemption Count"]
       + target_columns]
    .head(12)
    .to_string(index=False)
)
