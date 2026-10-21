
import pandas as pd

df = pd.read_csv("Toronto Island Ferry Tickets.csv")

count_columns = ["Redemption Count", "Sales Count"]

print("\n--- COUNT VALIDATION ---")

for column in count_columns:
    values = pd.to_numeric(df[column], errors="coerce")

    print(f"\n{column}")
    print("Invalid or missing values:", values.isna().sum())
    print("Negative values:", (values < 0).sum())
    print("Zero values:", (values == 0).sum())
    print("Maximum value:", values.max())
    print("Median:", values.median())
    print("95th percentile:", values.quantile(0.95))
    print("99th percentile:", values.quantile(0.99))

print("\n--- TIMESTAMP VALIDATION ---")
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
print("Invalid timestamps:", df["Timestamp"].isna().sum())
print("Duplicate timestamps:", df["Timestamp"].duplicated().sum())
print("Rows in original dataset:", len(df))
