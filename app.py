
import os
import joblib
import pandas as pd
import streamlit as st
from download_assets import ensure_assets

DATA_FILE = "ferry_model_data.csv"
MODEL_DIR = "saved_models"
RESULT_FILE = "historical_validation.csv"

st.set_page_config(
    page_title="Ferry Demand Forecasting",
    page_icon="⛴",
    layout="wide",
)

st.title("Ferry Ticket Demand Forecasting")
st.caption("Short-Term Predictive Decision Support System")

st.warning(
    "Historical-data demonstration only. The dataset ends on "
    "December 21, 2025. These are not live forecasts for today."
)

calendar_features = [
    "Hour",
    "Minute",
    "Day_of_Week",
    "Month",
    "Is_Weekend",
]

lag_features = [
    "Sales_Lag_15m",
    "Redemptions_Lag_15m",
    "Sales_Lag_30m",
    "Redemptions_Lag_30m",
    "Sales_Lag_60m",
    "Redemptions_Lag_60m",
    "Sales_Lag_120m",
    "Redemptions_Lag_120m",
]

features = calendar_features + lag_features


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_FILE, parse_dates=["Timestamp"])
    return data.sort_values("Timestamp").reset_index(drop=True)


@st.cache_data
def load_metrics():
    if os.path.exists(RESULT_FILE):
        return pd.read_csv(RESULT_FILE)
    return pd.DataFrame()


@st.cache_resource
def load_model(target):
    path = os.path.join(MODEL_DIR, f"{target}.joblib")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path}"
        )

    saved_model = joblib.load(path)

    if "model" not in saved_model or "features" not in saved_model:
        raise ValueError(
            f"Model file has an unexpected format: {path}"
        )

    return saved_model


def classify_sales(value):
    if value >= 332:
        return (
            "High historical demand band",
            "Verify recent ticket activity and assess operational readiness.",
        )

    elif value >= 126:
        return (
            "Above-typical historical demand",
            "Monitor ticket activity and compare with recent demand.",
        )

    return (
        "Typical-to-lower historical demand band",
        "Continue monitoring. Low predicted demand does not guarantee spare capacity.",
    )


# --------------------------------------------------
# 1. LOAD AND VALIDATE DATA
# --------------------------------------------------

try:
    with st.spinner("Preparing dataset and trained models..."):
        ensure_assets()
except Exception as exc:
    st.error(f"Could not download required assets: {exc}")
    st.stop()
    
if not os.path.exists(DATA_FILE):
    st.error(f"Dataset not found: {DATA_FILE}")
    st.stop()

try:
    df = load_data()

except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

required_columns = [
    "Timestamp",
    "Sales Count",
    "Redemption Count",
] + features

missing = [
    column for column in required_columns
    if column not in df.columns
]

if missing:
    st.error(f"Required columns are missing: {missing}")
    st.stop()

valid_rows = df.dropna(subset=features)

if valid_rows.empty:
    st.error(
        "No historical row has all required lag features available."
    )
    st.stop()

latest = valid_rows.iloc[-1]
X = latest[features].to_frame().T


# --------------------------------------------------
# 2. FORECAST REFERENCE
# --------------------------------------------------

st.subheader("Forecast Reference")

st.write(f"**Historical timestamp:** {latest['Timestamp']}")

st.write(
    "Forecasts represent total predicted counts over the following "
    "30, 60, or 120 minutes."
)


# --------------------------------------------------
# 3. GENERATE FORECASTS
# --------------------------------------------------

forecast_results = {}

try:
    for horizon in ["30m", "60m", "120m"]:
        row = {}

        for demand in ["Sales", "Redemptions"]:
            target = f"{demand}_Next_{horizon}"
            saved = load_model(target)

            model_features = saved["features"]

            missing_model_features = [
                feature for feature in model_features
                if feature not in X.columns
            ]

            if missing_model_features:
                raise ValueError(
                    f"Missing features for {target}: "
                    f"{missing_model_features}"
                )

            prediction = saved["model"].predict(
                X[model_features]
            )[0]

            row[demand] = max(
                0,
                round(float(prediction)),
            )

        forecast_results[horizon] = row

except Exception as exc:
    st.error(f"Could not load or run a forecast model: {exc}")
    st.stop()


# --------------------------------------------------
# 4. DISPLAY FORECAST CARDS
# --------------------------------------------------

st.subheader("Demand Forecasts")

cols = st.columns(3)

for col, horizon in zip(cols, ["30m", "60m", "120m"]):
    result = forecast_results[horizon]

    with col:
        st.metric(
            f"Next {horizon} — Sales",
            f"{result['Sales']:,}",
        )

        st.metric(
            f"Next {horizon} — Redemptions",
            f"{result['Redemptions']:,}",
        )


# --------------------------------------------------
# 5. DECISION SUPPORT
# --------------------------------------------------

st.subheader("30-Minute Decision Support")

sales_30 = forecast_results["30m"]["Sales"]

category, recommendation = classify_sales(sales_30)

st.write(f"**Demand category:** {category}")
st.info(recommendation)

st.caption(
    "Categories use historical sales-demand percentiles: 126 is the "
    "75th percentile and 332 is the 90th percentile. These are not "
    "verified ferry capacity or staffing thresholds. Extreme demand "
    "spikes may be underpredicted by the model."
)


# --------------------------------------------------
# 6. MODEL EVALUATION
# --------------------------------------------------

st.subheader("Model Evaluation")

metrics = load_metrics()

if not metrics.empty:
    st.dataframe(
        metrics,
        hide_index=True,
        width="stretch",
    )

else:
    st.info(
        "Evaluation metrics are unavailable. Run "
        "python validate_predictions.py first."
    )


# --------------------------------------------------
# 7. RECENT HISTORICAL RECORDS
# --------------------------------------------------

st.subheader("Recent Historical Records")

recent_records = (
    df[
        ["Timestamp", "Sales Count", "Redemption Count"]
    ]
    .tail(10)
    .sort_values("Timestamp", ascending=False)
)

st.dataframe(
    recent_records,
    hide_index=True,
    width="stretch",
)


# --------------------------------------------------
# 8. FOOTER
# --------------------------------------------------

st.caption(
    "Prototype for decision support only. Validate forecasts against "
    "current ticket activity and actual operational constraints before use."
)
