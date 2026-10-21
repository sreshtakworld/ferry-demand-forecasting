
import os
import urllib.request
import zipfile

DATA_URL = (
    "https://github.com/sreshtakworld/ferry-demand-forecasting/"
    "releases/download/v1.0.0/ferry_model_data.csv"
)

MODELS_URL = (
    "https://github.com/sreshtakworld/ferry-demand-forecasting/"
    "releases/download/v1.0.0/ferry_saved_models.zip"
)

DATA_FILE = "ferry_model_data.csv"
ZIP_FILE = "ferry_saved_models.zip"
MODEL_DIR = "saved_models"


def download_file(url, destination):
    if os.path.exists(destination) and os.path.getsize(destination) > 0:
        return

    temporary_file = destination + ".part"

    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "FerryForecastApp"},
        )

        with urllib.request.urlopen(request, timeout=300) as response:
            with open(temporary_file, "wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)

        if os.path.getsize(temporary_file) == 0:
            raise RuntimeError("The downloaded file is empty.")

        os.replace(temporary_file, destination)

    finally:
        if os.path.exists(temporary_file):
            os.remove(temporary_file)


def ensure_assets():
    download_file(DATA_URL, DATA_FILE)

    if not os.path.isdir(MODEL_DIR) or not all(
        os.path.isfile(os.path.join(MODEL_DIR, f"{name}.joblib"))
        for name in [
            "Sales_Next_30m",
            "Sales_Next_60m",
            "Sales_Next_120m",
            "Redemptions_Next_30m",
            "Redemptions_Next_60m",
            "Redemptions_Next_120m",
        ]
    ):
        download_file(MODELS_URL, ZIP_FILE)
        os.makedirs(MODEL_DIR, exist_ok=True)

        with zipfile.ZipFile(ZIP_FILE, "r") as archive:
            for item in archive.infolist():
                if item.is_dir():
                    continue

                filename = os.path.basename(item.filename)

                if filename.endswith(".joblib"):
                    with archive.open(item) as source:
                        with open(
                            os.path.join(MODEL_DIR, filename), "wb"
                        ) as destination:
                            while True:
                                chunk = source.read(1024 * 1024)
                                if not chunk:
                                    break
                                destination.write(chunk)

        os.remove(ZIP_FILE)

    required_models = [
        "Sales_Next_30m.joblib",
        "Sales_Next_60m.joblib",
        "Sales_Next_120m.joblib",
        "Redemptions_Next_30m.joblib",
        "Redemptions_Next_60m.joblib",
        "Redemptions_Next_120m.joblib",
    ]

    missing = [
        name for name in required_models
        if not os.path.isfile(os.path.join(MODEL_DIR, name))
    ]

    if missing:
        raise RuntimeError(f"Missing model files: {missing}")
