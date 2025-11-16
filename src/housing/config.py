import pathlib

PROJECT_DIR = pathlib.Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"
SALES_PATH = DATA_DIR / "kc_house_data.csv"
DEMOGRAPHICS_PATH = DATA_DIR / "zipcode_demographics.csv"

MODEL_DIR = PROJECT_DIR / "model"
PRODUCTION_MODEL_PATH = MODEL_DIR / "production_model.pkl"
DEV_MODEL_PATH = MODEL_DIR / "dev_model.pkl"

INFERENCE_COLUMNS = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "sqft_above",
    "sqft_basement",
    "zipcode",
]
