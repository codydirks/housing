import pathlib

PROJECT_DIR = pathlib.Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"
SALES_PATH = DATA_DIR / "kc_house_data.csv"
DEMOGRAPHICS_PATH = DATA_DIR / "zipcode_demographics.csv"

MODEL_DIR = PROJECT_DIR / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"

INFERENCE_COLUMNS = [
    'bedrooms',
    'bathrooms',
    'sqft_living',
    'sqft_lot',
    'floors',
    'sqft_above',
    'sqft_basement',
    'zipcode'
]