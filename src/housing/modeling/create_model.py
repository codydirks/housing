import json
from pathlib import Path
import pickle
from typing import List
from typing import Tuple

import pandas
from sklearn import model_selection
from sklearn import ensemble
from sklearn import pipeline
from sklearn import preprocessing

from housing.config import MODEL_DIR, SALES_PATH, DEMOGRAPHICS_PATH, INFERENCE_COLUMNS


# List of columns (subset) that will be taken from home sale data
SALES_COLUMN_SELECTION = ["price"] + INFERENCE_COLUMNS


def load_data(
    sales_path: Path, demographics_path: Path, sales_column_selection: List[str]
) -> Tuple[pandas.DataFrame, pandas.Series]:
    """Load the target and feature data by merging sales and demographics.

    Args:
        sales_path: path to CSV file with home sale data
        demographics_path: path to CSV file with home sale data
        sales_column_selection: list of columns from sales data to be used as
            features

    Returns:
        Tuple containg with two elements: a DataFrame and a Series of the same
        length.  The DataFrame contains features for machine learning, the
        series contains the target variable (home sale price).

    """
    data = pandas.read_csv(sales_path, usecols=sales_column_selection, dtype={"zipcode": str})
    demographics = pandas.read_csv(demographics_path, dtype={"zipcode": str})

    merged_data = data.merge(demographics, how="left", on="zipcode").drop(columns="zipcode")
    # Remove the target variable from the dataframe, features will remain
    y = merged_data.pop("price")
    x = merged_data

    return x, y


def main():
    """Load data, train model, and export artifacts."""
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
    x_train, _x_test, y_train, _y_test = model_selection.train_test_split(x, y, random_state=42)

    model = pipeline.make_pipeline(
        preprocessing.RobustScaler(), ensemble.RandomForestRegressor(n_estimators=100, random_state=42)
    ).fit(x_train, y_train)

    output_dir = MODEL_DIR
    output_dir.mkdir(exist_ok=True)

    # Output model artifacts: pickled model and JSON list of features
    pickle.dump(model, open(output_dir / "dev_model.pkl", "wb"))
    json.dump(list(x_train.columns), open(output_dir / "dev_model_features.json", "w"))


if __name__ == "__main__":
    main()
