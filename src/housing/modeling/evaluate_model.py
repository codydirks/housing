import pickle
import argparse
from pathlib import Path

from sklearn import model_selection
from sklearn.pipeline import Pipeline

from housing.modeling.create_model import load_data, SALES_COLUMN_SELECTION
from housing.config import SALES_PATH, DEMOGRAPHICS_PATH, DEV_MODEL_PATH, PRODUCTION_MODEL_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the housing model.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["dev", "prod"],
        help="Specify which model to evaluate: 'dev' for development model, 'prod' for production model.",
    )
    return parser.parse_args()


def load_model(model_path: Path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def main():
    args = parse_args()
    """Load data, train model, and export artifacts."""
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
    x_train, _x_test, y_train, _y_test = model_selection.train_test_split(x, y, random_state=42)
    model_path = DEV_MODEL_PATH if args.model == "dev" else PRODUCTION_MODEL_PATH
    model: Pipeline = load_model(model_path)
    train_score = model.score(x_train, y_train)  # For regressors, score corresponds to R^2
    test_score = model.score(_x_test, _y_test)
    print(f"Train score: {train_score}")
    print(f"Test score: {test_score}")


if __name__ == "__main__":
    main()
