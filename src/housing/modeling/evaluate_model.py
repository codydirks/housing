import pickle
from sklearn import model_selection
from sklearn.pipeline import Pipeline

from housing.modeling.create_model import load_data, SALES_COLUMN_SELECTION
from housing.config import SALES_PATH, DEMOGRAPHICS_PATH, MODEL_PATH

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model


def main():
    """Load data, train model, and export artifacts."""
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
    x_train, _x_test, y_train, _y_test = model_selection.train_test_split(
        x, y, random_state=42)
    
    model: Pipeline = load_model()
    train_score = model.score(x_train, y_train) # For KNeighborsRegressor, score corresponds to R^2
    test_score = model.score(_x_test, _y_test)
    print(f"Train score: {train_score}")
    print(f"Test score: {test_score}")


if __name__ == "__main__":
    main()