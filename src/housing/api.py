from typing import Optional, Union, List
from pydantic import BaseModel

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from housing.config import (
    PRODUCTION_INFERENCE_COLUMNS,
    DEV_INFERENCE_COLUMNS,
    PRODUCTION_MODEL_PATH,
    DEV_MODEL_PATH,
    DEMOGRAPHICS_PATH,
)


class FullInferenceRequest(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    zipcode: int
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int


class SimpleInferenceRequest(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    sqft_above: int
    sqft_basement: int
    zipcode: int


class HealthCheckResponse(BaseModel):
    status: str


class InferenceResponse(BaseModel):
    price: Optional[float] = None


class InferenceWrapper:
    model_path: Path
    inference_columns: List[str]

    def __init__(self):
        self.model = self.load_model()
        self.demographics = self.load_demographic_data()

    def _health_check(self) -> bool:
        return self.model is not None and self.demographics is not None

    def load_model(self) -> Pipeline:
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

        return self.model

    def load_demographic_data(self) -> pd.DataFrame:
        demographics = pd.read_csv(DEMOGRAPHICS_PATH, dtype={"zipcode": str})
        return demographics

    def form_input_from_request(self, input: Union[FullInferenceRequest, SimpleInferenceRequest]) -> pd.DataFrame:
        assert isinstance(input, FullInferenceRequest)
        data_dict = input.model_dump()
        sample = pd.DataFrame([data_dict])[self.inference_columns]
        return sample

    def inference(self, input_data: Union[FullInferenceRequest, SimpleInferenceRequest]) -> InferenceResponse:
        sample = self.form_input_from_request(input_data)
        sample["zipcode"] = sample["zipcode"].astype(str)
        sample = sample.merge(self.demographics, on="zipcode", how="left").drop(columns=["zipcode"])
        prediction = self.model.predict(sample)

        assert isinstance(prediction, np.ndarray) and prediction.shape == (1,) and isinstance(prediction[0], float)
        return InferenceResponse(price=prediction[0])


class ProductionInferenceWrapper(InferenceWrapper):
    model_path = PRODUCTION_MODEL_PATH
    inference_columns = PRODUCTION_INFERENCE_COLUMNS

    def form_input_from_request(self, input: Union[FullInferenceRequest, SimpleInferenceRequest]) -> pd.DataFrame:
        assert isinstance(input, FullInferenceRequest)
        data_dict = input.model_dump()
        sample = pd.DataFrame([data_dict])[self.inference_columns]
        return sample


class DevInferenceWrapper(ProductionInferenceWrapper):
    model_path = DEV_MODEL_PATH
    inference_columns = DEV_INFERENCE_COLUMNS
