from fastapi import FastAPI
from pydantic import BaseModel

from housing.api import InferenceRequest, InferenceWrapper, InferenceResponse

app = FastAPI()

INFERER = InferenceWrapper()

@app.get("/health")
def health():

    status = (
        "healthy"
        if INFERER is not None and INFERER._health_check()
        else "unhealthy"
    )
    return {"status": status}

@app.post('/model/inference')
def inference(request: InferenceRequest):
    return INFERER.inference(request)