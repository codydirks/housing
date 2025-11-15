from fastapi import FastAPI

from housing.api import InferenceRequest, InferenceResponse, InferenceWrapper, HealthCheckResponse

app = FastAPI()

INFERER = InferenceWrapper()

@app.get("/health")
def health() -> HealthCheckResponse:

    status = (
        "healthy"
        if INFERER is not None and INFERER._health_check()
        else "unhealthy"
    )
    return HealthCheckResponse(status=status)

@app.post('/model/inference')
def inference(request: InferenceRequest) -> InferenceResponse:
    return INFERER.inference(request)