from fastapi import FastAPI

from housing.api import (
    FullInferenceRequest,
    SimpleInferenceRequest,
    InferenceResponse,
    FullInferenceWrapper,
    SimpleInferenceWrapper,
    HealthCheckResponse,
)

app = FastAPI()

FULL_INFERER = FullInferenceWrapper()
SIMPLE_INFERER = SimpleInferenceWrapper()


@app.get("/health")
def health() -> HealthCheckResponse:

    status = (
        "healthy"
        if (
            FULL_INFERER is not None
            and FULL_INFERER._health_check()
            and SIMPLE_INFERER is not None
            and SIMPLE_INFERER._health_check()
        )
        else "unhealthy"
    )
    return HealthCheckResponse(status=status)


@app.post("/inference/full")
def inference(request: FullInferenceRequest) -> InferenceResponse:
    return FULL_INFERER.inference(request)


@app.post("/inference/simple")
def simple_inference(request: SimpleInferenceRequest) -> InferenceResponse:
    return SIMPLE_INFERER.inference(request)
