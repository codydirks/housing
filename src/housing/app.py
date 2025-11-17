from fastapi import FastAPI

from housing.api import (
    FullInferenceRequest,
    SimpleInferenceRequest,
    InferenceResponse,
    ProductionInferenceWrapper,
    DevInferenceWrapper,
    HealthCheckResponse,
)

app = FastAPI()

FULL_INFERER = ProductionInferenceWrapper()

# Since this wrapper explicitly selects the necessary columns for inference,
# there's functionally no difference between "full" and "simple" inference here.
# However, we keep them separate to allow for future modifications.
SIMPLE_INFERER = ProductionInferenceWrapper()

DEV_INFERER = DevInferenceWrapper()


@app.get("/health")
def health() -> HealthCheckResponse:
    inferers = [FULL_INFERER, SIMPLE_INFERER, DEV_INFERER]
    status = "healthy" if all(inferer is not None and inferer._health_check() for inferer in inferers) else "unhealthy"
    return HealthCheckResponse(status=status)


@app.post("/inference/production/full")
def inference(request: FullInferenceRequest) -> InferenceResponse:
    return FULL_INFERER.inference(request)


@app.post("/inference/production/simple")
def simple_inference(request: SimpleInferenceRequest) -> InferenceResponse:
    return SIMPLE_INFERER.inference(request)


@app.post("/inference/dev")
def dev_inference(request: FullInferenceRequest) -> InferenceResponse:
    return DEV_INFERER.inference(request)
