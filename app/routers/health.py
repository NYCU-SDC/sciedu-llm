from fastapi import APIRouter

from app.schema.health import HealthzResponse

router = APIRouter(tags=["Healthz"])


@router.get("/healthz", response_model=HealthzResponse)
async def healthz():
    """
    Indicates whether this LLM service is ready
    """
    return {"status": "ok"}
