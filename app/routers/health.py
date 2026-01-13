from fastapi import APIRouter

from app.schema.health import HealthzRespoonse

router = APIRouter(tags=["Healthz"])


@router.get("/healthz", response_model=HealthzRespoonse)
async def healthz():
    """
    Indicates whether this LLM service is ready
    """
    return {"status": "ok"}
