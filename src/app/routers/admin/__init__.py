"""The `/admin` API surface.

This package owns the `/admin` prefix; each sub-router below declares only its
own segment (or none, for the top-level metadata endpoints). Adding a new admin
area means dropping a module here and including its router.
"""

from fastapi import APIRouter

from app.routers.admin import evals, meta, presets, rag

router = APIRouter(prefix="/admin", tags=["Admin"])

router.include_router(meta.router)
router.include_router(rag.router)
router.include_router(presets.router)
router.include_router(evals.router)

__all__ = ["router"]
