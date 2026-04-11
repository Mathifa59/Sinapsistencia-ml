from fastapi import APIRouter
from app.config import settings

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    return {"status": "ok", "service": settings.app_name, "version": settings.app_version}
