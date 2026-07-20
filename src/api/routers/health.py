"""健康检查路由."""

from fastapi import APIRouter, Depends

from src.api.dependencies import get_pipeline, verify_api_key
from src.api.schemas import HealthResponse, ReadyResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """健康检查."""
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse)
async def ready(
    pipeline=Depends(get_pipeline),
    _=Depends(verify_api_key),
) -> ReadyResponse:
    """就绪检查，验证依赖是否可用."""
    checks = {
        "pipeline_loaded": pipeline is not None,
        "pipeline_components": all([
            pipeline.preprocessor is not None,
            pipeline.dictionary_corrector is not None,
            pipeline.context_corrector is not None,
        ]),
    }
    status = "ok" if all(checks.values()) else "not_ready"
    return ReadyResponse(status=status, checks=checks)
