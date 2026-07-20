"""配置信息路由."""

from fastapi import APIRouter, Depends

from src.api.dependencies import verify_api_key
from src.api.schemas import ServiceInfo
from src.config import get_settings
from src.metrics import get_metrics


router = APIRouter(prefix="/api/v1", tags=["config"])


@router.get("/info", response_model=ServiceInfo)
async def info(_=Depends(verify_api_key)) -> ServiceInfo:
    """获取服务信息."""
    settings = get_settings()
    return ServiceInfo(
        name="railway-asr-correction",
        version="0.1.0",
        llm_model=settings.llm_model,
        prompt_version=settings.llm_prompt_version,
        capabilities=[
            "layer1_itn",
            "layer2_dictionary",
            "layer3_context",
            "layer4_semantic",
            "layer4_rag",
            "layer4_harness",
            "hotword_management",
            "prompt_version_management",
        ],
    )


@router.get("/config")
async def config(_=Depends(verify_api_key)) -> dict:
    """获取当前配置快照（脱敏）."""
    settings = get_settings()
    return {
        "api_host": settings.api_host,
        "api_port": settings.api_port,
        "llm_model": settings.llm_model,
        "llm_prompt_version": settings.llm_prompt_version,
        "llm_max_concurrency": settings.llm_max_concurrency,
        "dify_enabled": settings.dify_enabled,
        "dify_base_url": settings.dify_base_url,
        "enable_entity_guard": settings.enable_entity_guard,
        "enable_cache": settings.enable_cache,
        "cache_size": settings.cache_size,
    }


@router.get("/metrics")
async def metrics(_=Depends(verify_api_key)) -> dict:
    """获取服务指标."""
    return get_metrics().snapshot()
