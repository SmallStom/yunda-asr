"""Dify 同步 API 路由."""

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.dependencies import verify_api_key
from src.config import get_settings
from src.dify_client import DifyClient, DifyClientError
from src.hotword_manager import get_hotword_manager
from src.logging_config import get_logger
from src.prompt_manager import get_prompt_manager


router = APIRouter(prefix="/api/v1/dify-sync", tags=["dify-sync"])
logger = get_logger(__name__)


class SyncRequest(BaseModel):
    """Dify 同步请求体."""

    dataset_id: Optional[str] = None
    version: Optional[str] = None


def _get_dataset_id(setting_attr: str, request_dataset_id: Optional[str]) -> str:
    """获取数据集 ID：请求参数优先，其次环境变量配置."""
    if request_dataset_id:
        return request_dataset_id
    settings = get_settings()
    dataset_id = getattr(settings, setting_attr, None)
    if not dataset_id:
        raise HTTPException(
            status_code=400,
            detail=f"{setting_attr} is not configured in .env and not provided in request",
        )
    return dataset_id


def _check_dify_enabled() -> None:
    """检查 Dify 是否启用."""
    settings = get_settings()
    if not settings.dify_enabled:
        raise HTTPException(status_code=403, detail="Dify integration is disabled")


@router.post("/hotwords/pull")
async def sync_hotwords_from_dify(
    request: SyncRequest = SyncRequest(),
    _=Depends(verify_api_key),
) -> dict:
    """从 Dify 拉取热词并同步到本地.

    请求体（JSON）：
    ```json
    {"dataset_id": "xxx", "version": "调度"}
    ```

    - dataset_id: 可选，覆盖 .env 中的 DIFY_HOTWORDS_DATASET_ID
    - version: 可选，若指定则保存为版本文件 hotwords_{version}.json
    """
    _check_dify_enabled()
    ds_id = _get_dataset_id("dify_hotwords_dataset_id", request.dataset_id)
    try:
        client = DifyClient()
        words = client.fetch_hotwords(ds_id, version=request.version)
        manager = get_hotword_manager()
        result = manager.reload_from_dify(words, version=request.version)
        client.close()
        return {"status": "ok", "dataset_id": ds_id, "version": request.version, **result}
    except DifyClientError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("failed to sync hotwords from Dify")
        raise HTTPException(status_code=500, detail=f"sync failed: {e}")


@router.post("/prompts/pull")
async def sync_prompts_from_dify(
    request: SyncRequest = SyncRequest(),
    _=Depends(verify_api_key),
) -> dict:
    """从 Dify 拉取 Prompt 并同步到本地.

    请求体（JSON）：
    ```json
    {"dataset_id": "xxx", "version": "调度"}
    ```
    """
    _check_dify_enabled()
    ds_id = _get_dataset_id("dify_prompts_dataset_id", request.dataset_id)
    try:
        client = DifyClient()
        prompts = client.fetch_prompts(ds_id, version=request.version)
        manager = get_prompt_manager()
        updated = manager.reload_from_dify(prompts)
        client.close()
        return {"status": "ok", "dataset_id": ds_id, "version": request.version, "updated": updated}
    except DifyClientError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("failed to sync prompts from Dify")
        raise HTTPException(status_code=500, detail=f"sync failed: {e}")


@router.post("/aliases/pull")
async def sync_aliases_from_dify(
    request: SyncRequest = SyncRequest(),
    _=Depends(verify_api_key),
) -> dict:
    """从 Dify 拉取正别名映射并同步到本地.

    请求体（JSON）：
    ```json
    {"dataset_id": "xxx", "version": "调度"}
    ```
    """
    _check_dify_enabled()
    ds_id = _get_dataset_id("dify_aliases_dataset_id", request.dataset_id)

    try:
        client = DifyClient()
        aliases = client.fetch_aliases(ds_id, version=request.version)
        client.close()

        from src.alias_manager import get_alias_manager

        manager = get_alias_manager()
        if request.version:
            path = manager.save_as_version(request.version, aliases)
        else:
            path = manager.save_as_active(aliases)

        return {
            "status": "ok",
            "dataset_id": ds_id,
            "version": request.version,
            "count": len(aliases),
            "path": str(path),
        }
    except DifyClientError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("failed to sync aliases from Dify")
        raise HTTPException(status_code=500, detail=f"sync failed: {e}")


@router.post("/knowledge/pull")
async def sync_knowledge_from_dify(
    dataset_id: Optional[str] = None,
    _=Depends(verify_api_key),
) -> dict:
    """从 Dify 拉取领域知识/错误模式."""
    settings = get_settings()
    if not settings.dify_enabled:
        raise HTTPException(status_code=403, detail="Dify integration is disabled")

    ds_id = _get_dataset_id("dify_knowledge_dataset_id", dataset_id)
    try:
        client = DifyClient()
        docs = client.fetch_knowledge(ds_id)
        # TODO: 将 docs 解析为 asr_error_pairs / aliases 并持久化
        # 当前仅返回文档数与内容预览
        client.close()
        return {
            "status": "ok",
            "dataset_id": ds_id,
            "documents": len(docs),
            "preview": [doc["name"] for doc in docs],
        }
    except DifyClientError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("failed to sync knowledge from Dify")
        raise HTTPException(status_code=500, detail=f"sync failed: {e}")


@router.get("/status")
async def dify_sync_status(_=Depends(verify_api_key)) -> dict:
    """获取 Dify 同步配置状态."""
    settings = get_settings()
    return {
        "dify_enabled": settings.dify_enabled,
        "dify_base_url": settings.dify_base_url,
        "hotwords_dataset_id": settings.dify_hotwords_dataset_id,
        "prompts_dataset_id": settings.dify_prompts_dataset_id,
        "aliases_dataset_id": settings.dify_aliases_dataset_id,
        "knowledge_dataset_id": settings.dify_knowledge_dataset_id,
        "sync_interval_seconds": settings.dify_sync_interval_seconds,
    }
