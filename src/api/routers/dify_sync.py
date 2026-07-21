"""Dify 同步 API 路由."""

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

    dataset_id: str
    version: Optional[str] = None


def _check_dify_enabled() -> None:
    """检查 Dify 是否启用."""
    settings = get_settings()
    if not settings.dify_enabled:
        raise HTTPException(status_code=403, detail="Dify integration is disabled")


@router.post("/hotwords/pull")
async def sync_hotwords_from_dify(
    request: SyncRequest,
    _=Depends(verify_api_key),
) -> dict:
    """从 Dify 拉取热词并同步到本地.

    请求体（JSON）：
    ```json
    {"dataset_id": "xxx", "version": "调度"}
    ```

    - dataset_id: 必填，Dify 热词知识库 ID
    - version: 可选，若指定则保存为版本文件 hotwords_{version}.json
    """
    _check_dify_enabled()
    client = DifyClient()
    try:
        words = client.fetch_hotwords(request.dataset_id, version=request.version)
        manager = get_hotword_manager()
        result = manager.reload_from_dify(words, version=request.version)
        return {"status": "ok", "dataset_id": request.dataset_id, "version": request.version, **result}
    except DifyClientError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("failed to sync hotwords from Dify")
        raise HTTPException(status_code=500, detail=f"sync failed: {e}")
    finally:
        client.close()


@router.post("/prompts/pull")
async def sync_prompts_from_dify(
    request: SyncRequest,
    _=Depends(verify_api_key),
) -> dict:
    """从 Dify 拉取 Prompt 并同步到本地.

    请求体（JSON）：
    ```json
    {"dataset_id": "xxx", "version": "调度"}
    ```
    """
    _check_dify_enabled()
    client = DifyClient()
    try:
        prompts = client.fetch_prompts(request.dataset_id, version=request.version)
        manager = get_prompt_manager()
        updated = manager.reload_from_dify(prompts)
        # 触发 pipeline 热重载（刷新 RAG/Harness 的 system prompt）
        try:
            from src.api.dependencies import get_pipeline
            get_pipeline().reload_prompts()
        except Exception:
            pass
        return {"status": "ok", "dataset_id": request.dataset_id, "version": request.version, "count": len(updated), "files": updated}
    except DifyClientError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("failed to sync prompts from Dify")
        raise HTTPException(status_code=500, detail=f"sync failed: {e}")
    finally:
        client.close()


@router.post("/aliases/pull")
async def sync_aliases_from_dify(
    request: SyncRequest,
    _=Depends(verify_api_key),
) -> dict:
    """从 Dify 拉取正别名映射并同步到本地.

    请求体（JSON）：
    ```json
    {"dataset_id": "xxx", "version": "调度"}
    ```
    """
    _check_dify_enabled()
    client = DifyClient()
    try:
        aliases = client.fetch_aliases(request.dataset_id, version=request.version)

        from src.alias_manager import get_alias_manager

        manager = get_alias_manager()
        if request.version:
            path, deleted = manager.save_as_version(request.version, aliases)
        else:
            path, deleted = manager.save_as_active(aliases)

        return {
            "status": "ok",
            "dataset_id": request.dataset_id,
            "version": request.version,
            "count": len(aliases),
            "deleted": deleted,
            "path": str(path),
        }
    except DifyClientError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("failed to sync aliases from Dify")
        raise HTTPException(status_code=500, detail=f"sync failed: {e}")
    finally:
        client.close()


@router.get("/status")
async def dify_sync_status(_=Depends(verify_api_key)) -> dict:
    """获取 Dify 同步配置状态."""
    settings = get_settings()
    return {
        "dify_enabled": settings.dify_enabled,
        "dify_base_url": settings.dify_base_url,
    }
