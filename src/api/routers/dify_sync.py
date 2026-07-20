"""Dify 同步 API 路由."""

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import verify_api_key
from src.config import get_settings
from src.dify_client import DifyClient, DifyClientError
from src.hotword_manager import get_hotword_manager
from src.logging_config import get_logger
from src.prompt_manager import get_prompt_manager


router = APIRouter(prefix="/api/v1/dify-sync", tags=["dify-sync"])
logger = get_logger(__name__)


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


@router.post("/hotwords/pull")
async def sync_hotwords_from_dify(
    dataset_id: Optional[str] = None,
    _=Depends(verify_api_key),
) -> dict:
    """从 Dify 拉取热词并同步到本地."""
    settings = get_settings()
    if not settings.dify_enabled:
        raise HTTPException(status_code=403, detail="Dify integration is disabled")

    ds_id = _get_dataset_id("dify_hotwords_dataset_id", dataset_id)
    try:
        client = DifyClient()
        words = client.fetch_hotwords(ds_id)
        manager = get_hotword_manager()
        result = manager.reload_from_dify(words)
        client.close()
        return {"status": "ok", "dataset_id": ds_id, **result}
    except DifyClientError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("failed to sync hotwords from Dify")
        raise HTTPException(status_code=500, detail=f"sync failed: {e}")


@router.post("/prompts/pull")
async def sync_prompts_from_dify(
    dataset_id: Optional[str] = None,
    _=Depends(verify_api_key),
) -> dict:
    """从 Dify 拉取 Prompt 并同步到本地."""
    settings = get_settings()
    if not settings.dify_enabled:
        raise HTTPException(status_code=403, detail="Dify integration is disabled")

    ds_id = _get_dataset_id("dify_prompts_dataset_id", dataset_id)
    try:
        client = DifyClient()
        prompts = client.fetch_prompts(ds_id)
        manager = get_prompt_manager()
        updated = manager.reload_from_dify(prompts)
        client.close()
        return {"status": "ok", "dataset_id": ds_id, "updated": updated}
    except DifyClientError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("failed to sync prompts from Dify")
        raise HTTPException(status_code=500, detail=f"sync failed: {e}")


@router.post("/aliases/pull")
async def sync_aliases_from_dify(
    dataset_id: Optional[str] = None,
    _=Depends(verify_api_key),
) -> dict:
    """从 Dify 拉取正别名映射并同步到本地."""
    settings = get_settings()
    if not settings.dify_enabled:
        raise HTTPException(status_code=403, detail="Dify integration is disabled")

    ds_id = _get_dataset_id("dify_aliases_dataset_id", dataset_id)
    alias_file = settings.lexicon_dir / "aliases.json"

    try:
        client = DifyClient()
        aliases = client.fetch_aliases(ds_id)
        client.close()

        # 备份原文件
        if alias_file.exists():
            backup_dir = settings.lexicon_dir / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / f"aliases.json.bak"
            try:
                backup_path.write_text(
                    alias_file.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
            except Exception:
                pass

        # 持久化新别名
        alias_file.parent.mkdir(parents=True, exist_ok=True)
        with open(alias_file, "w", encoding="utf-8") as f:
            json.dump(aliases, f, ensure_ascii=False, indent=2)

        # 运行时热重载
        from src import dictionary_corrector, phonetic_candidate

        phonetic_candidate.reload_aliases()
        dictionary_corrector.reload_aliases()

        # 刷新 API 流水线中的 RAG/Harness TermTool 索引
        try:
            from src.api.dependencies import get_pipeline

            get_pipeline().reload_aliases()
        except Exception:
            pass

        return {
            "status": "ok",
            "dataset_id": ds_id,
            "count": len(aliases),
            "path": str(alias_file),
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
