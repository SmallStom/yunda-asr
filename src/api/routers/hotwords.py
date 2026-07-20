"""热词管理 API 路由."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies import verify_api_key
from src.api.schemas import (
    HotwordCreateRequest,
    HotwordItem,
    HotwordsResponse,
    HotwordUpdateRequest,
)
from src.hotword_manager import get_hotword_manager
from src.logging_config import get_logger


router = APIRouter(prefix="/api/v1", tags=["hotwords"])
logger = get_logger(__name__)


def _item_to_schema(item) -> HotwordItem:
    return HotwordItem(
        id=item.id,
        word=item.word,
        category=item.category,
        enabled=item.enabled,
    )


@router.get("/hotwords", response_model=HotwordsResponse)
async def list_hotwords(
    category: Optional[str] = Query(None),
    enabled_only: bool = Query(True),
    _=Depends(verify_api_key),
) -> HotwordsResponse:
    """列出热词."""
    manager = get_hotword_manager()
    items = manager.list_all(category=category, enabled_only=enabled_only)
    return HotwordsResponse(
        total=len(items),
        items=[_item_to_schema(item) for item in items],
    )


@router.post("/hotwords", response_model=HotwordItem, status_code=201)
async def create_hotword(
    req: HotwordCreateRequest,
    _=Depends(verify_api_key),
) -> HotwordItem:
    """创建热词."""
    manager = get_hotword_manager()
    try:
        item = manager.create(
            word=req.word,
            category=req.category,
            enabled=req.enabled,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return _item_to_schema(item)


@router.get("/hotwords/{hotword_id}", response_model=HotwordItem)
async def get_hotword(
    hotword_id: str,
    _=Depends(verify_api_key),
) -> HotwordItem:
    """获取单个热词."""
    manager = get_hotword_manager()
    item = manager.get(hotword_id)
    if not item:
        raise HTTPException(status_code=404, detail="hotword not found")
    return _item_to_schema(item)


@router.put("/hotwords/{hotword_id}", response_model=HotwordItem)
async def update_hotword(
    hotword_id: str,
    req: HotwordUpdateRequest,
    _=Depends(verify_api_key),
) -> HotwordItem:
    """更新热词."""
    manager = get_hotword_manager()
    update_data = req.model_dump(exclude_unset=True)
    item = manager.update(hotword_id, **update_data)
    if not item:
        raise HTTPException(status_code=404, detail="hotword not found")
    return _item_to_schema(item)


@router.delete("/hotwords/{hotword_id}", status_code=204)
async def delete_hotword(
    hotword_id: str,
    _=Depends(verify_api_key),
) -> None:
    """删除热词."""
    manager = get_hotword_manager()
    if not manager.delete(hotword_id):
        raise HTTPException(status_code=404, detail="hotword not found")


@router.post("/hotwords/reload")
async def reload_hotwords(_=Depends(verify_api_key)) -> dict:
    """热重载热词."""
    manager = get_hotword_manager()
    manager.reload()
    return {"status": "ok", "total": len(manager.list_all(enabled_only=False))}


@router.post("/hotwords/push-to-asr")
async def push_hotwords_to_asr(_=Depends(verify_api_key)) -> dict:
    """将热词转换为上期 ASR 格式（具体推送逻辑需按 ASR 协议实现）."""
    manager = get_hotword_manager()
    payload = manager.to_asr_format()
    # TODO: 调用上游 ASR 热词接口
    logger.info("prepared hotwords for ASR push", extra={"count": len(payload["hotwords"])})
    return {"status": "prepared", "count": len(payload["hotwords"]), "payload": payload}
