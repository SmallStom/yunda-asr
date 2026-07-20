"""正别名映射管理 API 路由."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from src.alias_manager import get_alias_manager
from src.api.dependencies import verify_api_key


router = APIRouter(prefix="/api/v1", tags=["aliases"])


@router.get("/aliases/versions")
async def list_alias_versions(_=Depends(verify_api_key)) -> dict:
    """列出所有别名版本."""
    manager = get_alias_manager()
    return {"versions": manager.list_versions()}


@router.post("/aliases/switch-version")
async def switch_alias_version(
    version: str = Query(..., description="目标版本名，如 v2"),
    _=Depends(verify_api_key),
) -> dict:
    """切换当前活跃别名版本."""
    manager = get_alias_manager()
    ok = manager.switch_version(version)
    if not ok:
        raise HTTPException(status_code=404, detail=f"version '{version}' not found")
    return {"status": "ok", "version": version}
