"""Prompt 版本管理 API 路由."""

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import verify_api_key
from src.api.schemas import PromptContent, PromptUpdateRequest, PromptVersion
from src.config import get_settings
from src.logging_config import get_logger
from src.prompt_manager import get_prompt_manager


router = APIRouter(prefix="/api/v1", tags=["prompts"])
logger = get_logger(__name__)


@router.get("/prompts", response_model=list[PromptVersion])
async def list_prompts(_=Depends(verify_api_key)) -> list[PromptVersion]:
    """列出所有 Prompt 版本."""
    manager = get_prompt_manager()
    return [
        PromptVersion(
            version=v["version"],
            description=v["description"],
            created_at=v["created_at"],
            is_default=v["is_default"],
        )
        for v in manager.list_versions()
    ]


@router.get("/prompts/{version}", response_model=PromptContent)
async def get_prompt(
    version: str,
    _=Depends(verify_api_key),
) -> PromptContent:
    """获取指定版本 Prompt 内容."""
    manager = get_prompt_manager()
    info = manager.get(version)
    if not info:
        raise HTTPException(status_code=404, detail="prompt version not found")
    return PromptContent(
        version=info["version"],
        system=info["system"],
        user_template=info["user_template"],
    )


@router.put("/prompts/{version}")
async def update_prompt(
    version: str,
    req: PromptUpdateRequest,
    _=Depends(verify_api_key),
) -> dict:
    """更新指定版本 Prompt."""
    manager = get_prompt_manager()
    update_data = req.model_dump(exclude_unset=True)
    if not manager.update_version(version, **update_data):
        raise HTTPException(status_code=404, detail="prompt version not found")
    return {"status": "ok", "version": version}


@router.post("/prompts/{version}/set-default")
async def set_default_prompt(
    version: str,
    _=Depends(verify_api_key),
) -> dict:
    """切换默认 Prompt 版本."""
    manager = get_prompt_manager()
    if not manager.set_default(version):
        raise HTTPException(status_code=404, detail="prompt version not found")

    # 同步更新运行时配置
    settings = get_settings()
    # 注意：这里仅更新当前进程内存中的值；下次启动会读取 .env
    settings.llm_prompt_version = version

    logger.info(f"default prompt version switched to {version}")
    return {"status": "ok", "default_version": version}


@router.post("/prompts")
async def create_prompt(
    req: PromptUpdateRequest,
    version: str,
    _=Depends(verify_api_key),
) -> dict:
    """创建新 Prompt 版本."""
    manager = get_prompt_manager()
    if not req.system or not req.user_template:
        raise HTTPException(status_code=400, detail="system and user_template are required")

    from datetime import datetime

    created = manager.create_version(
        version=version,
        system=req.system,
        user_template=req.user_template,
        description=req.description or "",
        created_at=datetime.now().strftime("%Y-%m-%d"),
    )
    if not created:
        raise HTTPException(status_code=409, detail="prompt version already exists")
    return {"status": "created", "version": version}
