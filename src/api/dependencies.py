"""API 依赖注入."""

import time
import uuid
from typing import Optional

from fastapi import Header, HTTPException, Request

from src.config import get_settings
from src.pipeline import PostCorrectionPipeline


# 全局单例，延迟初始化
_pipeline_instance: Optional[PostCorrectionPipeline] = None


def get_pipeline() -> PostCorrectionPipeline:
    """获取或初始化 Pipeline 单例."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = PostCorrectionPipeline()
    return _pipeline_instance


def warmup_pipeline(sample_text: str = "18号道岔开通反位") -> None:
    """预热 Pipeline."""
    pipeline = get_pipeline()
    pipeline.warmup(sample_text)


async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    """API Key 鉴权（可选）."""
    settings = get_settings()
    if not settings.api_key:
        return
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


async def add_request_metadata(request: Request) -> None:
    """添加请求级元数据：request_id、开始时间."""
    request.state.request_id = str(uuid.uuid4())
    request.state.start_time = time.perf_counter()
