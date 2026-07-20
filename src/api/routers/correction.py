"""纠错 API 路由."""

import time
from typing import List

from fastapi import APIRouter, Depends, Request

from src.api.dependencies import get_pipeline, verify_api_key
from src.api.schemas import (
    BatchCorrectionRequest,
    BatchCorrectionResponse,
    CorrectionDetail,
    CorrectionRequest,
    CorrectionResponse,
)
from src.logging_config import get_logger
from src.pipeline import PipelineResult


router = APIRouter(prefix="/api/v1", tags=["correction"])
logger = get_logger(__name__)


def _pipeline_result_to_response(result: PipelineResult, latency_ms: float) -> CorrectionResponse:
    return CorrectionResponse(
        original=result.original,
        corrected=result.corrected,
        layers_applied=result.layers_applied,
        layer_outputs=result.layer_outputs,
        details=[CorrectionDetail(layer=d.layer, changes=d.changes) for d in result.details],
        latency_ms=latency_ms,
    )


@router.post("/correct", response_model=CorrectionResponse)
async def correct(
    request: Request,
    req: CorrectionRequest,
    pipeline=Depends(get_pipeline),
    _=Depends(verify_api_key),
) -> CorrectionResponse:
    """单条文本纠错."""
    start = time.perf_counter()

    # 临时热词注入（如果实现）
    if req.hotwords_override:
        # TODO: 实现请求级热词覆盖
        logger.info("hotwords_override provided but not yet implemented", extra={"request_id": getattr(request.state, "request_id", None)})

    result = pipeline.run(
        req.text,
        layers=req.layers,
        enable_semantic=req.enable_semantic,
        semantic_mode=req.semantic_mode,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "corrected text",
        extra={
            "request_id": getattr(request.state, "request_id", None),
            "latency_ms": latency_ms,
            "layers": req.layers,
            "semantic_mode": req.semantic_mode,
        },
    )
    return _pipeline_result_to_response(result, latency_ms)


@router.post("/correct/batch", response_model=BatchCorrectionResponse)
async def correct_batch(
    request: Request,
    req: BatchCorrectionRequest,
    pipeline=Depends(get_pipeline),
    _=Depends(verify_api_key),
) -> BatchCorrectionResponse:
    """批量文本纠错."""
    start = time.perf_counter()
    results: List[CorrectionResponse] = []

    for item in req.items:
        item_start = time.perf_counter()
        result = pipeline.run(
            item.text,
            layers=item.layers,
            enable_semantic=item.enable_semantic,
            semantic_mode=item.semantic_mode,
        )
        item_latency = (time.perf_counter() - item_start) * 1000
        results.append(_pipeline_result_to_response(result, item_latency))

    total_latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "batch corrected",
        extra={
            "request_id": getattr(request.state, "request_id", None),
            "count": len(req.items),
            "total_latency_ms": total_latency_ms,
        },
    )
    return BatchCorrectionResponse(results=results, total_latency_ms=total_latency_ms)
