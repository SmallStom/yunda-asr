"""FastAPI 应用入口."""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.dependencies import add_request_metadata, warmup_pipeline
from src.api.routers import aliases, config, correction, dify_sync, health, hotwords, prompts, transcribe
from src.config import get_settings
from src.logging_config import get_logger, setup_logging
from src.metrics import get_metrics


setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理."""
    # 启动时预热
    try:
        warmup_pipeline()
        logger.info("pipeline warmed up")
    except Exception as e:
        logger.warning(f"pipeline warmup failed: {e}")
    yield
    # 关闭时清理
    logger.info("shutting down")


app = FastAPI(
    title="轨道交通 ASR 后处理纠错服务",
    description="面向铁路调度场景的 ASR 文本纠错 API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """请求日志中间件."""
    await add_request_metadata(request)
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception as exc:
        logger.exception(
            "unhandled exception",
            extra={"request_id": getattr(request.state, "request_id", None)},
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    latency_ms = (time.perf_counter() - start_time) * 1000
    is_error = response.status_code >= 500
    get_metrics().record_request(latency_ms, is_error=is_error)
    logger.info(
        "request completed",
        extra={
            "request_id": getattr(request.state, "request_id", None),
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
        },
    )
    return response


# 注册路由
app.include_router(health.router)
app.include_router(correction.router)
app.include_router(transcribe.router)
app.include_router(config.router)
app.include_router(hotwords.router)
app.include_router(aliases.router)
app.include_router(prompts.router)
app.include_router(dify_sync.router)


@app.get("/")
async def root():
    return {"message": "轨道交通 ASR 后处理纠错服务", "docs": "/docs"}
