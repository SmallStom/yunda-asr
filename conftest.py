"""共享 pytest fixtures 和配置."""

import sys
from pathlib import Path

import pytest

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(scope="session")
def pipeline():
    """全局共享的流水线实例（session 级别，避免重复初始化）."""
    from src.pipeline import PostCorrectionPipeline
    p = PostCorrectionPipeline()
    p.warmup()
    return p


@pytest.fixture(scope="session")
def api_client():
    """FastAPI TestClient."""
    from fastapi.testclient import TestClient
    from src.api.main import app

    return TestClient(app)
