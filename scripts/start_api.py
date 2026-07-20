"""启动 API 服务脚本.

用法:
    python scripts/start_api.py

环境变量通过 .env 文件加载，详见 src/config.py。
"""

import os
import sys
from pathlib import Path

# 加载项目根目录下的 .env
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# 将项目根目录加入路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn

from src.config import get_settings


def main():
    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_level=settings.log_level.lower(),
        reload=False,
    )


if __name__ == "__main__":
    main()
