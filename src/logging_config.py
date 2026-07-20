"""日志配置.

统一 JSON 格式日志，包含 request_id、trace_id、latency 等字段。
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional

from src.config import get_settings


class JSONFormatter(logging.Formatter):
    """JSON 格式日志格式化器."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # 添加额外字段
        for key in ("request_id", "trace_id", "latency_ms", "path", "method", "status_code"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        # 添加异常信息
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False, default=str)


def setup_logging(log_level: Optional[str] = None) -> None:
    """配置根日志记录器."""
    settings = get_settings()
    level = (log_level or settings.log_level).upper()

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 清除已有 handler，避免重复输出
    root_logger.handlers.clear()

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(stdout_handler)

    # 降低第三方库日志级别
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("gradio").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """获取命名日志记录器."""
    return logging.getLogger(name)
