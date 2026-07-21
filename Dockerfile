# 轨道交通 ASR 后处理纠错服务 Docker 镜像
# 构建阶段
FROM python:3.12-slim-bookworm AS builder

WORKDIR /app

# 使用国内镜像加速 apt/pip（如服务器在海外，可注释下一行或替换为其他镜像）
ARG APT_MIRROR=mirrors.tuna.tsinghua.edu.cn
ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

RUN if [ -n "$APT_MIRROR" ]; then \
        sed -i "s|deb.debian.org|$APT_MIRROR|g" /etc/apt/sources.list.d/debian.sources 2>/dev/null || true; \
        sed -i "s|security.debian.org|$APT_MIRROR|g" /etc/apt/sources.list.d/debian.sources 2>/dev/null || true; \
        sed -i "s|deb.debian.org|$APT_MIRROR|g" /etc/apt/sources.list 2>/dev/null || true; \
        sed -i "s|security.debian.org|$APT_MIRROR|g" /etc/apt/sources.list 2>/dev/null || true; \
    fi

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖配置并安装
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip --index-url "$PIP_INDEX" && \
    pip install --no-cache-dir -e ".[audio,dify]" --index-url "$PIP_INDEX" || \
    pip install --no-cache-dir -e ".[dify]" --index-url "$PIP_INDEX" || \
    pip install --no-cache-dir -e "." --index-url "$PIP_INDEX"

# 运行阶段
FROM python:3.12-slim-bookworm

WORKDIR /app

ARG APT_MIRROR=mirrors.tuna.tsinghua.edu.cn

RUN if [ -n "$APT_MIRROR" ]; then \
        sed -i "s|deb.debian.org|$APT_MIRROR|g" /etc/apt/sources.list.d/debian.sources 2>/dev/null || true; \
        sed -i "s|security.debian.org|$APT_MIRROR|g" /etc/apt/sources.list.d/debian.sources 2>/dev/null || true; \
        sed -i "s|deb.debian.org|$APT_MIRROR|g" /etc/apt/sources.list 2>/dev/null || true; \
        sed -i "s|security.debian.org|$APT_MIRROR|g" /etc/apt/sources.list 2>/dev/null || true; \
    fi

# 安装运行时依赖（ffmpeg 用于音频处理）
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 从构建阶段复制已安装的 Python 包
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制项目代码
COPY src ./src
COPY scripts ./scripts
COPY data ./data
COPY pyproject.toml ./

# 创建非 root 用户
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# 启动命令
CMD ["python", "scripts/start_api.py"]
