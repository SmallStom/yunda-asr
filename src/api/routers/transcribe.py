"""语音转文本并纠错 API 路由.

完整链路：上传音频 -> [可选降噪] -> ASR -> 四级纠错 -> 输出纠错文本
"""

import tempfile
import time
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile

from src.api.dependencies import get_pipeline, verify_api_key
from src.asr_client import ASRClient
from src.logging_config import get_logger


router = APIRouter(prefix="/api/v1", tags=["transcribe"])
logger = get_logger(__name__)

# DeepFilterNet 固定参数（效果最好的配置）
DF_ATTEN_LIM_DB = 50.0


@router.post("/transcribe-and-correct")
async def transcribe_and_correct(
    file: UploadFile = File(..., description="音频文件（wav/mp3/flac/m4a 等）"),
    enable_denoise: bool = Form(default=False, description="是否开启 DeepFilterNet 降噪"),
    layers: Optional[str] = Form(default=None, description="启用的层号，逗号分隔，如 1,2,3"),
    enable_semantic: bool = Form(default=True, description="是否启用 Layer 4 语义精修"),
    semantic_mode: str = Form(default="rag", description="语义精修模式：baseline/rag/harness"),
    pipeline=Depends(get_pipeline),
    _=Depends(verify_api_key),
) -> dict:
    """上传音频，自动转文本并纠错.

    链路：音频 -> [降噪] -> ASR -> 纠错 -> 输出
    """
    start = time.perf_counter()
    steps = {}

    # 1. 保存上传的音频到临时文件
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        audio_path = Path(tmp.name)

    try:
        # 2. 可选降噪
        if enable_denoise:
            denoise_start = time.perf_counter()
            try:
                from src.audio_preprocessor import AudioPreprocessor

                processor = AudioPreprocessor(target_sr=16000)
                y, sr = processor.load(audio_path)
                y_denoised = processor.denoise(y, sr, method="deepfilternet", atten_lim_db=DF_ATTEN_LIM_DB)

                denoised_path = audio_path.with_suffix(".denoised.wav")
                processor.save(denoised_path, y_denoised, sr=sr)
                audio_path.unlink()
                audio_path = denoised_path
                steps["denoise"] = {"method": "deepfilternet", "atten_lim_db": DF_ATTEN_LIM_DB}
            except Exception as e:
                logger.warning(f"denoise failed, skipping: {e}")
                steps["denoise"] = {"skipped": str(e)}
            steps["denoise_latency_ms"] = (time.perf_counter() - denoise_start) * 1000

        # 3. ASR 转文本
        asr_start = time.perf_counter()
        asr_client = ASRClient()
        asr_text = asr_client.transcribe(audio_path)
        steps["asr_latency_ms"] = (time.perf_counter() - asr_start) * 1000
        steps["asr_text"] = asr_text

        if not asr_text:
            return {
                "status": "empty",
                "message": "ASR 返回空文本",
                "steps": steps,
            }

        # 4. 文本纠错
        layer_list = None
        if layers:
            try:
                layer_list = [int(x.strip()) for x in layers.split(",") if x.strip()]
            except ValueError:
                layer_list = None  # 解析失败，默认走全部层

        correct_start = time.perf_counter()
        result = pipeline.run(
            asr_text,
            layers=layer_list,
            enable_semantic=enable_semantic,
            semantic_mode=semantic_mode,
        )
        steps["correct_latency_ms"] = (time.perf_counter() - correct_start) * 1000

        total_latency = (time.perf_counter() - start) * 1000

        return {
            "status": "ok",
            "original_audio": file.filename,
            "asr_text": asr_text,
            "corrected": result.corrected,
            "layers_applied": result.layers_applied,
            "layer_outputs": result.layer_outputs,
            "details": [{"layer": d.layer, "changes": d.changes} for d in result.details],
            "steps": steps,
            "total_latency_ms": total_latency,
        }
    finally:
        # 清理临时文件
        if audio_path.exists():
            audio_path.unlink()
