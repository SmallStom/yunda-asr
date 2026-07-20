"""音频预处理模块：在ASR之前对输入音频进行降噪/增强.

支持的降噪方案（按推荐度排序）：
- deepfilternet：基于深度学习的语音增强，效果最佳，但需要安装 deepfilternet 和 torch。
- noisereduce：基于谱减法的轻量降噪，无需大模型，对稳态/缓变噪声效果好。
- ffmpeg_afftdn：基于 ffmpeg afftdn 滤波器的快速降噪（需要系统安装 ffmpeg）。
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Literal

import librosa
import numpy as np
import soundfile as sf

DenoiseMethod = Literal["noisereduce", "deepfilternet", "ffmpeg_afftdn"]

# 便携 Python 3.11 环境路径（用于在主环境无法直接 import deepfilternet 时调用）
_PY311_PYTHON = Path(__file__).parent.parent / "tools" / "py311" / "python.exe"


def _get_ffmpeg_exe() -> str | None:
    """获取可用的 ffmpeg 可执行文件路径.

    优先使用系统 PATH 中的 ffmpeg；找不到时回退到 imageio-ffmpeg 内置的 ffmpeg。
    """
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg
        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and Path(bundled).exists():
            return bundled
    except Exception:
        pass
    return None


class AudioPreprocessor:
    """音频预处理器：统一音频格式 + 可选降噪."""

    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr

    def load(self, path: str | Path) -> tuple[np.ndarray, int]:
        """加载音频并转换为单声道、目标采样率."""
        y, sr = librosa.load(str(path), sr=self.target_sr, mono=True)
        return y, sr

    def save(self, path: str | Path, y: np.ndarray, sr: int | None = None) -> None:
        """保存音频，峰值归一化防止削波."""
        sr = sr or self.target_sr
        y = self.normalize(y)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), y, sr, subtype="PCM_16")

    @staticmethod
    def normalize(y: np.ndarray, peak: float = 0.95) -> np.ndarray:
        """峰值归一化."""
        if y.size == 0:
            return y
        max_val = np.max(np.abs(y))
        if max_val == 0:
            return y
        return y / max_val * peak

    def denoise(
        self,
        y: np.ndarray,
        sr: int,
        method: DenoiseMethod = "noisereduce",
        **kwargs,
    ) -> np.ndarray:
        """对音频进行降噪处理."""
        if method == "noisereduce":
            return self._denoise_noisereduce(y, sr, **kwargs)
        if method == "deepfilternet":
            return self._denoise_deepfilternet(y, sr, **kwargs)
        if method == "ffmpeg_afftdn":
            return self._denoise_ffmpeg_afftdn(y, sr, **kwargs)
        raise ValueError(f"不支持的降噪方法: {method}")

    def _denoise_noisereduce(
        self,
        y: np.ndarray,
        sr: int,
        stationary: bool = False,
        prop_decrease: float = 1.0,
        time_constant_s: float = 2.0,
        **kwargs,
    ) -> np.ndarray:
        try:
            import noisereduce as nr
        except ImportError as exc:
            raise ImportError(
                "使用 noisereduce 降噪需要安装 noisereduce：pip install noisereduce"
            ) from exc

        return nr.reduce_noise(
            y=y,
            sr=sr,
            stationary=stationary,
            prop_decrease=prop_decrease,
            time_constant_s=time_constant_s,
            **kwargs,
        )

    def _denoise_deepfilternet(
        self,
        y: np.ndarray,
        sr: int,
        atten_lim_db: float = 50.0,
        **kwargs,
    ) -> np.ndarray:
        """DeepFilterNet 降噪。

        主环境能 import deepfilternet 时直接调用；否则回退到 tools/py311 便携环境。
        """
        try:
            from df.enhance import enhance, init_df, load_audio, save_audio
        except ImportError:
            return self._denoise_deepfilternet_subprocess(y, sr, atten_lim_db)

        model, df_state, _ = init_df()
        df_sr = df_state.sr()

        # 将 numpy 音频写入临时文件，复用 deepfilternet 的 IO 逻辑
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = Path(tmpdir) / "input.wav"
            out_path = Path(tmpdir) / "output.wav"
            self.save(in_path, y, sr=sr)

            audio = load_audio(str(in_path), sr=df_sr)
            enhanced = enhance(model, df_state, audio, atten_lim_db=atten_lim_db)
            save_audio(str(out_path), enhanced, sr=df_sr)

            enhanced_y, _ = librosa.load(str(out_path), sr=self.target_sr, mono=True)

        return enhanced_y

    def _denoise_deepfilternet_subprocess(
        self,
        y: np.ndarray,
        sr: int,
        atten_lim_db: float,
    ) -> np.ndarray:
        """通过 tools/py311 子进程调用 DeepFilterNet（主环境缺少依赖时使用）."""
        if not _PY311_PYTHON.exists():
            raise RuntimeError(
                "主环境未安装 deepfilternet，且找不到便携环境 tools/py311/python.exe。"
                "请先运行 scripts/setup_deepfilternet_py311.ps1"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = Path(tmpdir) / "input.wav"
            out_path = Path(tmpdir) / "output_deepfilternet_att50.wav"
            self.save(in_path, y, sr=sr)

            cmd = [
                str(_PY311_PYTHON),
                str(Path(__file__).parent.parent / "scripts" / "denoise_deepfilternet_py311.py"),
                str(in_path),
                "--output-dir",
                str(tmpdir),
                "--atten-lim-db",
                str(atten_lim_db),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if not out_path.exists():
                # 脚本命名规则：{stem}_deepfilternet_att{int(db)}.wav
                candidates = list(Path(tmpdir).glob("*_deepfilternet*.wav"))
                if not candidates:
                    raise RuntimeError("DeepFilterNet 子进程未生成输出文件")
                out_path = candidates[0]

            enhanced_y, _ = librosa.load(str(out_path), sr=self.target_sr, mono=True)

        return enhanced_y

    def _denoise_ffmpeg_afftdn(
        self,
        y: np.ndarray,
        sr: int,
        noise_reduction_db: float = 12.0,
        noise_floor_db: float = -25.0,
        **kwargs,
    ) -> np.ndarray:
        ffmpeg_exe = _get_ffmpeg_exe()
        if ffmpeg_exe is None:
            raise RuntimeError(
                "使用 ffmpeg_afftdn 需要 ffmpeg。可安装系统 ffmpeg，或执行 pip install imageio-ffmpeg"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = Path(tmpdir) / "input.wav"
            out_path = Path(tmpdir) / "output.wav"
            self.save(in_path, y, sr=sr)

            # afftdn 参数说明：
            # nf 噪声系数（dB），值越小降噪越强（范围 -80~-20）；
            # nr 噪声抑制量（dB，范围 0.01~97）
            cmd = [
                ffmpeg_exe,
                "-y",
                "-i",
                str(in_path),
                "-af",
                f"afftdn=nf={noise_floor_db:.0f}:nr={noise_reduction_db:.2f}",
                "-ar",
                str(self.target_sr),
                "-ac",
                "1",
                str(out_path),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            out_y, _ = librosa.load(str(out_path), sr=self.target_sr, mono=True)

        return out_y

    def process_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        method: DenoiseMethod = "noisereduce",
        **kwargs,
    ) -> dict:
        """端到端处理单个文件：加载 -> 降噪 -> 保存.

        Returns:
            处理信息字典，包含 input/output 路径和音频时长等。
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        y, sr = self.load(input_path)
        duration = len(y) / sr

        y_denoised = self.denoise(y, sr, method=method, **kwargs)
        self.save(output_path, y_denoised, sr=sr)

        return {
            "method": method,
            "input": str(input_path),
            "output": str(output_path),
            "duration": duration,
            "sr": sr,
        }


def denoise_file(
    input_path: str | Path,
    output_path: str | Path,
    method: DenoiseMethod = "noisereduce",
    target_sr: int = 16000,
    **kwargs,
) -> dict:
    """便捷函数：对单个音频文件降噪."""
    processor = AudioPreprocessor(target_sr=target_sr)
    return processor.process_file(input_path, output_path, method=method, **kwargs)


def available_methods() -> list[DenoiseMethod]:
    """返回当前环境可用的降噪方法列表."""
    methods: list[DenoiseMethod] = []

    try:
        import noisereduce  # noqa: F401
        methods.append("noisereduce")
    except ImportError:
        pass

    try:
        from df.enhance import enhance  # noqa: F401
        methods.append("deepfilternet")
    except ImportError:
        # 主环境没有 deepfilternet，但便携 Python 3.11 环境可用时也算可用
        if _PY311_PYTHON.exists():
            methods.append("deepfilternet")

    if _get_ffmpeg_exe() is not None:
        methods.append("ffmpeg_afftdn")

    return methods
