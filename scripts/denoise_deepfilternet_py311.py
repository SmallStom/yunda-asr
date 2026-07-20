"""使用独立 Python 3.11 环境中的 DeepFilterNet 对音频降噪.

本脚本通过 soundfile 绕过 torchaudio 后端，兼容 torchaudio 2.11+。
"""

from __future__ import annotations

import argparse
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import types

# ---- 补丁：让 DeepFilterNet 在 torchaudio 2.11+ 下能导入 ----
# torchaudio 2.11 不再暴露 torchaudio.backend.common.AudioMetaData，
# 但 df.io 只在类型注解里用到它，实际运行走我们替换的 load/save。
AudioMetaData = namedtuple("AudioMetaData", ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"])

import torchaudio  # noqa: E402

if not hasattr(torchaudio, "backend"):
    torchaudio.backend = sys.modules["torchaudio.backend"] = types.ModuleType("torchaudio.backend")
torchaudio.backend.common = sys.modules["torchaudio.backend.common"] = types.ModuleType("torchaudio.backend.common")
torchaudio.backend.common.AudioMetaData = AudioMetaData

# 替换 df.io 的 IO 函数，避免依赖 torchaudio 的 ffmpeg 后端
from df import io  # noqa: E402
from df.enhance import enhance as df_enhance, init_df  # noqa: E402


def _load_audio(path: str, sr: int | None = None, **kwargs):
    data, orig_sr = sf.read(str(path), dtype="float32")
    if data.ndim == 1:
        data = data[np.newaxis, :]
    else:
        data = data.T  # [channels, samples]
    audio = torch.from_numpy(data)
    meta = AudioMetaData(sample_rate=orig_sr, num_frames=audio.shape[-1], num_channels=audio.shape[0], bits_per_sample=16, encoding="PCM")
    if sr is not None and orig_sr != sr:
        import torchaudio.functional
        audio = torchaudio.functional.resample(audio, orig_sr, sr)
    return audio.contiguous(), meta


def _save_audio(path: str, audio: torch.Tensor | np.ndarray, sr: int, output_dir: str | None = None, suffix: str | None = None, **kwargs):
    outpath = Path(path)
    if suffix:
        outpath = outpath.with_stem(f"{outpath.stem}_{suffix}")
    if output_dir:
        outpath = Path(output_dir) / outpath.name
    outpath.parent.mkdir(parents=True, exist_ok=True)
    arr = audio.numpy() if isinstance(audio, torch.Tensor) else np.asarray(audio)
    if arr.ndim == 2:
        arr = arr.T
    sf.write(str(outpath), arr, sr, subtype="PCM_16")


io.load_audio = _load_audio
io.save_audio = _save_audio


def denoise_file(input_path: Path, output_dir: Path, model, df_state, atten_lim_db: float | None = None) -> Path:
    audio, _ = _load_audio(str(input_path), sr=df_state.sr())
    with torch.no_grad():
        enhanced = df_enhance(model, df_state, audio, atten_lim_db=atten_lim_db)
    suffix = f"deepfilternet_att{int(atten_lim_db)}" if atten_lim_db is not None else "deepfilternet"
    out_path = output_dir / f"{input_path.stem}_{suffix}.wav"
    _save_audio(str(input_path), enhanced, sr=df_state.sr(), output_dir=str(output_dir), suffix=suffix)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output-dir", default="outputs/denoised")
    parser.add_argument("--atten-lim-db", type=float, default=None, help="降噪上限(dB)，越小保留噪声越多，语音更自然")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, df_state, _ = init_df()

    for value in args.inputs:
        parts = value.replace("\\", "/").split("/")
        if len(parts) == 2:
            folder, name = parts
            src = Path("dataset/语音识别样本集/录音普通话-长文本") / folder / "录音" / f"{name}.wav"
        else:
            src = Path(value)
        if not src.exists():
            print(f"[跳过] 找不到 {value}")
            continue
        out = denoise_file(src, output_dir, model, df_state, atten_lim_db=args.atten_lim_db)
        print(f"[完成] {src} -> {out}")


if __name__ == "__main__":
    main()
