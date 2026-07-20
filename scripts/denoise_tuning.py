"""批量生成不同降噪参数的对比文件.

用法：
    python scripts/denoise_tuning.py 3/train_0023 2/train_0100 1/train_0040
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio_preprocessor import denoise_file

DATASET_ROOT = Path("dataset/语音识别样本集/录音普通话-长文本")
OUTPUT_DIR = Path("outputs/denoised_tuning")


def resolve_input(value: str) -> Path:
    p = Path(value)
    if p.exists():
        return p
    parts = value.replace("\\", "/").split("/")
    if len(parts) == 2:
        folder, name = parts
        candidate = DATASET_ROOT / folder / "录音" / f"{name}.wav"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(value)


# ---- 参数网格 ----
NOISEREDUCE_CONFIGS = [
    # (文件名后缀, kwargs)
    ("nr_stat_p100", {"method": "noisereduce", "stationary": True, "prop_decrease": 1.0}),
    ("nr_nonstat_p100_t2", {"method": "noisereduce", "stationary": False, "prop_decrease": 1.0, "time_constant_s": 2.0}),
    ("nr_nonstat_p95_t2", {"method": "noisereduce", "stationary": False, "prop_decrease": 0.95, "time_constant_s": 2.0}),
    ("nr_nonstat_p80_t2", {"method": "noisereduce", "stationary": False, "prop_decrease": 0.80, "time_constant_s": 2.0}),
    ("nr_nonstat_p100_t1", {"method": "noisereduce", "stationary": False, "prop_decrease": 1.0, "time_constant_s": 1.0}),
    ("nr_nonstat_p100_t4", {"method": "noisereduce", "stationary": False, "prop_decrease": 1.0, "time_constant_s": 4.0}),
    ("nr_nonstat_p100_t2_fm250", {"method": "noisereduce", "stationary": False, "prop_decrease": 1.0, "time_constant_s": 2.0, "freq_mask_smooth_hz": 250}),
    ("nr_nonstat_p100_t2_tm100", {"method": "noisereduce", "stationary": False, "prop_decrease": 1.0, "time_constant_s": 2.0, "time_mask_smooth_ms": 100}),
]

FFMPEG_CONFIGS = [
    ("ff_nf-20_nr12", {"method": "ffmpeg_afftdn", "noise_reduction_db": 12}),
    ("ff_nf-30_nr12", {"method": "ffmpeg_afftdn", "noise_reduction_db": 12, "noise_floor": -30}),
    ("ff_nf-40_nr12", {"method": "ffmpeg_afftdn", "noise_reduction_db": 12, "noise_floor": -40}),
    ("ff_nf-30_nr6", {"method": "ffmpeg_afftdn", "noise_reduction_db": 6, "noise_floor": -30}),
    ("ff_nf-30_nr24", {"method": "ffmpeg_afftdn", "noise_reduction_db": 24, "noise_floor": -30}),
]

DEEPFILTERNET_ATTENS = [None, 80, 50, 30, 12]


def run_noisereduce_ffmpeg(src: Path, configs: list, output_dir: Path) -> list[Path]:
    outputs = []
    for suffix, kwargs in configs:
        dst = output_dir / f"{src.stem}_{suffix}.wav"
        info = denoise_file(src, dst, **kwargs)
        outputs.append(Path(info["output"]))
        print(f"  {suffix}: {info['output']}")
    return outputs


def run_deepfilternet(src: Path, output_dir: Path) -> list[Path]:
    outputs = []
    py311 = Path("tools/py311/python.exe")
    for att in DEEPFILTERNET_ATTENS:
        cmd = [
            str(py311),
            "scripts/denoise_deepfilternet_py311.py",
            str(src),
            "--output-dir",
            str(output_dir),
        ]
        if att is not None:
            cmd += ["--atten-lim-db", str(att)]
        subprocess.run(cmd, check=True)
        suffix = f"deepfilternet_att{int(att)}" if att is not None else "deepfilternet"
        out = output_dir / f"{src.stem}_{suffix}.wav"
        outputs.append(out)
        print(f"  df_att{att}: {out}")
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for value in args.inputs:
        try:
            src = resolve_input(value)
        except FileNotFoundError as e:
            print(f"[跳过] {e}")
            continue

        print(f"\n[处理] {value} -> {src}")

        print("  noisereduce 参数组...")
        run_noisereduce_ffmpeg(src, NOISEREDUCE_CONFIGS, OUTPUT_DIR)

        print("  ffmpeg afftdn 参数组...")
        run_noisereduce_ffmpeg(src, FFMPEG_CONFIGS, OUTPUT_DIR)

        print("  DeepFilterNet 参数组...")
        run_deepfilternet(src, OUTPUT_DIR)

    print(f"\n完成，输出目录: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
