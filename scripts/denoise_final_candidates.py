"""使用用户筛选出的候选参数，在更多随机样本上生成对比文件.

用法：
    python scripts/denoise_final_candidates.py --per-folder 2 --seed 42
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio_preprocessor import denoise_file

DATASET_ROOT = Path("dataset/语音识别样本集/录音普通话-长文本")
OUTPUT_DIR = Path("outputs/denoised_final")

# ---- 用户觉得效果还不错的参数组合（去重） ----
NOISEREDUCE_CONFIGS = [
    ("nr_stat_p100", {"method": "noisereduce", "stationary": True, "prop_decrease": 1.0}),
    ("nr_nonstat_p100_t4", {"method": "noisereduce", "stationary": False, "prop_decrease": 1.0, "time_constant_s": 4.0}),
    ("nr_nonstat_p100_t2_tm100", {"method": "noisereduce", "stationary": False, "prop_decrease": 1.0, "time_constant_s": 2.0, "time_mask_smooth_ms": 100}),
    ("nr_nonstat_p100_t2_fm250", {"method": "noisereduce", "stationary": False, "prop_decrease": 1.0, "time_constant_s": 2.0, "freq_mask_smooth_hz": 250}),
    ("nr_nonstat_p100_t2", {"method": "noisereduce", "stationary": False, "prop_decrease": 1.0, "time_constant_s": 2.0}),
    ("nr_nonstat_p100_t1", {"method": "noisereduce", "stationary": False, "prop_decrease": 1.0, "time_constant_s": 1.0}),
    ("nr_nonstat_p95_t2", {"method": "noisereduce", "stationary": False, "prop_decrease": 0.95, "time_constant_s": 2.0}),
]

FFMPEG_CONFIGS = [
    ("ff_nf-30_nr24", {"method": "ffmpeg_afftdn", "noise_floor_db": -30, "noise_reduction_db": 24}),
]

DEEPFILTERNET_ATTENS = [None, 80, 50, 30]


def discover_files() -> dict[str, list[Path]]:
    files: dict[str, list[Path]] = {}
    for folder in ["1", "2", "3"]:
        folder_dir = DATASET_ROOT / folder / "录音"
        if folder_dir.exists():
            files[folder] = sorted(folder_dir.glob("*.wav"))
    return files


def pick_samples(files: dict[str, list[Path]], per_folder: int, seed: int, exclude: set[str]) -> list[tuple[str, Path]]:
    rng = random.Random(seed)
    chosen: list[tuple[str, Path]] = []
    for folder, paths in files.items():
        candidates = [p for p in paths if p.stem not in exclude]
        if len(candidates) < per_folder:
            print(f"警告：{folder} 可选文件不足 {per_folder}，只取 {len(candidates)}")
        sample = rng.sample(candidates, min(per_folder, len(candidates)))
        chosen.extend([(folder, p) for p in sample])
    return chosen


def run_nr_ff(src: Path, output_dir: Path):
    for suffix, kwargs in NOISEREDUCE_CONFIGS + FFMPEG_CONFIGS:
        dst = output_dir / f"{src.stem}_{suffix}.wav"
        info = denoise_file(src, dst, **kwargs)
        print(f"  {suffix}: {info['output']}")


def run_deepfilternet(samples: list[tuple[str, Path]], output_dir: Path):
    py311 = Path("tools/py311/python.exe")
    files = [str(p) for _, p in samples]
    for att in DEEPFILTERNET_ATTENS:
        cmd = [
            str(py311),
            "scripts/denoise_deepfilternet_py311.py",
            *files,
            "--output-dir",
            str(output_dir),
        ]
        if att is not None:
            cmd += ["--atten-lim-db", str(att)]
        print(f"  deepfilternet att={att} ...")
        subprocess.run(cmd, check=True)


def copy_originals(samples: list[tuple[str, Path]], output_dir: Path):
    for _, src in samples:
        import shutil
        dst = output_dir / f"{src.stem}_original.wav"
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-folder", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    exclude = {"train_0040", "train_0100", "train_0023"}
    files = discover_files()
    samples = pick_samples(files, args.per_folder, args.seed, exclude)
    print(f"随机挑选样本 ({len(samples)} 条):")
    for folder, p in samples:
        print(f"  {folder}/{p.name}")

    print("\n[1/3] 复制原音...")
    copy_originals(samples, OUTPUT_DIR)

    print("\n[2/3] noisereduce / ffmpeg 候选参数...")
    for _, src in samples:
        print(f"\n  {src.stem}")
        run_nr_ff(src, OUTPUT_DIR)

    print("\n[3/3] DeepFilterNet 候选参数...")
    run_deepfilternet(samples, OUTPUT_DIR)

    print(f"\n完成，输出目录: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
