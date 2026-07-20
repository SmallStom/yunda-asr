"""音频降噪测试脚本.

用法示例：
    python scripts/denoise_audio.py 3/train_0023 2/train_0100 1/train_0040
    python scripts/denoise_audio.py 3/train_0023 --method deepfilternet --output-dir outputs/denoised
    python scripts/denoise_audio.py path/to/audio.wav --method noisereduce
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# 把项目根目录加入路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio_preprocessor import available_methods, denoise_file


DATASET_ROOT = Path("dataset/语音识别样本集/录音普通话-长文本")


def resolve_input(value: str) -> Path:
    """支持两种输入格式：
    - {folder}/{basename}  如 3/train_0023
    - 绝对/相对文件路径
    """
    p = Path(value)
    if p.exists():
        return p

    # 尝试按 {folder}/{basename} 解析
    parts = value.replace("\\", "/").split("/")
    if len(parts) == 2:
        folder, name = parts
        candidate = DATASET_ROOT / folder / "录音" / f"{name}.wav"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"找不到音频文件: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="对音频进行降噪并输出试听文件")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="音频标识，如 '3/train_0023'，或直接传入 wav 文件路径",
    )
    parser.add_argument(
        "--method",
        choices=["noisereduce", "deepfilternet", "ffmpeg_afftdn"],
        default="noisereduce",
        help="降噪方案，默认 noisereduce",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/denoised",
        help="降噪后文件输出目录，默认 outputs/denoised",
    )
    parser.add_argument(
        "--stationary",
        action="store_true",
        help="noisereduce 平稳噪声模式（默认非平稳）",
    )
    parser.add_argument(
        "--prop-decrease",
        type=float,
        default=1.0,
        help="noisereduce 降噪强度，默认 1.0",
    )
    parser.add_argument(
        "--copy-original",
        action="store_true",
        help="同时在输出目录复制一份原始音频，方便 A/B 对比",
    )

    args = parser.parse_args()

    usable = available_methods()
    if args.method not in usable:
        print(f"[错误] 当前环境不可用 '{args.method}'，可用方案: {usable}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for value in args.inputs:
        try:
            src = resolve_input(value)
        except FileNotFoundError as e:
            print(f"[跳过] {e}")
            continue

        # 输出文件名：{folder}_{name}_{method}.wav
        rel_parts = src.stem
        folder = src.parent.parent.name  # 如 3
        out_name = f"{folder}_{src.stem}_{args.method}.wav"
        dst = output_dir / out_name

        print(f"\n[处理] {value}")
        print(f"  原始: {src}")
        print(f"  方法: {args.method}")

        info = denoise_file(
            src,
            dst,
            method=args.method,
            stationary=args.stationary,
            prop_decrease=args.prop_decrease,
        )
        print(f"  降噪: {info['output']} (时长 {info['duration']:.2f}s, sr={info['sr']})")

        if args.copy_original:
            orig_dst = output_dir / f"{folder}_{src.stem}_original.wav"
            shutil.copy2(src, orig_dst)
            print(f"  原音: {orig_dst}")

    print(f"\n完成，输出目录: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
