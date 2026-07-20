"""本地 CPU 跑通 Qwen3-ASR 单条示例（绕过 Windows symlink 限制）."""
import os
import shutil
import sys
from pathlib import Path

# 绕过 huggingface_hub 在 Windows 上创建 symlink 失败的问题
import huggingface_hub.file_download as _fd

_orig_create_symlink = _fd._create_symlink


def _copy_or_create_symlink(src: str, dst: str, new_blob: bool = False):
    try:
        _orig_create_symlink(src, dst, new_blob=new_blob)
    except OSError:
        # 复制文件代替软链接
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

_fd._create_symlink = _copy_or_create_symlink

import torch
from qwen_asr import Qwen3ASRModel


def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "dataset/语音识别样本集/录音普通话-长文本/1/录音/train_0008.wav"
    print("Loading Qwen3-ASR-1.7B on CPU...")
    model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        dtype=torch.float32,
        device_map="cpu",
        max_new_tokens=256,
    )
    print(f"Transcribing {audio_path} ...")
    results = model.transcribe(audio=audio_path, language="zh")
    print("---")
    print("Text:", results[0].text)
    print("Language:", results[0].language)


if __name__ == "__main__":
    main()
