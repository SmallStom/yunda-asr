"""为构建 qwen3 few-shot 提示词采集真实错误样本（排除 100 条测试集）.

从长文本测试集中挑选非测试样本，做 DeepFilterNet 降噪 + qwen3-asr 转写，
保存为 data/asr_testset/qwen3_fewshot_samples.jsonl。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio_preprocessor import denoise_file
from scripts.evaluate_qwen3_pipeline import transcribe_qwen3, load_hotwords

DATASET_ROOT = Path("dataset/语音识别样本集/录音普通话-长文本")
LONG_TESTSET = Path("data/asr_testset/asr_test_pairs_long.jsonl")
ELEVEN_TESTSET = Path("data/asr_testset/asr_test_pairs_elevenlabs.jsonl")
OUT = Path("data/asr_testset/qwen3_fewshot_samples.jsonl")


def resolve_audio(record_id):
    parts = record_id.split("/")
    if len(parts) != 2:
        return None
    folder, name = parts
    candidate = DATASET_ROOT / folder / "录音" / f"{name}.wav"
    return candidate if candidate.exists() else None


def main():
    test_ids = set()
    with open(ELEVEN_TESTSET, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_ids.add(json.loads(line)["id"])

    records = []
    with open(LONG_TESTSET, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r["id"] not in test_ids:
                    records.append(r)
    print(f"[INFO] 非测试样本 {len(records)} 条")

    # 每个子目录取 10 条
    selected = []
    for sub in ["1", "2", "3"]:
        sub_records = [r for r in records if r["id"].startswith(sub + "/")][:10]
        selected.extend(sub_records)
    print(f"[INFO] 选取 {len(selected)} 条采集 qwen3 转写")

    hotwords = load_hotwords()
    results = []
    for idx, r in enumerate(selected, 1):
        audio = resolve_audio(r["id"])
        if audio is None:
            continue
        denoised = Path("outputs/qwen3_fewshot_denoised") / f"{audio.stem}_att50.wav"
        denoised.parent.mkdir(parents=True, exist_ok=True)
        try:
            denoise_file(str(audio), str(denoised), method="deepfilternet")
        except Exception as e:
            print(f"  降噪失败 {r['id']}: {e}")
            denoised = audio
        try:
            asr = transcribe_qwen3(denoised, hotwords, "http://192.168.1.119:8014", "/models/Qwen3-ASR-1.7B")
        except Exception as e:
            print(f"  ASR失败 {r['id']}: {e}")
            continue
        rec = {"id": r["id"], "asr": asr, "correct": r["correct"]}
        results.append(rec)
        print(f"  [{idx}/{len(selected)}] {r['id']} done")

    with OUT.open("w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[INFO] 保存 {len(results)} 条 -> {OUT}")


if __name__ == "__main__":
    main()
