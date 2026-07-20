"""调用VibeVoice ASR服务，构建ASR输出与正确文本的对照测试集."""

import argparse
import base64
import json
import re
import sys
import time
import wave
from pathlib import Path

import requests

ASR_BASE_URL = "http://192.168.1.119:8015"
DATASET_DIR = Path("dataset/语音识别样本集/录音普通话-全量")
OUTPUT_DIR = Path("data/asr_testset")

SYSTEM_PROMPT = "You are a helpful assistant that transcribes audio input into text output in JSON format."


def get_audio_duration(path: Path) -> float:
    """获取wav音频时长（秒）."""
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def load_text_mapping(text_file: Path) -> dict[str, str]:
    """从text.txt加载 audio_id -> 正确文本 的映射.

    格式: train_0001 预告
    """
    mapping = {}
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                mapping[parts[0]] = parts[1].strip()
    return mapping


def call_asr(audio_path: Path, base_url: str = ASR_BASE_URL, timeout: int = 300, hotwords: str = "") -> str:
    """调用VibeVoice ASR服务，返回原始文本输出.

    Args:
        audio_path: 音频文件路径
        base_url: ASR服务地址
        timeout: 超时秒数
        hotwords: 热词字符串（逗号分隔），作为context prompt传入VibeVoice-ASR
    """
    duration = get_audio_duration(audio_path)
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
    data_url = f"data:audio/wav;base64,{audio_b64}"
    prompt_text = (
        f"This is a {duration:.2f} seconds audio, "
        f"please transcribe it with these keys: Start time, End time, Speaker ID, Content"
    )
    # VibeVoice-ASR 通过 prompt 参数提供上下文/热词
    # 在 OpenAI 兼容API中，热词作为额外文本拼接到 prompt_text 中
    if hotwords:
        # 截取前100个热词避免prompt过长（VibeVoice建议热词不宜过多）
        hw_list = [w.strip() for w in hotwords.split(",") if w.strip()][:100]
        prompt_text += f"\n\nContext keywords: {', '.join(hw_list)}"
    payload = {
        "model": "vibevoice",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": data_url}},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ],
        "max_tokens": 32768,
        "temperature": 0.0,
        "stream": True,
        "top_p": 1.0,
    }
    response = requests.post(
        f"{base_url}/v1/chat/completions", json=payload, stream=True, timeout=timeout
    )
    if response.status_code != 200:
        raise RuntimeError(f"ASR请求失败: {response.status_code} {response.text[:500]}")

    # 流式拼接完整文本
    full_text = ""
    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8")
        if not decoded.startswith("data: "):
            continue
        json_str = decoded[6:]
        if json_str.strip() == "[DONE]":
            break
        try:
            data = json.loads(json_str)
            content = data["choices"][0]["delta"].get("content", "")
            if content:
                if full_text and content.startswith(full_text):
                    full_text = content
                else:
                    full_text += content
        except (json.JSONDecodeError, KeyError, IndexError):
            pass
    return full_text


def extract_transcript(raw_output: str) -> str:
    """从ASR原始JSON输出中提取纯文本.

    输入形如: [{"Start":0,"End":7.76,"Speaker":0,"Content":"按照列车..."}]
    提取所有Content字段并用空格拼接，过滤[Silence]等标记.
    """
    if not raw_output.strip():
        return ""
    try:
        segments = json.loads(raw_output)
        texts = []
        for seg in segments:
            content = seg.get("Content", "").strip()
            if content and content != "[Silence]":
                texts.append(content)
        return " ".join(texts)
    except json.JSONDecodeError:
        # 如果不是JSON，直接返回原始文本
        return raw_output.strip()


def load_progress(progress_file: Path) -> set[str]:
    """加载已完成的ID集合（用于断点续跑）."""
    if not progress_file.exists():
        return set()
    with open(progress_file, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def save_result(output_file: Path, record: dict) -> None:
    """追加写入一条结果到JSONL文件."""
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_progress(progress_file: Path, item_id: str) -> None:
    """记录已完成的ID."""
    with open(progress_file, "a", encoding="utf-8") as f:
        f.write(item_id + "\n")


def main():
    parser = argparse.ArgumentParser(description="构建ASR纠错测试集")
    parser.add_argument(
        "--base-url", default=ASR_BASE_URL, help=f"ASR服务地址 (默认: {ASR_BASE_URL})"
    )
    parser.add_argument(
        "--subdirs", nargs="*", default=["1", "2", "3", "4"],
        help="要处理的子目录列表 (默认: 1 2 3 4)",
    )
    parser.add_argument(
        "--max-retries", type=int, default=3, help="每个文件最大重试次数 (默认: 3)"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="限制处理文件数 (0=不限制)"
    )
    parser.add_argument(
        "--dataset", default="full",
        choices=["full", "long"],
        help="数据集类型: full=录音普通话-全量(短文本), long=录音普通话-长文本 (默认: full)",
    )
    parser.add_argument(
        "--output", default="",
        help="输出文件名 (默认: asr_test_pairs.jsonl 或 asr_test_pairs_long.jsonl)",
    )
    parser.add_argument(
        "--hotwords-file", default="",
        help="热词文件路径（CSV格式，逗号分隔）。如 data/lexicon/hotwords.csv",
    )
    parser.add_argument(
        "--no-hotwords", action="store_true",
        help="禁用热词（即使热词文件存在）",
    )
    args = parser.parse_args()

    # 加载热词
    hotwords_str = ""
    if not args.no_hotwords:
        hotwords_file = args.hotwords_file
        if not hotwords_file:
            # 默认热词文件路径
            default_hw = Path(__file__).parent.parent / "data" / "lexicon" / "hotwords.csv"
            if default_hw.exists():
                hotwords_file = str(default_hw)
        if hotwords_file and Path(hotwords_file).exists():
            with open(hotwords_file, "r", encoding="utf-8") as f:
                hotwords_str = f.read().strip()
            print(f"[INFO] 加载热词: {len(hotwords_str.split(','))} 个 (来源: {hotwords_file})")
        else:
            print("[INFO] 未找到热词文件，将不使用热词")

    # 根据数据集类型选择目录
    global DATASET_DIR
    if args.dataset == "long":
        DATASET_DIR = Path("dataset/语音识别样本集/录音普通话-长文本")

    # 准备目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        output_file = OUTPUT_DIR / args.output
    else:
        output_file = OUTPUT_DIR / ("asr_test_pairs_long.jsonl" if args.dataset == "long" else "asr_test_pairs.jsonl")
    progress_file = OUTPUT_DIR / (f"progress_{args.dataset}.txt")

    # 加载正确文本映射（长文本数据集每个子目录有自己的text.txt）
    text_mapping = {}
    if args.dataset == "long":
        # 长文本：每个子目录下都有 text.txt
        for subdir in args.subdirs:
            text_file = DATASET_DIR / subdir / "text.txt"
            if text_file.exists():
                sub_mapping = load_text_mapping(text_file)
                text_mapping.update(sub_mapping)
                print(f"[INFO] 加载 {subdir}/text.txt: {len(sub_mapping)} 条")
    else:
        # 短文本：根目录下的 text.txt
        text_file = DATASET_DIR / "text.txt"
        if not text_file.exists():
            print(f"[ERROR] 文本文件不存在: {text_file}", file=sys.stderr)
            sys.exit(1)
        text_mapping = load_text_mapping(text_file)
    print(f"[INFO] 加载文本映射总计: {len(text_mapping)} 条")

    # 收集所有待处理音频文件
    audio_files = []
    for subdir in args.subdirs:
        sub_path = DATASET_DIR / subdir
        if not sub_path.exists():
            print(f"[WARN] 子目录不存在，跳过: {sub_path}")
            continue
        # 长文本数据集音频在 "录音/" 子目录下
        if args.dataset == "long":
            audio_dir = sub_path / "录音"
            if not audio_dir.exists():
                audio_dir = sub_path  # 回退到子目录本身
        else:
            audio_dir = sub_path
        for wav in sorted(audio_dir.glob("*.wav")):
            audio_id = wav.stem  # e.g. train_0001
            if audio_id not in text_mapping:
                print(f"[WARN] 未找到对应文本，跳过: {subdir}/{wav.name}")
                continue
            item_id = f"{subdir}/{audio_id}"
            audio_files.append((item_id, audio_id, wav))

    print(f"[INFO] 待处理音频文件: {len(audio_files)} 个")
    if args.limit > 0:
        audio_files = audio_files[: args.limit]
        print(f"[INFO] 限制处理前 {args.limit} 个")

    # 加载已完成记录
    done_ids = load_progress(progress_file)
    print(f"[INFO] 已完成: {len(done_ids)} 个（将跳过）")

    # 处理
    total = len(audio_files)
    processed = 0
    failed = 0
    t_start = time.time()

    for idx, (item_id, audio_id, wav_path) in enumerate(audio_files, 1):
        if item_id in done_ids:
            continue

        correct_text = text_mapping[audio_id]
        raw_output = ""
        asr_text = ""
        success = False

        for attempt in range(1, args.max_retries + 1):
            try:
                raw_output = call_asr(wav_path, base_url=args.base_url, hotwords=hotwords_str)
                asr_text = extract_transcript(raw_output)
                success = True
                break
            except Exception as e:
                print(f"  [RETRY {attempt}/{args.max_retries}] {item_id} 失败: {e}")
                if attempt < args.max_retries:
                    time.sleep(2 * attempt)

        if not success:
            print(f"  [FAIL] {item_id} 重试{args.max_retries}次后仍失败，跳过")
            failed += 1
            # 记录失败也写入，避免反复重试
            record = {
                "id": item_id,
                "audio_id": audio_id,
                "asr": "",
                "correct": correct_text,
                "raw": "",
                "error": "asr_failed",
            }
            save_result(output_file, record)
            save_progress(progress_file, item_id)
            continue

        record = {
            "id": item_id,
            "audio_id": audio_id,
            "asr": asr_text,
            "correct": correct_text,
            "raw": raw_output,
        }
        save_result(output_file, record)
        save_progress(progress_file, item_id)
        processed += 1

        elapsed = time.time() - t_start
        speed = processed / elapsed if elapsed > 0 else 0
        print(
            f"[{idx}/{total}] {item_id} | "
            f"ASR: {asr_text[:40]}{'...' if len(asr_text) > 40 else ''} | "
            f"正确: {correct_text[:40]}{'...' if len(correct_text) > 40 else ''} | "
            f"{speed:.1f}条/秒"
        )

    elapsed = time.time() - t_start
    print(f"\n[DONE] 处理完成: 成功={processed}, 失败={failed}, 耗时={elapsed:.1f}s")
    print(f"[DONE] 结果已保存: {output_file}")


if __name__ == "__main__":
    main()
