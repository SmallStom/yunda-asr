"""分析测试集中规则纠错后仍存在的错误模式.

提取高频错误对（asr片段 → correct片段），用于补充纠错库.

Usage:
    python scripts/analyze_error_patterns.py
    python scripts/analyze_error_patterns.py --testset data/asr_testset/asr_test_pairs_qwen3.jsonl
"""

import argparse
import json
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import PostCorrectionPipeline
from tests.utils.metrics import cer


def extract_error_pairs(asr_text: str, correct_text: str) -> list:
    """通过序列比对提取错误对（asr片段 → correct片段）.

    Returns:
        错误对列表，每项为 (asr片段, correct片段)
    """
    pairs = []
    matcher = SequenceMatcher(None, asr_text, correct_text)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            asr_frag = asr_text[i1:i2]
            correct_frag = correct_text[j1:j2]
            # 过滤掉标点和单字
            if len(asr_frag) >= 2 or len(correct_frag) >= 2:
                pairs.append((asr_frag, correct_frag))
        elif tag == "delete":
            asr_frag = asr_text[i1:i2]
            if len(asr_frag) >= 2:
                pairs.append((asr_frag, ""))
        elif tag == "insert":
            correct_frag = correct_text[j1:j2]
            if len(correct_frag) >= 2:
                pairs.append(("", correct_frag))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="分析规则纠错后残留错误模式")
    parser.add_argument(
        "--testset",
        type=Path,
        default=Path("data/asr_testset/asr_test_pairs_elevenlabs.jsonl"),
        help="测试集JSONL文件路径",
    )
    args = parser.parse_args()
    testset_path = args.testset
    records = []
    with open(testset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"[INFO] 加载 {len(records)} 条记录 ({testset_path})")

    # 初始化pipeline
    pipeline = PostCorrectionPipeline()
    pipeline.warmup()

    # 收集所有错误对
    error_pairs = []  # (asr_frag, correct_frag, context_before, context_after)
    case_details = []  # 详细case信息

    for idx, record in enumerate(records, 1):
        asr_text = record["asr"]
        correct_text = record["correct"]

        # 规则纠错
        rule_result = pipeline.run(asr_text, layers=[1, 2, 3])
        rule_text = rule_result.corrected

        cer_rule = cer(rule_text, correct_text)
        if cer_rule < 0.01:
            continue  # 规则纠错后已经很好，跳过

        # 提取规则纠错后仍存在的错误
        pairs = extract_error_pairs(rule_text, correct_text)
        for asr_frag, correct_frag in pairs:
            error_pairs.append((asr_frag, correct_frag))
            case_details.append({
                "id": record.get("id", ""),
                "asr_frag": asr_frag,
                "correct_frag": correct_frag,
                "rule_text": rule_text[:100],
                "correct_text": correct_text[:100],
                "cer_rule": round(cer_rule, 4),
            })

    # 统计高频错误对
    pair_counter = Counter((a, c) for a, c, *_ in [(p[0], p[1]) for p in error_pairs])

    print(f"\n{'='*80}")
    print(f"规则纠错后仍存在的错误模式分析（共{len(error_pairs)}个错误片段）")
    print(f"{'='*80}")
    print(f"\n高频错误对 Top 40:")
    print(f"{'ASR片段':<20s} {'正确片段':<20s} {'出现次数':>8s}")
    print("-" * 50)

    for (asr_frag, correct_frag), count in pair_counter.most_common(40):
        asr_display = asr_frag if asr_frag else "[缺失]"
        correct_display = correct_frag if correct_frag else "[多余]"
        print(f"{asr_display:<20s} {correct_display:<20s} {count:>8d}")

    # 按类型分类
    print(f"\n{'='*80}")
    print("错误类型分类")
    print(f"{'='*80}")

    # 1. 词语替换（asr和correct都非空）
    replacements = [(a, c, n) for (a, c), n in pair_counter.most_common() if a and c]
    print(f"\n1. 词语替换错误 ({len(replacements)}种):")
    for asr_frag, correct_frag, count in replacements[:25]:
        print(f"   {asr_frag:<16s} → {correct_frag:<16s}  ({count}次)")

    # 2. 缺失内容（asr为空）
    missing = [(c, n) for (a, c), n in pair_counter.most_common() if not a and c]
    print(f"\n2. 规则纠错后仍缺失的内容 ({len(missing)}种):")
    for correct_frag, count in missing[:15]:
        print(f"   缺失: {correct_frag:<20s}  ({count}次)")

    # 3. 多余内容（correct为空）
    extra = [(a, n) for (a, c), n in pair_counter.most_common() if a and not c]
    print(f"\n3. 规则纠错后仍多余的内容 ({len(extra)}种):")
    for asr_frag, count in extra[:15]:
        print(f"   多余: {asr_frag:<20s}  ({count}次)")

    # 输出可补充到纠错库的建议
    print(f"\n{'='*80}")
    print("建议补充到 phonetic_candidate.py 的映射规则")
    print(f"{'='*80}")

    # 筛选适合补充的：2-6字词语替换，出现>=2次
    candidates = [(a, c, n) for (a, c), n in pair_counter.most_common()
                  if a and c and 2 <= len(a) <= 8 and 2 <= len(c) <= 8 and n >= 2]

    print(f"\n词语映射候选（出现>=2次，长度2-8字）:")
    for asr_frag, correct_frag, count in candidates:
        print(f'    ("{asr_frag}", "{correct_frag}"),  # {count}次')


if __name__ == "__main__":
    main()
