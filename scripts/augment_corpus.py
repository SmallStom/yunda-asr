"""语料扩增与清洗脚本.

从调度日志/录音转写文本中自动提取候选语料，
经规则过滤和去重后合并到现有语料库.

Usage:
    python scripts/augment_corpus.py --input new_texts.txt --output data/corpus/railway_corpus.txt
"""

import argparse
import random
import sys
from pathlib import Path
from typing import List, Set

sys.path.insert(0, str(Path(__file__).parent.parent))


# 核心铁路术语列表（用于过滤）
CORE_TERMS = [
    "道岔", "信号机", "进路", "闭塞", "闭塞分区",
    "预告", "接车", "发车", "调车", "通过",
    "定位", "反位", "开放", "关闭", "点灯", "灭灯",
    "股道", "咽喉区", "站界", "区间", "限速",
    "列车", "车次", "调度", "扳道员", "值班员",
    "无异常", "空闲", "占用", "锁闭", "解锁",
    "加锁", "单锁", "故障", "好了", "明白",
]


def parse_args():
    parser = argparse.ArgumentParser(description="语料扩增与清洗")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="新增语料来源文件（每行一条）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/corpus/railway_corpus.txt"),
        help="合并后的语料输出路径",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=10,
        help="最小长度 (默认: 10)",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=200,
        help="最大长度 (默认: 200)",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.9,
        help="去重编辑距离阈值 (默认: 0.9, 高于此值视为重复)",
    )
    return parser.parse_args()


def load_existing_corpus(path: Path) -> Set[str]:
    """加载现有语料."""
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def filter_text(text: str, min_len: int, max_len: int) -> bool:
    """规则过滤单条语料."""
    if not text:
        return False
    if len(text) < min_len or len(text) > max_len:
        return False
    # 必须包含至少一个铁路术语
    if not any(term in text for term in CORE_TERMS):
        return False
    # 去除纯数字/纯英文片段（简单判断：中文字符占比）
    chinese_chars = [c for c in text if "\u4e00" <= c <= "\u9fff"]
    if len(chinese_chars) < 3:
        return False
    return True


def is_duplicate(text: str, existing: Set[str], threshold: float) -> bool:
    """判断是否与现有语料重复（基于快速预筛选 + 归一化编辑距离）.

    优化策略：
    1. 先用长度差预筛选（长度差>30%直接跳过编辑距离计算）
    2. 用 rapidfuzz 替代纯Python编辑距离（C实现，快10-50倍）
    """
    from rapidfuzz import fuzz

    text_len = len(text)
    for existing_text in existing:
        if not existing_text:
            continue
        # 长度差过大直接跳过
        if abs(text_len - len(existing_text)) / max(text_len, len(existing_text), 1) > 0.3:
            continue
        # rapidfuzz 的 ratio 返回 0-100 的相似度
        similarity = fuzz.ratio(text, existing_text) / 100.0
        if similarity >= threshold:
            return True
    return False


def main():
    args = parse_args()

    print(f"[INFO] 加载新增语料: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        raw_texts = [line.strip() for line in f if line.strip()]
    print(f"[INFO] 原始条数: {len(raw_texts)}")

    # 加载现有语料
    existing = load_existing_corpus(args.output)
    print(f"[INFO] 现有语料: {len(existing)} 条")

    # 过滤
    filtered = []
    for text in raw_texts:
        if filter_text(text, args.min_len, args.max_len):
            filtered.append(text)
    print(f"[INFO] 规则过滤后: {len(filtered)} 条")

    # 去重
    new_texts = []
    for text in filtered:
        if text in existing:
            continue
        if is_duplicate(text, existing, args.dedup_threshold):
            continue
        new_texts.append(text)
        existing.add(text)  # 避免新增语料内部重复

    print(f"[INFO] 去重后新增: {len(new_texts)} 条")

    if not new_texts:
        print("[INFO] 无新增语料")
        return

    # 合并并写入
    all_texts = list(existing) + new_texts
    random.shuffle(all_texts)  # 打乱顺序，避免训练时数据分布不均

    with open(args.output, "w", encoding="utf-8") as f:
        for text in all_texts:
            f.write(text + "\n")

    print(f"[INFO] 语料已更新: {args.output} (总计 {len(all_texts)} 条)")

    # 抽检报告
    sample_size = max(1, int(len(new_texts) * 0.05))
    sample = random.sample(new_texts, min(sample_size, len(new_texts)))
    print(f"\n[INFO] 随机抽检 {len(sample)} 条新增语料:")
    for i, text in enumerate(sample, 1):
        print(f"  {i}. {text[:60]}{'...' if len(text) > 60 else ''}")


if __name__ == "__main__":
    main()
