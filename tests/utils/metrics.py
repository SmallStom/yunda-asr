"""评估指标工具函数.

提供CER计算、术语准确率、实体保真率等评估指标.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 使用 rapidfuzz 加速编辑距离计算（C实现，比纯Python快10-50倍）
try:
    from rapidfuzz.distance import Levenshtein
    USE_RAPIDFUZZ = True
except ImportError:
    USE_RAPIDFUZZ = False


def edit_distance(s1: str, s2: str) -> int:
    """计算两个字符串的编辑距离（Levenshtein距离）."""
    if USE_RAPIDFUZZ:
        return Levenshtein.distance(s1, s2)
    
    # 回退到纯Python实现
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def cer(hypothesis: str, reference: str) -> float:
    """计算字符错误率（Character Error Rate）.

    CER = 编辑距离 / max(len(hyp), len(ref))
    返回0.0-1.0之间的小数，0表示完全正确.
    """
    if not reference and not hypothesis:
        return 0.0
    if not reference:
        return 1.0
    dist = edit_distance(hypothesis, reference)
    max_len = max(len(hypothesis), len(reference))
    return dist / max_len


def extract_railway_terms(text: str) -> List[str]:
    """从文本中提取标准铁路术语（简单正则匹配）.

    实际生产环境应从railway_terms.json加载术语列表.
    """
    # 核心术语列表（硬编码常用术语，避免循环导入）
    core_terms = [
        "道岔", "信号机", "进路", "闭塞", "闭塞分区",
        "预告", "接车", "发车", "调车", "通过",
        "定位", "反位", "开放", "关闭", "点灯", "灭灯",
        "股道", "咽喉区", "站界", "区间", "限速",
        "列车", "车次", "调度", "扳道员", "值班员",
        "无异常", "空闲", "占用", "锁闭", "解锁",
        "加锁", "单锁", "故障", "好了", "明白",
    ]
    found = []
    for term in core_terms:
        if term in text:
            found.append(term)
    return found


def term_accuracy(hypothesis: str, reference: str) -> Tuple[float, int, int]:
    """计算术语准确率.

    返回: (准确率, 命中数, 参考术语总数)
    """
    ref_terms = set(extract_railway_terms(reference))
    hyp_terms = set(extract_railway_terms(hypothesis))

    if not ref_terms:
        return 1.0, 0, 0

    hits = len(ref_terms & hyp_terms)
    return hits / len(ref_terms), hits, len(ref_terms)


# 数字实体提取正则（与EntityGuard保持一致）
_ENTITY_PATTERNS = {
    "train": re.compile(r"[GDKTZC]\d+次"),
    "switch": re.compile(r"\d+号道岔"),
    "track": re.compile(r"\d+道(?![岔路])"),
    "kilometer": re.compile(r"[Kk]\d+\+\d{3}"),
    "speed": re.compile(r"限速\d+km/h"),
}


def extract_entities(text: str) -> Dict[str, List[str]]:
    """从文本中提取各类数字实体."""
    entities: Dict[str, List[str]] = {}
    for entity_type, pattern in _ENTITY_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            entities[entity_type] = matches
    return entities


def entity_fidelity(hypothesis: str, reference: str) -> Tuple[float, int, int]:
    """计算数字实体保真率.

    返回: (保真率, 命中数, 参考实体总数)
    """
    ref_entities = extract_entities(reference)
    hyp_entities = extract_entities(hypothesis)

    total_ref = 0
    hits = 0

    for entity_type, ref_list in ref_entities.items():
        total_ref += len(ref_list)
        hyp_list = hyp_entities.get(entity_type, [])
        for e in ref_list:
            if e in hyp_list:
                hits += 1

    if total_ref == 0:
        return 1.0, 0, 0

    return hits / total_ref, hits, total_ref


def is_valid_asr_text(text: str) -> bool:
    """判断ASR输出是否为有效普通话中文文本（过滤Silence、方言、外语等）."""
    if not text or not text.strip():
        return False
    stripped = text.strip()
    # 过滤已知无效标记
    invalid_markers = ["[Silence]", "[Unintelligible Speech]", "[No Speech]"]
    for marker in invalid_markers:
        if marker in stripped:
            return False
    
    # 过滤方言特征字符（粤语、闽南语等）
    dialect_chars = [
        "喺", "嘅", "唔", "咁", "乜", "佢", "冇", "咗", "嚟", "嘞",  # 粤语
        "阮", "咱", "俺",  # 方言代词
    ]
    for char in dialect_chars:
        if char in stripped:
            return False
    
    # 过滤外语字符（泰语、俄语、韩语等非中文Unicode块）
    foreign_ranges = [
        (0x0E00, 0x0E7F),   # 泰语
        (0x0400, 0x04FF),   # 俄语/西里尔字母
        (0x1100, 0x11FF),   # 韩语
        (0x3130, 0x318F),   # 韩语兼容字母
        (0xAC00, 0xD7AF),   # 韩语音节
        (0x0600, 0x06FF),   # 阿拉伯语
    ]
    for char in stripped:
        code = ord(char)
        for start, end in foreign_ranges:
            if start <= code <= end:
                return False
    
    # 至少包含一些中文字符，且中文字符占比超过50%
    chinese_chars = re.findall(r'[\u4e00-\u9fa5]', stripped)
    if len(chinese_chars) < 2:
        return False
    # 计算中文字符占比（排除标点和空格）
    meaningful_chars = re.findall(r'[\u4e00-\u9fa5A-Za-z0-9]', stripped)
    if meaningful_chars and len(chinese_chars) / len(meaningful_chars) < 0.5:
        return False
    
    return True


def load_asr_test_pairs(path: Optional[Path] = None, dataset: str = "full") -> List[dict]:
    """加载ASR测试对，过滤无效样本.

    Args:
        path: 指定JSONL文件路径。如果为None，根据dataset参数选择默认路径。
        dataset: 数据集类型，"full"=短文本，"long"=长文本，"long_hotwords"=长文本(热词).
    """
    if path is None:
        base_dir = Path(__file__).parent.parent.parent / "data" / "asr_testset"
        if dataset == "long":
            path = base_dir / "asr_test_pairs_long.jsonl"
        elif dataset == "long_hotwords":
            path = base_dir / "asr_test_pairs_long_hotwords.jsonl"
        else:
            path = base_dir / "asr_test_pairs.jsonl"

    records = []
    if not path.exists():
        return records

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            asr_text = record.get("asr", "")
            if is_valid_asr_text(asr_text):
                records.append(record)

    return records
