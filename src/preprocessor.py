"""Layer 1: 文本预处理层.

职责：
    1. 逆文本规范化（ITN）：将口语化数字、时间转为书面格式
    2. 车次号规范化：统一大小写格式
    3. 道岔/信号机编号规范化
    4. 轻量标点补全（基于规则）
"""

import re
from dataclasses import dataclass, field
from typing import List

from src.itn_rules import ITN_RULES


@dataclass
class PreprocessResult:
    text: str
    original: str
    changes: List[dict] = field(default_factory=list)


class Preprocessor:
    """文本预处理器."""

    # 停顿词列表：在这些词后倾向于插入逗号
    PAUSE_WORDS = [
        "值班员", "调度员", "扳道员", "司机", "站长", "干部",
        "工务", "电务", "车站", "列车",
        "好了", "明白", "收到", "执行", "注意",
    ]

    # 句末模式：在这些词后倾向于插入句号
    SENTENCE_ENDS = [
        r"司机明白$",
        r"明白$",
        r"收到$",
        r"好了$",
        r"完毕$",
        r"正常$",
        r"停妥$",
        r"到达$",
        r"出发$",
        r"通过$",
        r"出站$",
        r"已?下道$",
        r"返回安全区域$",
    ]

    def __init__(self):
        self.sentence_end_patterns = [re.compile(p) for p in self.SENTENCE_ENDS]

    def process(self, text: str) -> PreprocessResult:
        """执行预处理."""
        original = text
        changes = []
        current = text

        # Step 1: 应用ITN规则
        for rule in ITN_RULES:
            new_text, modified = rule.apply(current)
            if modified:
                changes.append({
                    "layer": "itn",
                    "rule": rule.name,
                    "before": current,
                    "after": new_text,
                })
            current = new_text

        # Step 2: 轻量标点补全
        punctuated = self._add_punctuation(current)
        if punctuated != current:
            changes.append({
                "layer": "punctuation",
                "rule": "add_punctuation",
                "before": current,
                "after": punctuated,
            })
        current = punctuated

        return PreprocessResult(
            text=current,
            original=original,
            changes=changes,
        )

    def _add_punctuation(self, text: str) -> str:
        """基于规则插入标点符号."""
        if not text.strip():
            return text

        # 如果文本已经包含标点，仅检查是否以句末标点结尾
        if any(p in text for p in "，。；！？、"):
            if not text.endswith((".", "。", "!", "！", "?", "？")):
                return text + "。"
            return text

        # 分段处理：按照一些明显的分隔词切分
        segments = self._split_segments(text)
        result_parts = []

        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            # 检查是否是句末
            if any(p.search(seg) for p in self.sentence_end_patterns):
                if not seg.endswith((".", "。", "!", "！", "?", "？")):
                    seg += "。"
            elif len(seg) > 10:
                # 长句在停顿词后加逗号
                seg = self._insert_commas(seg)
            result_parts.append(seg)

        result = "".join(result_parts)
        if not result.endswith((".", "。", "!", "！", "?", "？")):
            result += "。"
        return result

    def _split_segments(self, text: str) -> List[str]:
        """将长文本按明显停顿切分为句段."""
        # 按角色转换词切分，增加更多铁路调度场景角色
        split_pattern = re.compile(
            r"(?=(报告值班员|车站值班员|列车调度员|内勤|外勤|站长|司机|列车长|乘务员|随车机械师|调度所|指挥中心))"
        )
        segments = split_pattern.split(text)
        # 合并分割结果
        merged = []
        i = 0
        while i < len(segments):
            if segments[i] is None:
                i += 1
                continue
            if i + 1 < len(segments) and segments[i + 1] in (
                "报告值班员", "车站值班员", "列车调度员", "内勤", "外勤", "站长", "司机",
                "列车长", "乘务员", "随车机械师", "调度所", "指挥中心",
            ):
                merged.append(segments[i] + segments[i + 1])
                i += 2
            else:
                merged.append(segments[i])
                i += 1
        return [m for m in merged if m.strip()]

    # 编号类保护模式：避免在编号中间插入逗号
    _NUMBERED_TERM_PROTECT = re.compile(r"\d+号(?:道岔|信号机|按钮|轨道电路|区段)")

    def _insert_commas(self, text: str) -> str:
        """在长句的停顿词后插入逗号."""
        result = text
        for word in self.PAUSE_WORDS:
            # 在停顿词后插入逗号（如果后面还没有标点）
            pattern = re.compile(rf"({re.escape(word)})([^，。；！？、\s])")
            # 逐次替换，每次替换前检查是否落在编号保护区内
            offset = 0
            for m in pattern.finditer(result):
                # 获取当前匹配位置（考虑已插入标点导致的偏移）
                start = m.start(1) + offset
                end = m.end(1) + offset
                # 检查该位置是否在 "X号道岔/信号机" 等编号内部
                # 简单策略：如果 word 前后紧邻编号字符，则跳过
                before = result[max(0, start - 3):start]
                after = result[end:end + 5]
                # 如果后面紧跟 "号道岔"、"号信号机" 等，说明是编号内部，跳过
                if self._NUMBERED_TERM_PROTECT.search(before + word + after):
                    continue
                # 执行替换
                result = result[:end] + "，" + result[end:]
                offset += 1
        return result


# 全局预处理器实例
_preprocessor = None


def get_preprocessor() -> Preprocessor:
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = Preprocessor()
    return _preprocessor
