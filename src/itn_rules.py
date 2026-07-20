"""逆文本规范化(ITN)规则模块.

将ASR输出的口语化文本转换为书面规范格式.
"""

import re
from typing import Callable

import cn2an


class ITNRule:
    """单条ITN规则."""

    def __init__(self, name: str, pattern: str | Callable[[str], str], replacer: str | Callable[[re.Match], str] | None = None):
        self.name = name
        if isinstance(pattern, str):
            self.regex = re.compile(pattern)
            self.replacer = replacer or ""
        else:
            self.regex = None
            self.replacer = pattern

    def apply(self, text: str) -> tuple[str, bool]:
        """应用规则，返回(结果, 是否发生修改)."""
        if self.regex is not None:
            new_text = self.regex.sub(self.replacer, text)
            return new_text, new_text != text
        else:
            new_text = self.replacer(text)
            return new_text, new_text != text


# ==================== 规则定义 ====================

# 1. 中文年份 -> 阿拉伯数字年份
RE_YEAR = re.compile(r"([一二三四五六七八九零〇]{4})年")

# 2. 中文月份/日期 -> 阿拉伯数字
RE_MONTH_DAY = re.compile(r"([一二三四五六七八九十]{1,2})(月|日|号)")

# 3. 百分比
RE_PERCENT = re.compile(r"百分之([一二三四五六七八九十百千万零]+)")

# 4. 车次号：K一千零二十三 -> K1023
RE_TRAIN_NUMBER = re.compile(r"([GDKTZC])\s*([一二三四五六七八九十百千万零]+)\s*次")

# 5. 道岔/信号机编号：十八号道岔 -> 18号道岔
RE_NUMBERED_TERM = re.compile(
    r"([一二三四五六七八九十百千万]+)\s*号?\s*(道岔|信号机|轨道电路|区段|道|线|按钮|计数器|锁闭按钮)"
)

# 5b. 独立的"X号"（后面不跟术语）：二号， -> 2号，
RE_STANDALONE_NUMBER = re.compile(
    r"([一二三四五六七八九十])\s*号(?![道岔信号机轨道区段线按钮计数器锁闭])"
)

# 6. 公里标：一百三十四公里八百米 -> 134km800m 或 K134+800
RE_KILOMETER = re.compile(
    r"([Kk])?\s*([一二三四五六七八九十百千万零]+)\s*公里\s*([一二三四五六七八九十百千万零]+)\s*米"
)

# 7. 时间表达：十点三十分 -> 10:30
RE_TIME = re.compile(r"([一二三四五六七八九十百千万零]+)点([一二三四五六七八九十百千万零]+)分")

# 8. 速度表达：限速八十公里每小时 -> 限速80km/h
RE_SPEED = re.compile(r"限速([一二三四五六七八九十百千万零]+)公里每小时")

# 9. 纯数字序列（如计数器编号）
RE_CN_NUMBER = re.compile(r"([一二三四五六七八九十百千万零]+)")

# 10. 口语化"X点X分"
RE_TIME_X = re.compile(r"X点X分")


# 铁路术语保护列表：这些词前面的"号"不应被替换为"日"
RAILWAY_PROTECTED_TERMS = [
    "道岔", "信号机", "轨道电路", "区段", "道", "线", "按钮", "锁闭按钮", "计数器",
    "锁闭", "加锁", "解锁", "单锁",
]


def _replace_year(text: str) -> str:
    """替换中文年份."""
    def repl(m):
        try:
            num = cn2an.cn2an(m.group(1), "strict")
            return f"{int(num)}年"
        except Exception:
            # 回退：逐字映射（处理二零二五这种逐字读法）
            year_chars = m.group(1)
            mapping = {
                '一': '1', '二': '2', '三': '3', '四': '4',
                '五': '5', '六': '6', '七': '7', '八': '8',
                '九': '9', '零': '0', '〇': '0',
            }
            digits = ''.join(mapping.get(c, c) for c in year_chars)
            if digits.isdigit():
                return f"{digits}年"
            return m.group(0)
    return RE_YEAR.sub(repl, text)


def _replace_month_day(text: str) -> str:
    """替换中文月份/日期.

    铁路场景策略：仅在明确日期上下文（X月X号）中才将"号"转为"日"，
    避免将"X号道岔/信号机"等铁路编号误转为日期。
    """
    def repl(m):
        unit = m.group(2)
        if unit == "号":
            # 检查前面是否有"X月"上下文，只有日期场景才转"日"
            before = text[:m.start()]
            if not re.search(r"[一二三四五六七八九十]{1,2}月$", before):
                # 非日期上下文，保留"号"（可能是铁路编号）
                return m.group(0)
        try:
            num = cn2an.cn2an(m.group(1), "strict")
            if unit == "号":
                unit = "日"
            return f"{int(num)}{unit}"
        except Exception:
            return m.group(0)
    return RE_MONTH_DAY.sub(repl, text)


def _replace_percent(text: str) -> str:
    """替换百分比."""
    def repl(m):
        try:
            num = cn2an.cn2an(m.group(1), "strict")
            return f"{int(num)}%"
        except Exception:
            return m.group(0)
    return RE_PERCENT.sub(repl, text)


def _replace_train_number(text: str) -> str:
    """替换车次号中的中文数字."""
    def repl(m):
        prefix = m.group(1).upper()
        try:
            num = int(cn2an.cn2an(m.group(2), "strict"))
            return f"{prefix}{num}次"
        except Exception:
            return m.group(0)
    return RE_TRAIN_NUMBER.sub(repl, text)


# 逐字读法的中文数字（如"四八六幺五" -> 48615）
RE_DIGIT_BY_DIGIT = re.compile(
    r"([GDKTZC])?\s*([零〇一二三四五六七八九幺]{2,})\s*(次|号|道)"
)

# 逐字数字映射
_DIGIT_MAP = {
    '零': '0', '〇': '0', '一': '1', '二': '2', '三': '3', '四': '4',
    '五': '5', '六': '6', '七': '7', '八': '8', '九': '9', '幺': '1',
    '两': '2',
}


def _replace_digit_by_digit(text: str) -> str:
    """替换逐字读法的中文数字（如"四八六幺五次" -> 48615次）.
    
    ASR常将数字逐字读出，如车次号"48615次"被识别为"四八六幺五次"。
    """
    def repl(m):
        prefix = m.group(1)
        cn_digits = m.group(2)
        suffix = m.group(3)
        
        # 逐字转换
        digits = ''.join(_DIGIT_MAP.get(c, c) for c in cn_digits)
        if not digits.isdigit():
            return m.group(0)
        
        # 如果有前缀（G/D/K/T/Z/C），规范化为大写
        if prefix:
            return f"{prefix.upper()}{digits}{suffix}"
        return f"{digits}{suffix}"
    
    return RE_DIGIT_BY_DIGIT.sub(repl, text)


def _replace_numbered_term(text: str) -> str:
    """替换编号类术语中的中文数字."""
    def repl(m):
        try:
            num = int(cn2an.cn2an(m.group(1), "strict"))
            term = m.group(2)
            full_match = m.group(0)
            if term in ("道", "线"):
                # 如果匹配文本原本包含"号"（如"十八号道"），保留"号"
                # 避免"十八号道差/道岔"被误识别为股道而丢失"号"
                if "号" in full_match:
                    return f"{num}号{term}"
                return f"{num}{term}"
            return f"{num}号{term}"
        except Exception:
            return m.group(0)
    return RE_NUMBERED_TERM.sub(repl, text)


def _replace_standalone_number(text: str) -> str:
    """替换独立的中文数字编号（如"二号，"->"2号，"）."""
    def repl(m):
        try:
            num = int(cn2an.cn2an(m.group(1), "strict"))
            return f"{num}号"
        except Exception:
            return m.group(0)
    return RE_STANDALONE_NUMBER.sub(repl, text)


def _replace_time(text: str) -> str:
    """替换中文时间表达."""
    def repl(m):
        try:
            hour = int(cn2an.cn2an(m.group(1), "strict"))
            minute = int(cn2an.cn2an(m.group(2), "strict"))
            return f"{hour}:{minute:02d}"
        except Exception:
            return m.group(0)
    return RE_TIME.sub(repl, text)


def _replace_speed(text: str) -> str:
    """替换中文速度表达."""
    def repl(m):
        try:
            num = int(cn2an.cn2an(m.group(1), "strict"))
            return f"限速{num}km/h"
        except Exception:
            return m.group(0)
    return RE_SPEED.sub(repl, text)


def _replace_kilometer(text: str) -> str:
    """替换公里标."""
    def repl(m):
        k_prefix = m.group(1) or "K"
        k_prefix = k_prefix.upper()
        try:
            km = int(cn2an.cn2an(m.group(2), "strict"))
            m_val = int(cn2an.cn2an(m.group(3), "strict"))
            return f"{k_prefix}{km}+{m_val:03d}"
        except Exception:
            return m.group(0)
    # 同时处理 "K 一百三十四加八百" 这种变体
    text = RE_KILOMETER.sub(repl, text)
    # 补充变体：一百三十四公里八百米（无K前缀，已有RE_KILOMETER处理）
    # 以及 "K134+800" 这种已是标准格式，无需处理
    return text


def _normalize_train_prefix(text: str) -> str:
    """规范化车次号前缀：g534次 -> G534次."""
    pattern = re.compile(r"([gdktzc])(\d+)(次)")
    return pattern.sub(lambda m: f"{m.group(1).upper()}{m.group(2)}{m.group(3)}", text)


def _fix_time_x(text: str) -> str:
    """保留X点X分标记（实际业务中通常是未知时间占位）."""
    # 不需要转换，保持原样即可
    return text


def _normalize_punctuation(text: str) -> str:
    """轻量标点规范化：去除中文字符之间及其与数字/字母之间的多余空格，保留英文短语间空格."""
    # 去除中文字符之间及其与数字/字母之间的空格
    text = re.sub(r"([\u4e00-\u9fa5])\s+([\u4e00-\u9fa5])", r"\1\2", text)
    text = re.sub(r"([\u4e00-\u9fa5])\s+(\w)", r"\1\2", text)
    text = re.sub(r"(\w)\s+([\u4e00-\u9fa5])", r"\1\2", text)
    # 去除单字母与数字之间的空格（车次号场景：G 1023 -> G1023）
    text = re.sub(r"\b([A-Za-z])\s+(\d)", r"\1\2", text)
    # 去除首尾空格
    return text.strip()


# ASR常见误识别："到"→"道"（股道编号场景，如"一到发车"→"1道发车"）
RE_TRACK_NUMBER = re.compile(
    r"([一二三四五六七八九十])\s*到(?=(?:发车|接车|进路|停车|出站|进站))"
)


def _replace_track_number(text: str) -> str:
    """将"X到+铁路动词"修正为"X道"（ASR将"道"误识别为"到"）."""
    def repl(m):
        try:
            num = int(cn2an.cn2an(m.group(1), "strict"))
            return f"{num}道"
        except Exception:
            return m.group(0)
    return RE_TRACK_NUMBER.sub(repl, text)


# 标点规范化：顿号→逗号、去除句末多余句号
RE_PUNCT_NORMALIZE_1 = re.compile(r"、")  # 顿号→逗号
RE_PUNCT_NORMALIZE_2 = re.compile(r"。{2,}")  # 多个句号→单个
RE_PUNCT_NORMALIZE_3 = re.compile(r"。(?=。)")  # 去重复句号


def _normalize_chinese_punctuation(text: str) -> str:
    """标点规范化：顿号转逗号、去除重复句号."""
    text = RE_PUNCT_NORMALIZE_1.sub("，", text)
    text = RE_PUNCT_NORMALIZE_2.sub("。", text)
    return text


# 规则列表：按顺序执行
ITN_RULES: list[ITNRule] = [
    ITNRule("year", _replace_year),
    ITNRule("month_day", _replace_month_day),
    ITNRule("time", _replace_time),
    ITNRule("percent", _replace_percent),
    ITNRule("speed", _replace_speed),
    ITNRule("kilometer", _replace_kilometer),
    ITNRule("train_number", _replace_train_number),
    ITNRule("digit_by_digit", _replace_digit_by_digit),
    ITNRule("numbered_term", _replace_numbered_term),
    ITNRule("track_number", _replace_track_number),
    ITNRule("standalone_number", _replace_standalone_number),
    ITNRule("train_prefix", _normalize_train_prefix),
    ITNRule("punctuation", _normalize_punctuation),
    ITNRule("cn_punctuation", _normalize_chinese_punctuation),
]
