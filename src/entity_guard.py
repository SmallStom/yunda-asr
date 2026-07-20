"""实体安全校验模块.

防止 LLM 篡改关键数字实体（车次号、股道号、道岔号、公里标、速度等）。
"""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Entity:
    type: str       # "train" | "switch" | "track" | "kilometer" | "speed"
    text: str       # 完整匹配文本
    value: str      # 核心数字/标识
    start: int
    end: int


class EntityGuard:
    """实体一致性校验器."""

    PATTERNS = {
        "train": re.compile(r"[GDKTZC]\d+次"),
        "switch": re.compile(r"\d+号道岔"),
        "track": re.compile(r"\d+道(?![岔路])"),  # 避免匹配 "道岔"
        "kilometer": re.compile(r"[Kk]\d+\+\d{3}"),
        "speed": re.compile(r"限速\d+km/h"),
        # 新增：时间表达
        "time": re.compile(r"\d+[点时:：]\d+(分)?"),
    }

    # 各类型实体的数字提取正则
    VALUE_PATTERNS = {
        "train": re.compile(r"[GDKTZC](\d+)次"),
        "switch": re.compile(r"(\d+)号道岔"),
        "track": re.compile(r"(\d+)道"),
        "kilometer": re.compile(r"[Kk](\d+)\+(\d{3})"),
        "speed": re.compile(r"限速(\d+)km/h"),
        "time": re.compile(r"(\d+[点时:：]\d+)"),
    }

    def extract(self, text: str) -> List[Entity]:
        """从文本中提取所有受保护实体."""
        entities = []
        for entity_type, pattern in self.PATTERNS.items():
            for m in pattern.finditer(text):
                value = self._extract_value(entity_type, m.group(0))
                entities.append(Entity(
                    type=entity_type,
                    text=m.group(0),
                    value=value,
                    start=m.start(),
                    end=m.end(),
                ))
        return entities

    def _extract_value(self, entity_type: str, text: str) -> str:
        """提取实体的核心值."""
        pat = self.VALUE_PATTERNS.get(entity_type)
        if pat:
            m = pat.search(text)
            if m:
                return "".join(m.groups())
        return text

    def validate(self, before: str, after: str) -> tuple[bool, Optional[str]]:
        """校验 LLM 输出是否篡改了实体.

        返回: (是否通过, 失败原因)
        规则：
        1. before 中存在的实体，after 中必须存在且值相同
        2. after 长度不得小于 before 的 50%（防止过度删减）
        """
        # 规则1：实体一致性（按 type+value 做联合键，避免不同类型同值实体互相覆盖）
        before_entities = self.extract(before)
        after_entities = self.extract(after)
        after_entity_map: dict[tuple[str, str], Entity] = {(e.type, e.value): e for e in after_entities}

        for be in before_entities:
            key = (be.type, be.value)
            if key not in after_entity_map:
                return False, f"实体丢失或被篡改: {be.text} (类型: {be.type})"
            ae = after_entity_map[key]
            if ae.type != be.type:
                return False, f"实体类型变化: {be.text} -> {ae.text}"

        # 规则2：长度校验
        if len(after) < len(before) * 0.5:
            return False, f"输出过度删减: {len(before)} -> {len(after)} 字符"

        return True, None
