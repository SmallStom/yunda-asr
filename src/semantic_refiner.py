"""Layer 4: 语义精修层.

基于本地 LLM 的跨句段语义一致性修正与格式规范化。
包含 Prompt 工程、LLM 调用、输出清洗、实体安全校验。
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.entity_guard import EntityGuard
from src.llm_client import LLMClient


# 默认System Prompt（硬编码回退，与prompts/v1/system.txt保持同步）
DEFAULT_SYSTEM_PROMPT = """你是铁路调度ASR文本纠错专家。前置规则层已完成同音字、近音字的术语级纠错，你的任务是处理规则层无法覆盖的语义级错误。

## 纠错范围（仅限以下三类）
1. **语义近音**：拼音不同但听感相近的词被ASR混淆（如"消极"→"销记"、"反馈"→"反位"、"扩张"→"故障"）
2. **无意义插入**：ASR插入的冗余词或语序混乱（如"松引"等插入词应删除）
3. **术语组合错误**：多个词组合后语义不通（如"控制台任务表示"→"控制台仍无表示"）

## 严禁操作
1. **不改数字**：车次号、道岔号、股道号、设备号、时间、速度中的数字原样保留
2. **不增删标点**：不添加逗号、不删除现有标点、不补充书名号《》、仅在句末补全句号
3. **不合并/拆分词语**：如"二号"不改为"2"，"48615次"不改为"248615次"
4. **不添加信息**：不得添加原文没有的内容，不得删除有意义的原文内容
5. **不改写正确句子**：语法正确、语义通顺的句子保持原样
6. **不过度纠正**：修正时用最简改法，不画蛇添足（如"5表示"改为"无表示"即可，不要改成"仍无表示"）
7. **不上下文复制**：不得用前文出现过的短语替换后文内容

## 判断流程
- 先判断前置纠错结果是否已正确，若正确则原样输出
- 仅在有明确错误时才修改，宁可少改不可多改
- 对不确定的地方保持原样

## 输出格式
直接输出修正后的文本，不解释，不使用markdown格式。"""

DEFAULT_USER_TEMPLATE = """请修正以下铁路调度ASR文本中的语义级错误。

【ASR原始输出】
{original_text}

【规则层纠错结果】
{layer3_text}
{changes_section}
## 任务要求
1. 规则层已完成同音字纠错，你只需处理规则层遗漏的语义错误
2. 如果规则层结果已正确，原样输出，不要强行修改
3. 只输出修正后的文本，不解释"""


class PromptLoader:
    """Prompt版本管理器."""

    PROMPTS_DIR = Path(__file__).parent / "prompts"
    REGISTRY_FILE = PROMPTS_DIR / "registry.json"

    @classmethod
    def load(cls) -> tuple[str, str]:
        """加载当前版本的system prompt和user template.

        返回: (system_prompt, user_template)
        """
        # 优先使用 PromptManager（支持运行时切换）
        try:
            from src.prompt_manager import get_prompt_manager
            from src.config import get_settings

            manager = get_prompt_manager()
            settings = get_settings()
            version = settings.llm_prompt_version
            prompt_info = manager.get(version)
            if prompt_info:
                return prompt_info["system"], prompt_info["user_template"]
        except Exception:
            pass

        version = os.getenv("LLM_PROMPT_VERSION", "v2")

        # 尝试从文件加载
        if cls.REGISTRY_FILE.exists():
            try:
                with open(cls.REGISTRY_FILE, "r", encoding="utf-8") as f:
                    registry = json.load(f)
                # 环境变量优先；若未设置则使用registry默认值
                if "LLM_PROMPT_VERSION" not in os.environ:
                    version = registry.get("default", version)
                versions = registry.get("versions", {})
                if version in versions:
                    vinfo = versions[version]
                    system_path = cls.PROMPTS_DIR / vinfo.get("system", "")
                    template_path = cls.PROMPTS_DIR / vinfo.get("user_template", "")

                    system_prompt = DEFAULT_SYSTEM_PROMPT
                    if system_path.exists():
                        with open(system_path, "r", encoding="utf-8") as f:
                            system_prompt = f.read().strip()

                    user_template = DEFAULT_USER_TEMPLATE
                    if template_path.exists():
                        with open(template_path, "r", encoding="utf-8") as f:
                            user_template = f.read().strip()

                    return system_prompt, user_template
            except Exception:
                pass

        return DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_TEMPLATE


# 模块级加载（带安全回退，避免文件异常导致整个模块导入失败）
try:
    SYSTEM_PROMPT, USER_TEMPLATE = PromptLoader.load()
except Exception:
    SYSTEM_PROMPT, USER_TEMPLATE = DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_TEMPLATE


def build_user_prompt(original_text: str, layer3_text: str, changes_history: List[dict], user_template: str = None) -> str:
    """构造用户 Prompt."""
    if user_template is None:
        # 每次构造时重新加载，支持运行时切换
        _, user_template = PromptLoader.load()
    if changes_history:
        changes_lines = "\n".join(
            f"- {c.get('type', '修正')}: {c.get('before', '')} → {c.get('after', '')}"
            for c in changes_history
        )
        changes_section = f"【已确认的修改】\n{changes_lines}\n\n"
    else:
        changes_section = ""

    return user_template.format(
        original_text=original_text,
        layer3_text=layer3_text,
        changes_section=changes_section,
    )


@dataclass
class SemanticRefineResult:
    text: str
    original: str
    changes: List[dict] = field(default_factory=list)
    llm_raw: Optional[str] = None
    guard_passed: bool = True


class SemanticRefiner:
    """语义精修器."""

    def __init__(self, client: Optional[LLMClient] = None, system_prompt: Optional[str] = None):
        self.client = client or LLMClient()
        self.guard = EntityGuard()
        # 使用传入的 prompt；否则每次重新加载，支持运行时切换
        self._fixed_system_prompt = system_prompt

    @property
    def system_prompt(self) -> str:
        if self._fixed_system_prompt:
            return self._fixed_system_prompt
        prompt, _ = PromptLoader.load()
        return prompt

    def process(
        self,
        original_text: str,
        layer3_text: str,
        changes_history: List[dict],
    ) -> SemanticRefineResult:
        """执行语义精修."""
        # 1. 构造 Prompt
        prompt = build_user_prompt(original_text, layer3_text, changes_history)

        # 2. 调用 LLM
        try:
            llm_output = self.client.complete(self.system_prompt, prompt)
        except Exception as e:
            import sys
            print(f"[LLM] 调用失败: {type(e).__name__}: {e}", file=sys.stderr)
            return SemanticRefineResult(
                text=layer3_text,
                original=original_text,
                changes=[],
                llm_raw=None,
                guard_passed=False,
            )

        # 3. 清洗输出
        cleaned = self._clean_output(llm_output)

        # 4. 实体校验
        passed, reason = self.guard.validate(layer3_text, cleaned)
        if not passed:
            return SemanticRefineResult(
                text=layer3_text,
                original=original_text,
                changes=[],
                llm_raw=cleaned,
                guard_passed=False,
            )

        # 4b. 重复片段检测：防止LLM上下文复制幻觉
        if self._has_repeat_injection(layer3_text, cleaned):
            return SemanticRefineResult(
                text=layer3_text,
                original=original_text,
                changes=[],
                llm_raw=cleaned,
                guard_passed=False,
            )

        # 5. 计算变更
        changes = self._diff(layer3_text, cleaned)

        return SemanticRefineResult(
            text=cleaned,
            original=original_text,
            changes=changes,
            llm_raw=cleaned,
            guard_passed=True,
        )

    def _clean_output(self, text: str) -> str:
        """清洗 LLM 输出，去除可能的 markdown 代码块、引号等."""
        text = text.strip()
        # 去除 markdown 代码块标记（支持带语言标签：```python, ```text 等）
        if text.startswith("```"):
            # 去掉开头的 ``` 及可能的语言标签
            first_newline = text.find("\n")
            if first_newline != -1:
                first_line = text[:first_newline].strip()
                # 确认第一行是代码块标记（``` 或 ```language）
                if first_line.startswith("```"):
                    text = text[first_newline + 1:]
                else:
                    text = text[3:]
            else:
                text = text[3:]
        # 去除结尾的 ```
        text = text.rstrip()
        if text.endswith("```"):
            text = text[:-3].rstrip()
        # 去除首尾引号（支持中英文引号）
        text = text.strip('"\'""''「」')
        return text

    def _diff(self, before: str, after: str) -> List[dict]:
        """计算两段文本的差异，返回变更列表."""
        if before == after:
            return []
        return [{
            "layer": "semantic",
            "type": "llm_refine",
            "before": before,
            "after": after,
        }]

    def _has_repeat_injection(self, original: str, corrected: str) -> bool:
        """检测LLM是否将原文中已出现的长片段复制到了其他位置.

        典型case：原文有"行车设备检查登记簿"，LLM把后文的"治安联防"改成了"行车设备检查登记簿"。
        检测逻辑：如果corrected中存在一个>=5字的片段，在original中出现过>=2次，
        且在original的对应位置不是该片段，则判定为重复注入。
        """
        if len(corrected) <= len(original) + 2:
            return False  # 长度没增加多少，不太可能注入

        # 提取original中所有>=5字的子串，统计出现次数
        from collections import Counter
        min_len = 5
        orig_fragments = Counter()
        for i in range(len(original) - min_len + 1):
            frag = original[i:i + min_len]
            orig_fragments[frag] += 1

        # 检查corrected中是否有original里已出现>=2次的片段被注入到新位置
        # 简化检测：如果corrected中某>=5字片段在original中出现>=1次，
        # 且corrected中该片段出现次数 > original中出现次数，则可能是注入
        corr_fragments = Counter()
        for i in range(len(corrected) - min_len + 1):
            frag = corrected[i:i + min_len]
            corr_fragments[frag] += 1

        for frag, corr_count in corr_fragments.items():
            orig_count = orig_fragments.get(frag, 0)
            if orig_count >= 1 and corr_count > orig_count:
                # corrected中该片段出现次数比original多，说明LLM可能复制了片段
                return True

        return False
