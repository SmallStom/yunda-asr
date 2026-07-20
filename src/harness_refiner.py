"""Harness纠错器：多策略竞争 + 裁判选择.

方案三：并行运行多个纠错策略，由裁判LLM选择最优结果。
- 策略A：保守策略（RAG纠错）
- 策略B：激进策略（RAG+hint纠错）
- 裁判：比较两个结果，选择更优的或融合

通过竞争机制自动选择最优策略，不需要人工指定模式。
"""

import sys
from typing import List, Optional

from src.entity_guard import EntityGuard
from src.knowledge_retriever import KnowledgeRetriever
from src.llm_client import LLMClient
from src.semantic_refiner import SemanticRefineResult


# 裁判提示词
JUDGE_PROMPT = """你是铁路调度ASR文本纠错裁判。有两个纠错器给出了不同的结果，你需要选择更优的一个。

【ASR原始输出】
{original}

【规则层纠错结果】
{rule_text}

【纠错器A结果】
{result_a}

【纠错器B结果】
{result_b}

【术语工具查询结果】
{tool_hints}

选择标准：
1. 数字一致性：不得修改车次号、道岔号、股道号、时间等数字
2. 术语准确性：修改后的词应在术语工具查询结果中有对应候选
3. 最小修改：在修正错误的前提下，修改越少越好
4. 语义通顺：修改后文本语义应通顺合理

判断规则：
- 如果A和B相同，直接选A
- 如果其中一个有数字修改，选另一个
- 如果其中一个的修改在术语工具有候选支撑，选那个
- 如果都不确定，选修改更少的那个（更保守）

输出格式：
- 只输出"A"或"B"（选择哪个纠错器的结果）
- 不要输出其他任何内容"""


class HarnessRefiner:
    """Harness纠错器：多策略竞争 + 裁判选择."""

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        retriever: Optional[KnowledgeRetriever] = None,
    ):
        self.client = client or LLMClient()
        self.guard = EntityGuard()
        self.retriever = retriever or KnowledgeRetriever()

        # 延迟导入避免循环依赖
        from src.semantic_refiner import SemanticRefiner
        from src.rag_refiner import RAGRefiner
        from src.term_tool import TermTool

        # 策略A：基线LLM（保守，无额外知识）
        self.strategy_a = SemanticRefiner(client=self.client)
        # 策略B：RAG+hint（激进，有领域知识+工具）
        self.strategy_b = RAGRefiner(client=self.client, retriever=self.retriever)
        self.term_tool = TermTool(self.retriever)

    def process(
        self,
        original_text: str,
        layer3_text: str,
        changes_history: List[dict],
    ) -> SemanticRefineResult:
        """执行Harness多策略竞争纠错."""
        # 1. 策略A：基线LLM纠错
        result_a = self.strategy_a.process(
            original_text, layer3_text, changes_history
        )

        # 2. 策略B：RAG+hint纠错
        result_b = self.strategy_b.process(
            original_text, layer3_text, changes_history
        )

        text_a = result_a.text
        text_b = result_b.text

        # 3. 如果两个结果相同，直接返回
        if text_a == text_b:
            return result_b  # RAG结果通常更可信

        # 4. 裁判选择
        tool_hints = self.strategy_b._precompute_tool_hints(layer3_text)
        winner = self._judge(original_text, layer3_text, text_a, text_b, tool_hints)

        if winner == "A":
            chosen = text_a
        else:
            chosen = text_b

        # 5. 实体校验
        passed, reason = self.guard.validate(layer3_text, chosen)
        if not passed:
            return SemanticRefineResult(
                text=layer3_text, original=original_text, changes=[],
                llm_raw=chosen, guard_passed=False,
            )

        changes = self._diff(layer3_text, chosen)
        return SemanticRefineResult(
            text=chosen, original=original_text, changes=changes,
            llm_raw=chosen, guard_passed=True,
        )

    def _judge(
        self, original: str, rule_text: str, text_a: str, text_b: str, tool_hints: str
    ) -> str:
        """裁判LLM选择更优结果."""
        prompt = JUDGE_PROMPT.format(
            original=original,
            rule_text=rule_text,
            result_a=text_a,
            result_b=text_b,
            tool_hints=tool_hints or "（无）",
        )
        try:
            response = self.client.complete(
                "你是铁路调度ASR文本纠错裁判。只输出A或B，不解释。", prompt
            )
        except Exception as e:
            print(f"[Harness] 裁判失败: {e}", file=sys.stderr)
            return "B"  # 默认选RAG结果

        response = response.strip().upper()
        if "A" in response[:5]:
            return "A"
        return "B"

    def _diff(self, before: str, after: str) -> List[dict]:
        if before == after:
            return []
        return [{"layer": "harness", "type": "llm_refine", "before": before, "after": after}]
