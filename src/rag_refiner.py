"""RAG增强纠错器.

集成 KnowledgeRetriever 进行领域知识增强的ASR纠错。
在语义精修基础上，动态检索铁路领域知识辅助LLM纠错。
同时集成Agent的TermTool预检索，提供拼音相似术语候选。
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from src.term_tool import TermTool
from src.entity_guard import EntityGuard
from src.knowledge_retriever import KnowledgeRetriever
from src.llm_client import LLMClient
from src.semantic_refiner import SemanticRefineResult


# RAG增强系统提示（与semantic_refiner核心原则一致，增加领域知识使用指引）
RAG_SYSTEM_PROMPT = """你是铁路调度ASR文本纠错专家。前置规则层已完成同音字、近音字的术语级纠错，你将参考检索到的领域知识处理语义级错误。

## 纠错范围（仅限以下三类）
1. **语义近音**：拼音不同但听感相近的词被ASR混淆（如"消极"→"销记"、"反馈"→"反位"、"扩张"→"故障"）
2. **无意义插入**：ASR插入的冗余词或语序混乱（如"松引"等插入词应删除）
3. **术语组合错误**：多个词组合后语义不通（如"控制台任务表示"→"控制台仍无表示"）

## 如何使用领域知识
- 领域知识中的"术语映射"是已验证的标准对应关系，可信赖
- 优先使用领域知识中的映射，而非自己的判断
- 如果领域知识中没有相关映射，则仅做明显的语法纠错，不做语义猜测

## 如何使用术语工具查询结果
- 高置信：拼音几乎相同，强烈建议采用
- 中置信：拼音相近，结合上下文判断是否采用
- 低置信：仅供参考，通常不采用

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


class RAGRefiner:
    """RAG增强纠错器，基于领域知识检索辅助LLM纠错."""

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        retriever: Optional[KnowledgeRetriever] = None,
        system_prompt: Optional[str] = None,
    ):
        """初始化RAG纠错器.

        Args:
            client: LLM客户端实例，默认新建
            retriever: 知识检索器实例，默认新建
            system_prompt: 自定义系统提示，默认使用RAG增强提示
        """
        self.client = client or LLMClient()
        self.retriever = retriever or KnowledgeRetriever()
        self.guard = EntityGuard()
        self.system_prompt = system_prompt or self._load_versioned_system_prompt()
        self.term_tool = TermTool(self.retriever)

    def _load_versioned_system_prompt(self) -> str:
        """根据 LLM_PROMPT_VERSION 环境变量加载对应版本的 system prompt."""
        version = os.getenv("LLM_PROMPT_VERSION", "v1")
        if version == "v1":
            return RAG_SYSTEM_PROMPT
        prompts_dir = Path(__file__).parent / "prompts"
        system_path = prompts_dir / version / "system.txt"
        if system_path.exists():
            try:
                return system_path.read_text(encoding="utf-8").strip()
            except Exception:
                pass
        return RAG_SYSTEM_PROMPT

    def process(
        self,
        original_text: str,
        layer3_text: str,
        changes_history: List[dict],
    ) -> SemanticRefineResult:
        """执行RAG增强纠错.

        Args:
            original_text: ASR原始输出文本
            layer3_text: 前置纠错（Layer 3）的输出文本
            changes_history: 已确认的修改历史

        Returns:
            SemanticRefineResult: 纠错结果
        """
        # 1. 检索领域知识
        knowledge = self.retriever.retrieve(layer3_text)

        # 1.5 预检索：拼音相似术语工具查询
        tool_hints = self._precompute_tool_hints(layer3_text)

        # 2. 构造RAG增强Prompt
        user_prompt = self._build_rag_prompt(
            original_text, layer3_text, changes_history, knowledge, tool_hints
        )

        # 3. 调用LLM纠错
        try:
            llm_output = self.client.complete(self.system_prompt, user_prompt)
        except Exception as e:
            print(f"[RAG-LLM] 调用失败: {type(e).__name__}: {e}", file=sys.stderr)
            return SemanticRefineResult(
                text=layer3_text,
                original=original_text,
                changes=[],
                llm_raw=None,
                guard_passed=False,
            )

        # 4. 清洗输出
        cleaned = self._clean_output(llm_output)

        # 5. 实体校验
        passed, reason = self.guard.validate(layer3_text, cleaned)
        if not passed:
            print(f"[RAG-Guard] 实体校验失败: {reason}", file=sys.stderr)
            return SemanticRefineResult(
                text=layer3_text,
                original=original_text,
                changes=[],
                llm_raw=cleaned,
                guard_passed=False,
            )

        # 6. 计算变更
        changes = self._diff(layer3_text, cleaned)

        return SemanticRefineResult(
            text=cleaned,
            original=original_text,
            changes=changes,
            llm_raw=cleaned,
            guard_passed=True,
        )

    def _build_rag_prompt(
        self,
        original_text: str,
        layer3_text: str,
        changes_history: List[dict],
        knowledge: List[dict],
        tool_hints: str = "",
    ) -> str:
        """构造RAG增强用户Prompt.

        Args:
            original_text: ASR原始输出
            layer3_text: 前置纠错结果
            changes_history: 修改历史
            knowledge: 检索到的领域知识列表
            tool_hints: 术语工具预检索结果

        Returns:
            完整的用户Prompt字符串
        """
        # 构建领域知识参考部分
        knowledge_lines = self._format_knowledge(knowledge)

        if knowledge_lines:
            knowledge_section = f"【领域知识参考】（根据当前文本动态检索）\n{knowledge_lines}\n\n"
        else:
            knowledge_section = ""

        # 构建术语工具hint部分
        if tool_hints:
            tool_section = f"{tool_hints}\n\n"
        else:
            tool_section = ""

        # 构建修改历史部分
        if changes_history:
            changes_lines = "\n".join(
                f"- {c.get('type', '修正')}: {c.get('before', '')} → {c.get('after', '')}"
                for c in changes_history
            )
            changes_section = f"【已确认的修改】\n{changes_lines}\n\n"
        else:
            changes_section = ""

        prompt = f"""{knowledge_section}{tool_section}【ASR原始输出】
{original_text}

【前置纠错结果】
{layer3_text}

{changes_section}请参考以上领域知识和术语工具查询结果，修正ASR识别错误。只输出修正后的文本。"""
        return prompt

    def _precompute_tool_hints(self, text: str) -> str:
        """预计算术语工具查询结果：对文本中的可疑词自动查询拼音相似术语."""
        import jieba
        words = jieba.lcut(text)
        hints = []

        for word in words:
            if len(word) < 2 or len(word) > 4:
                continue
            if word.isdigit() or word in ("，", "。", "、", "的", "了", "是", "在", "和"):
                continue

            results = self.term_tool.lookup(word)
            good_results = [
                r for r in results
                if r["similarity"] >= 0.6 and r["term"] != word
            ]
            if good_results:
                top3 = good_results[:3]
                # 按相似度分级标注置信度
                candidates_parts = []
                for r in top3:
                    if r["similarity"] >= 0.9:
                        tag = "高置信"
                    elif r["similarity"] >= 0.7:
                        tag = "中置信"
                    else:
                        tag = "低置信"
                    candidates_parts.append(f"{r['term']}({tag})")
                candidates = " / ".join(candidates_parts)
                hints.append(f"  {word} → 可能是: {candidates}")

        if hints:
            return "【术语工具查询结果】（以下词可能存在ASR误识别，仅高/中置信建议采用）\n" + "\n".join(hints)
        return ""

    def _format_knowledge(self, knowledge: List[dict]) -> str:
        """将检索到的领域知识格式化为Prompt文本.

        Args:
            knowledge: 检索到的知识列表

        Returns:
            格式化后的知识参考文本
        """
        if not knowledge:
            return ""

        # 按类型分组
        terms: Dict[str, List[str]] = {}  # 正确词 -> [错误词列表]
        patterns: List[str] = []
        scenes: List[str] = []

        for item in knowledge:
            k_type = item.get("type", "")
            content = item.get("content", {})

            if k_type == "term":
                # 术语映射：按正确词分组
                correct = content.get("correct", "")
                error = content.get("error", "")
                pinyin = content.get("pinyin", "")
                if correct not in terms:
                    terms[correct] = []
                if error not in terms[correct]:
                    terms[correct].append(error)
            elif k_type == "error_pattern":
                # 历史错误模式
                pattern = content.get("pattern", "")
                context = content.get("context", "")
                freq = content.get("freq", 0)
                patterns.append(f"- 历史错误：{context}（{pattern}，出现{freq}次）")
            elif k_type == "scene":
                # 场景规则
                scene_name = content.get("scene", "")
                scene_terms = content.get("terms", [])
                terms_str = "、".join(scene_terms)
                # 使用中文引号 chr(0x201C)/chr(0x201D) 包裹场景名
                lq, rq = "\u201c", "\u201d"
                scenes.append(f"- 场景规则：当前文本属于{lq}{scene_name}{rq}场景，标准术语包括：{terms_str}")

        lines: List[str] = []

        # 格式化术语映射
        # 需要重新获取pinyin信息
        term_pinyin: Dict[str, str] = {}
        for item in knowledge:
            if item.get("type") == "term":
                content = item.get("content", {})
                correct = content.get("correct", "")
                pinyin = content.get("pinyin", "")
                if correct and pinyin:
                    term_pinyin[correct] = pinyin

        for correct, errors in terms.items():
            errors_str = "/".join(errors)
            pinyin = term_pinyin.get(correct, "")
            pinyin_part = f"（拼音: {pinyin}）" if pinyin else ""
            lines.append(f"- 术语映射：{errors_str} → {correct}{pinyin_part}")

        # 添加历史错误模式
        lines.extend(patterns)

        # 添加场景规则
        lines.extend(scenes)

        return "\n".join(lines)

    def _clean_output(self, text: str) -> str:
        """清洗LLM输出，去除markdown代码块、引号等.

        Args:
            text: LLM原始输出

        Returns:
            清洗后的文本
        """
        text = text.strip()
        # 去除markdown代码块标记
        if text.startswith("```"):
            first_newline = text.find("\n")
            if first_newline != -1:
                first_line = text[:first_newline].strip()
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
        """计算两段文本的差异，返回变更列表.

        Args:
            before: 修改前文本
            after: 修改后文本

        Returns:
            变更列表
        """
        if before == after:
            return []
        return [{
            "layer": "rag_refine",
            "type": "llm_refine",
            "before": before,
            "after": after,
        }]
