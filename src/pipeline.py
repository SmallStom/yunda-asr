"""后处理纠错流水线编排器.

按顺序执行 Layer 1 -> Layer 2 -> Layer 3 -> Layer 4（可选）
"""

import functools
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.context_corrector import ContextCorrector
from src.dictionary_corrector import DictionaryCorrector, get_dictionary_corrector
from src.preprocessor import Preprocessor, get_preprocessor

try:
    from src.semantic_refiner import SemanticRefiner
except ImportError:
    SemanticRefiner = None

try:
    from src.rag_refiner import RAGRefiner
except ImportError:
    RAGRefiner = None

try:
    from src.harness_refiner import HarnessRefiner
except ImportError:
    HarnessRefiner = None


@dataclass
class CorrectionDetail:
    layer: str
    changes: List[dict]


@dataclass
class PipelineResult:
    original: str
    corrected: str
    layers_applied: List[str]
    details: List[CorrectionDetail]
    layer_outputs: Dict[str, str] = field(default_factory=dict)


class PostCorrectionPipeline:
    """ASR后处理纠错流水线."""

    def __init__(
        self,
        preprocessor: Optional[Preprocessor] = None,
        dictionary_corrector: Optional[DictionaryCorrector] = None,
        context_corrector: Optional[ContextCorrector] = None,
        semantic_refiner: Optional["SemanticRefiner"] = None,
        cache_size: int = 128,
    ):
        self.preprocessor = preprocessor or get_preprocessor()
        self.dictionary_corrector = dictionary_corrector or get_dictionary_corrector()
        self.context_corrector = context_corrector or ContextCorrector()
        if semantic_refiner is not None:
            self.semantic_refiner = semantic_refiner
        elif SemanticRefiner is not None:
            try:
                self.semantic_refiner = SemanticRefiner()
            except Exception:
                self.semantic_refiner = None
        else:
            self.semantic_refiner = None

        # RAG增强纠错器
        self.rag_refiner = None
        if RAGRefiner is not None:
            try:
                self.rag_refiner = RAGRefiner()
            except Exception:
                self.rag_refiner = None

        # Harness纠错器（多策略竞争 + 裁判）
        self.harness_refiner = None
        if HarnessRefiner is not None:
            try:
                self.harness_refiner = HarnessRefiner()
            except Exception:
                self.harness_refiner = None
        # LRU 缓存：对相同输入复用结果（适合重复出现的标准指令）
        self._cache_size = cache_size
        self._cache: dict[tuple, PipelineResult] = {}
        self._cache_order: list[tuple] = []

    def run(
        self,
        text: str,
        layers: Optional[List[int]] = None,
        enable_semantic: bool = False,
        semantic_mode: str = "baseline",
        nbest_candidates: Optional[List[dict]] = None,
    ) -> PipelineResult:
        """
        执行纠错流水线.

        Args:
            text: ASR原始输出文本
            layers: 指定启用的层号列表，如 [1, 2]。None表示启用全部基础层
            enable_semantic: 是否启用Layer 4语义精修
            semantic_mode: 语义精修模式，可选值：
                "baseline" - 基线模式（原始SemanticRefiner）
                "nbest" - 方向一：N-best候选融合+约束解码
                "rag" - 方向二：RAG增强+动态Few-shot
                "fusion" - 融合模式：RAG知识增强 + N-best约束解码
            nbest_candidates: N-best候选列表（仅semantic_mode="nbest"时使用），
                每项格式 {"text": "...", "score": 0.85}

        Returns:
            PipelineResult: 包含纠错结果和修改详情
        """
        if layers is None:
            layers = [1, 2, 3]

        # 缓存查找
        cache_key = (text, tuple(layers), enable_semantic, semantic_mode)
        if cache_key in self._cache:
            return self._cache[cache_key]

        current = text
        details = []
        applied_layers = []
        layer_outputs = {}

        # Layer 1: 预处理
        if 1 in layers:
            result = self.preprocessor.process(current)
            if result.changes:
                applied_layers.append("preprocessor")
                details.append(CorrectionDetail(
                    layer="preprocessor",
                    changes=result.changes,
                ))
            current = result.text
            layer_outputs["layer1"] = current

        # Layer 2: 词典纠错
        if 2 in layers:
            result = self.dictionary_corrector.process(current)
            if result.changes:
                applied_layers.append("dictionary")
                details.append(CorrectionDetail(
                    layer="dictionary",
                    changes=result.changes,
                ))
            current = result.text
            layer_outputs["layer2"] = current

        # Layer 3: 上下文感知纠错
        if 3 in layers and self.context_corrector is not None:
            result = self.context_corrector.process(current)
            if result.changes:
                applied_layers.append("context")
                details.append(CorrectionDetail(
                    layer="context",
                    changes=result.changes,
                ))
            current = result.text
            layer_outputs["layer3"] = current

        # Layer 4: 语义精修
        if enable_semantic:
            changes_history = [
                c for d in details for c in d.changes
            ]

            # 选择性LLM调用：规则层无修改且文本很短时跳过LLM，减少劣化风险
            rule_made_changes = bool(changes_history)
            is_very_short_text = len(current) <= 8
            if not rule_made_changes and is_very_short_text:
                # 规则层没改且文本短，LLM大概率不会改善反而可能劣化
                layer_outputs["layer4"] = current
            else:
                result = self._run_semantic_layer(
                    text, current, changes_history, semantic_mode, nbest_candidates
                )
                if result and result.changes:
                    applied_layers.append("semantic")
                    details.append(CorrectionDetail(
                        layer="semantic",
                        changes=result.changes,
                    ))
                    current = result.text
                    layer_outputs["layer4"] = current
                elif result:
                    # 即使无变更也更新输出（经过校验的相同文本）
                    current = result.text
                    layer_outputs["layer4"] = current

        result = PipelineResult(
            original=text,
            corrected=current,
            layers_applied=applied_layers,
            details=details,
            layer_outputs=layer_outputs,
        )

        # 写入缓存（LRU 淘汰）
        if len(self._cache) >= self._cache_size and self._cache_order:
            old_key = self._cache_order.pop(0)
            self._cache.pop(old_key, None)
        self._cache[cache_key] = result
        self._cache_order.append(cache_key)

        return result

    def _run_semantic_layer(
        self,
        original_text: str,
        layer3_text: str,
        changes_history: List[dict],
        semantic_mode: str,
        nbest_candidates: Optional[List[dict]] = None,
    ):
        """根据模式选择不同的语义精修器执行Layer 4.

        Args:
            original_text: ASR原始输出
            layer3_text: Layer 3纠错结果
            changes_history: 已确认的修改历史
            semantic_mode: 精修模式
            nbest_candidates: N-best候选（仅nbest模式）

        Returns:
            SemanticRefineResult 或 None（无可用refiner时）
        """
        if semantic_mode == "rag" and self.rag_refiner is not None:
            return self.rag_refiner.process(
                original_text=original_text,
                layer3_text=layer3_text,
                changes_history=changes_history,
            )
        elif semantic_mode == "harness" and self.harness_refiner is not None:
            return self.harness_refiner.process(
                original_text=original_text,
                layer3_text=layer3_text,
                changes_history=changes_history,
            )
        elif self.semantic_refiner is not None:
            # 默认基线模式
            return self.semantic_refiner.process(
                original_text=original_text,
                layer3_text=layer3_text,
                changes_history=changes_history,
            )
        return None

    def run_batch(
        self,
        texts: List[str],
        layers: Optional[List[int]] = None,
        enable_semantic: bool = False,
    ) -> List[PipelineResult]:
        """批量处理多条文本.

        当前为串行处理，但共享已加载的模型和词典实例，
        避免每条文本重复初始化。
        """
        return [
            self.run(text, layers=layers, enable_semantic=enable_semantic)
            for text in texts
        ]

    def warmup(self, sample_text: str = "18号道岔开通反位") -> None:
        """预热流水线，预加载所有模型并触发一次完整推理.

        避免首次调用的冷启动延迟（如jieba分词首次加载词典）。
        """
        try:
            self.run(sample_text, layers=[1, 2, 3])
        except Exception:
            pass

    def reload_aliases(self) -> None:
        """热重载别名映射：刷新词典、上下文、RAG/Harness 各层."""
        self._cache.clear()
        self._cache_order.clear()
        self.dictionary_corrector.reload()
        # context_corrector 使用 phonetic_candidate 单例，单例已被全局重载
        if self.rag_refiner is not None:
            try:
                self.rag_refiner.term_tool.reload()
            except Exception:
                pass
        if self.harness_refiner is not None:
            try:
                self.harness_refiner.term_tool.reload()
            except Exception:
                pass

    def reload_hotwords(self) -> None:
        """热重载热词：刷新 RAG/Harness 的 TermTool 拼音索引."""
        self._cache.clear()
        self._cache_order.clear()
        if self.rag_refiner is not None:
            try:
                self.rag_refiner.term_tool.reload()
            except Exception:
                pass
        if self.harness_refiner is not None:
            try:
                self.harness_refiner.term_tool.reload()
            except Exception:
                pass

    def reload_prompts(self) -> None:
        """热重载 Prompt：刷新 RAG/Harness 的 system prompt.

        SemanticRefiner 通过 property 动态加载，无需手动刷新。
        RAGRefiner 在 init 时缓存了 system_prompt，需要手动刷新。
        """
        self._cache.clear()
        self._cache_order.clear()
        if self.rag_refiner is not None:
            try:
                self.rag_refiner.reload_prompt()
            except Exception:
                pass
        if self.harness_refiner is not None:
            try:
                self.harness_refiner.strategy_b.reload_prompt()
            except Exception:
                pass


# 全局流水线实例
_pipeline = None


def get_pipeline() -> PostCorrectionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = PostCorrectionPipeline()
    return _pipeline
