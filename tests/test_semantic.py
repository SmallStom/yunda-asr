"""Layer 4 语义精修单元测试."""

import pytest

from src.entity_guard import EntityGuard
from src.semantic_refiner import SemanticRefiner, build_user_prompt


class MockLLMClient:
    """Mock LLM 客户端，用于测试."""

    def __init__(self, response: str):
        self.response = response

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self.response


class TestEntityGuard:
    def test_extract_train_number(self):
        """测试提取车次号实体."""
        guard = EntityGuard()
        entities = guard.extract("G1023次列车预告")
        assert len(entities) == 1
        assert entities[0].type == "train"
        assert entities[0].value == "1023"

    def test_extract_switch(self):
        """测试提取道岔号实体."""
        guard = EntityGuard()
        entities = guard.extract("18号道岔开通反位")
        assert len(entities) == 1
        assert entities[0].type == "switch"
        assert entities[0].value == "18"

    def test_validate_pass(self):
        """测试校验通过."""
        guard = EntityGuard()
        passed, reason = guard.validate("18号道岔开通反位", "18号道岔开通反位。")
        assert passed is True
        assert reason is None

    def test_validate_rejects_number_change(self):
        """测试校验拦截数字篡改."""
        guard = EntityGuard()
        passed, reason = guard.validate("18号道岔开通反位", "8号道岔开通反位。")
        assert passed is False
        assert "实体丢失" in reason

    def test_validate_rejects_deletion(self):
        """测试校验拦截过度删减或实体丢失."""
        guard = EntityGuard()
        passed, reason = guard.validate("18号道岔开通反位", "道岔。")
        assert passed is False
        # 可能因实体丢失或长度删减而失败
        assert "实体丢失" in reason or "删减" in reason


class TestSemanticRefiner:
    def test_basic_correction(self):
        """测试正常语义修正."""
        mock = MockLLMClient("18号道岔开通反位，信号好了。")
        refiner = SemanticRefiner(client=mock)
        result = refiner.process(
            "那个十八号道岔好了信号也好了",
            "18号道岔开通反位，信号好了",
            [],
        )
        assert "18号道岔" in result.text
        assert result.guard_passed is True

    def test_guard_rejects_number_change(self):
        """测试实体校验拦截数字篡改."""
        mock = MockLLMClient("8号道岔开通反位。")
        refiner = SemanticRefiner(client=mock)
        result = refiner.process("十八号道岔", "18号道岔", [])
        assert result.text == "18号道岔"  # 回退到 Layer 3
        assert result.guard_passed is False

    def test_guard_rejects_deletion(self):
        """测试实体校验拦截过度删减."""
        mock = MockLLMClient("道岔。")
        refiner = SemanticRefiner(client=mock)
        result = refiner.process("18号道岔开通反位", "18号道岔开通反位", [])
        assert result.text == "18号道岔开通反位"
        assert result.guard_passed is False

    def test_llm_failure_fallback(self):
        """测试 LLM 调用失败时降级."""
        class FailingClient:
            def complete(self, *args, **kwargs):
                raise RuntimeError("连接失败")

        refiner = SemanticRefiner(client=FailingClient())
        result = refiner.process("原文", "layer3结果", [])
        assert result.text == "layer3结果"
        assert result.guard_passed is False

    def test_clean_output_markdown(self):
        """测试清洗 markdown 代码块."""
        mock = MockLLMClient('```text\n18号道岔开通反位。\n```')
        refiner = SemanticRefiner(client=mock)
        result = refiner.process("", "18号道岔开通反位", [])
        assert "```" not in result.text
        assert result.text == "18号道岔开通反位。"

    def test_build_user_prompt(self):
        """测试 Prompt 构造."""
        prompt = build_user_prompt("原文", "layer3结果", [
            {"type": "alias_replace", "before": "道差", "after": "道岔"},
        ])
        assert "原文" in prompt
        assert "layer3结果" in prompt
        assert "道差" in prompt
        assert "道岔" in prompt

    def test_pipeline_semantic_integration(self):
        """测试 Layer 4 在流水线中的集成（Mock 模式）."""
        from src.pipeline import PostCorrectionPipeline

        # Mock 返回与 Layer 3 不同的结果，确保 semantic 层被标记为已应用
        mock_refiner = SemanticRefiner(client=MockLLMClient("18号道岔开通反位，信号好了。"))
        pipeline = PostCorrectionPipeline(semantic_refiner=mock_refiner)

        result = pipeline.run("十八号道差开通反位", layers=[1, 2, 3], enable_semantic=True)
        assert "18号道岔" in result.corrected
        assert "semantic" in result.layers_applied
        assert "layer4" in result.layer_outputs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
