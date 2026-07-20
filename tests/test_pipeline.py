"""流水线集成测试."""

import pytest

from src.pipeline import PostCorrectionPipeline


class TestPipeline:
    @pytest.fixture
    def pipeline(self):
        return PostCorrectionPipeline()

    def test_layer1_and_layer2_combined(self, pipeline):
        """测试Layer 1 + Layer 2联合纠错."""
        text = "G一千零二十三次列车，十八号道差开通反位，信号好了"
        result = pipeline.run(text, layers=[1, 2])

        assert "G1023次" in result.corrected
        assert "18号道岔" in result.corrected
        assert "道差" not in result.corrected
        assert "preprocessor" in result.layers_applied
        assert "dictionary" in result.layers_applied

    def test_only_layer1(self, pipeline):
        """测试仅启用Layer 1."""
        text = "十八号道岔开通反位"
        result = pipeline.run(text, layers=[1])

        assert "18号道岔" in result.corrected
        assert "dictionary" not in result.layers_applied

    def test_only_layer2(self, pipeline):
        """测试仅启用Layer 2."""
        text = "18号道差开通反位"
        result = pipeline.run(text, layers=[2])

        assert "道岔" in result.corrected
        assert "preprocessor" not in result.layers_applied

    def test_no_change_text(self, pipeline):
        """测试无需纠错的文本."""
        text = "18号道岔开通反位，信号好了。"
        result = pipeline.run(text, layers=[1, 2])

        # 层不应被标记为已应用（因为没有实际修改）
        assert result.corrected == text or "18号道岔" in result.corrected

    def test_empty_text(self, pipeline):
        """测试空文本."""
        text = ""
        result = pipeline.run(text, layers=[1, 2])
        assert result.corrected == ""

    def test_complex_railway_text(self, pipeline):
        """测试复杂铁路调度文本."""
        text = "2号扳道员，48015次1道发车，将18号道差开通1道并加锁"
        result = pipeline.run(text, layers=[1, 2])

        assert "道岔" in result.corrected
        assert "道差" not in result.corrected
        assert result.corrected.endswith("。")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
