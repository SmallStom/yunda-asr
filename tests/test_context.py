"""Layer 3 上下文感知纠错单元测试."""

import pytest

from src.context_corrector import ContextCorrector
from src.ngram_model import NgramModel
from src.phonetic_candidate import PhoneticCandidateGenerator


class TestContextCorrector:
    @pytest.fixture
    def corrector(self):
        return ContextCorrector()

    def test_collocation_switch_valid(self, corrector):
        """测试道岔在合法共现语境下不被修改."""
        text = "18号道岔开通反位"
        result = corrector.process(text)
        assert "道岔" in result.text
        # 合法共现，不应有共现规则触发的修改
        # 注意：可能有短语模式触发的修改，但"道岔"应保留
        collocation_changes = [c for c in result.changes if c.get("type") == "collocation_fix"]
        assert collocation_changes == []

    def test_collocation_signal_valid(self, corrector):
        """测试信号机在合法共现语境下不被修改."""
        text = "出站信号机点灯"
        result = corrector.process(text)
        assert "信号机" in result.text
        assert result.changes == []

    def test_no_over_correct(self, corrector):
        """测试不应被错误修改的文本."""
        text = "3道空闲，无调车作业"
        result = corrector.process(text)
        assert "3道" in result.text
        assert "空闲" in result.text

    def test_phonetic_candidate_recall(self, corrector):
        """测试拼音混淆候选召回（假设 Layer 2 漏掉的词）."""
        # 构造一个包含未收录别名变体的文本
        # 注意：这个测试依赖于实际别名数据，如果数据未覆盖可能会失败
        text = "下行咽喉区无异常"
        result = corrector.process(text)
        # 至少不应破坏原文
        assert "下行" in result.text
        assert "无异常" in result.text

    def test_ngram_model_basic(self):
        """测试 N-gram 模型基本功能."""
        model = NgramModel(n=2)
        texts = [
            "18号道岔开通反位",
            "出站信号机点灯",
            "3道空闲",
            "准备接车进路",
        ]
        model.train(texts)

        # 合法搭配得分应更高（绝对值更小）
        score_valid = model.score_text("18号道岔开通反位")
        # 单独测试 unigram 概率都低，但序列内相对关系更重要
        assert score_valid < 0  # log-prob 应为负

    def test_ngram_save_load(self, tmp_path):
        """测试 N-gram 模型保存和加载."""
        model = NgramModel(n=2)
        model.train(["18号道岔开通反位", "出站信号机点灯"])

        path = tmp_path / "ngram.json"
        model.save(path)

        loaded = NgramModel.load(path)
        assert loaded.n == 2
        assert len(loaded.vocab) > 0

    def test_collocation_rule_score(self):
        """测试共现规则得分计算."""
        from src.collocation_rules import CollocationRule

        rule = CollocationRule(
            term="道岔",
            before=["号", "故障"],
            after=["定位", "反位"],
        )

        # 合法搭配
        score = rule.check(["号"], ["定位"])
        assert score == 1.0

        # 部分合法
        score = rule.check(["号"], ["发车"])
        assert score == 0.5

        # 不合法
        score = rule.check(["列车"], ["发车"])
        assert score == 0.0

    def test_phonetic_candidate_generator(self):
        """测试拼音混淆候选生成器."""
        gen = PhoneticCandidateGenerator()
        # 对已知别名测试
        cands = gen.generate_candidates("道差", top_k=3)
        # "道差" 本身是别名，候选中不应包含自身
        # 但可能返回其他同音/近音候选（如"道岔"）
        for c in cands:
            assert c["candidate"] != "道差"

    def test_context_pipeline_integration(self):
        """测试 Layer 3 在流水线中的集成."""
        from src.pipeline import PostCorrectionPipeline

        pipeline = PostCorrectionPipeline()
        text = "G一千零二十三次列车，十八号道差开通反位"
        result = pipeline.run(text, layers=[1, 2, 3])

        assert "G1023次" in result.corrected
        assert "18号道岔" in result.corrected
        assert "preprocessor" in result.layers_applied
        assert "dictionary" in result.layers_applied
        assert "layer1" in result.layer_outputs
        assert "layer2" in result.layer_outputs

    def test_layer_outputs_tracking(self):
        """测试 layer_outputs 正确记录每层输出."""
        from src.pipeline import PostCorrectionPipeline

        pipeline = PostCorrectionPipeline()
        result = pipeline.run("十八号道差开通反位", layers=[1, 2, 3])

        assert "layer1" in result.layer_outputs
        assert "layer2" in result.layer_outputs
        # layer1 应为 ITN 后结果
        assert "18号" in result.layer_outputs["layer1"]
        # layer2 应为词典纠错后结果
        assert "道岔" in result.layer_outputs["layer2"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
