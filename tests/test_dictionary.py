"""Layer 2 词典纠错单元测试."""

import pytest

from src.dictionary_corrector import DictionaryCorrector


class TestDictionaryCorrector:
    @pytest.fixture
    def corrector(self):
        return DictionaryCorrector()

    def test_alias_replace_daocha(self, corrector):
        """测试道岔别名替换."""
        text = "18号道差开通反位"
        result = corrector.process(text)
        assert "道岔" in result.text
        assert "道差" not in result.text

    def test_alias_replace_yanhouqu(self, corrector):
        """测试咽喉区别名替换."""
        text = "下行烟后区无异常"
        result = corrector.process(text)
        assert "咽喉区" in result.text
        assert "烟后区" not in result.text

    def test_alias_replace_bise(self, corrector):
        """测试闭塞别名替换."""
        text = "必色分区故障"
        result = corrector.process(text)
        assert "闭塞分区" in result.text
        assert "必色分区" not in result.text

    def test_alias_replace_signal(self, corrector):
        """测试信号相关别名替换."""
        text = "出站新号机点灯"
        result = corrector.process(text)
        assert "信号机" in result.text
        assert "新号机" not in result.text

    def test_no_false_positive(self, corrector):
        """测试不应被错误替换的情况."""
        text = "道岔位置正确"
        result = corrector.process(text)
        # "道岔"是标准词，不应被修改
        assert "道岔" in result.text

    def test_multiple_aliases(self, corrector):
        """测试同一句中多个别名替换."""
        text = "道差开通反位，新号机点灯"
        result = corrector.process(text)
        assert "道岔" in result.text
        assert "信号机" in result.text
        assert "道差" not in result.text
        assert "新号机" not in result.text

    def test_track_occupied(self, corrector):
        """测试占用/空闲等状态词."""
        text = "3道控闲"
        result = corrector.process(text)
        assert "空闲" in result.text

    def test_report_duty_officer(self, corrector):
        """测试值班员别名替换."""
        text = "报告值斑员设备正常"
        result = corrector.process(text)
        assert "值班员" in result.text

    def test_pattern_active_replace_switch(self, corrector):
        """测试正则模式主动替换道岔别名（pattern兜底）."""
        text = "18号道差开通定位"
        result = corrector.process(text)
        assert "18号道岔开通定位" in result.text
        assert "道差" not in result.text

    def test_pattern_active_replace_signal(self, corrector):
        """测试正则模式主动替换信号机别名."""
        text = "出站新号机灯光熄灭"
        result = corrector.process(text)
        assert "信号机" in result.text
        assert "新号机" not in result.text

    def test_short_alias_whole_word_protection(self, corrector):
        """测试短别名全词匹配保护（2字别名不在长词内部误触）."""
        # 假设未来有短别名，测试保护机制生效
        # 当前数据中没有 <=2 字的别名，测试构造一个场景
        text = "保持到发线空闲"
        result = corrector.process(text)
        # "到发线" 中的 "到" 不应被误替换（即使未来有 "到"->"道" 的别名）
        assert "到发线" in result.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
