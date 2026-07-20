"""Layer 1 预处理单元测试."""

import pytest

from src.preprocessor import Preprocessor


class TestPreprocessor:
    @pytest.fixture
    def preprocessor(self):
        return Preprocessor()

    def test_train_number_normalization(self, preprocessor):
        """测试车次号规范化."""
        text = "G一千零二十三次列车预告"
        result = preprocessor.process(text)
        assert "G1023次" in result.text

    def test_train_prefix_uppercase(self, preprocessor):
        """测试车次前缀大写."""
        text = "g534次列车接近"
        result = preprocessor.process(text)
        assert "G534次" in result.text

    def test_switch_number_normalization(self, preprocessor):
        """测试道岔编号规范化."""
        text = "十八号道岔开通反位"
        result = preprocessor.process(text)
        assert "18号道岔" in result.text

    def test_signal_number_normalization(self, preprocessor):
        """测试信号机编号规范化."""
        text = "下行进站信号机灯光熄灭"
        result = preprocessor.process(text)
        # 信号机不需要数字替换，保持不变
        assert "下行进站信号机" in result.text

    def test_percent_normalization(self, preprocessor):
        """测试百分比规范化."""
        text = "限速百分之四十运行"
        result = preprocessor.process(text)
        assert "40%" in result.text

    def test_year_normalization(self, preprocessor):
        """测试年份规范化."""
        text = "二零二五年六月十七号"
        result = preprocessor.process(text)
        assert "2025年" in result.text
        assert "17日" in result.text or "17号" in result.text

    def test_kilometer_normalization(self, preprocessor):
        """测试公里标规范化."""
        text = "停于内江北至资中北站上行区间一百三十四公里八百米处"
        result = preprocessor.process(text)
        assert "K134+800" in result.text

    def test_punctuation_addition(self, preprocessor):
        """测试标点补全."""
        text = "2号扳道员解锁14号道岔"
        result = preprocessor.process(text)
        # 无标点文本应补全标点
        assert result.text.endswith("。")

    def test_no_over_punctuation(self, preprocessor):
        """测试已有标点的文本不被过度修改."""
        text = "2号扳道员，解锁14号道岔。"
        result = preprocessor.process(text)
        assert "，" in result.text
        assert "。" in result.text

    def test_track_number_normalization(self, preprocessor):
        """测试股道编号规范化."""
        text = "3道空闲本站无调车作业"
        result = preprocessor.process(text)
        assert "3道" in result.text

    def test_switch_alias_with_number(self, preprocessor):
        """测试道岔别名（道差）前的编号保留号字."""
        text = "十八号道差开通反位"
        result = preprocessor.process(text)
        # 关键："十八号道差" 不应被错误转为 "18道差"，而应保留 "号"
        assert "18号道差" in result.text
        assert "18道差" not in result.text

    def test_button_number_normalization(self, preprocessor):
        """测试按钮编号规范化."""
        text = "按下一号按钮"
        result = preprocessor.process(text)
        assert "1号按钮" in result.text

    def test_track_with_chinese_number(self, preprocessor):
        """测试中文数字股道（不含号）仍正确转换."""
        text = "三道发车"
        result = preprocessor.process(text)
        assert "3道" in result.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
