"""ITN规则独立单元测试."""

import pytest

from src.itn_rules import (
    _replace_year,
    _replace_month_day,
    _replace_time,
    _replace_percent,
    _replace_speed,
    _replace_kilometer,
    _replace_train_number,
    _replace_numbered_term,
    _normalize_train_prefix,
    _normalize_punctuation,
)


class TestITNRules:
    def test_replace_year(self):
        """测试中文年份替换."""
        assert "2025年" in _replace_year("二零二五年")
        assert "2024年" in _replace_year("二零二四年")

    def test_replace_year_fallback(self):
        """测试年份逐字读法回退."""
        assert "2025年" in _replace_year("二零二五年")

    def test_replace_month_day(self):
        """测试月份/日期替换."""
        result = _replace_month_day("六月十七号")
        assert "6月" in result
        assert "17日" in result or "17号" in result

    def test_replace_month_day_protect_railway(self):
        """测试铁路术语保护（号->日不应替换）."""
        result = _replace_month_day("十八号道岔")
        assert "十八号道岔" == result

    def test_replace_time(self):
        """测试中文时间替换."""
        assert "10:30" in _replace_time("十点三十分")
        assert "9:05" in _replace_time("九点五分")

    def test_replace_percent(self):
        """测试百分比替换."""
        assert "40%" in _replace_percent("限速百分之四十")
        assert "25%" in _replace_percent("百分之二十五")

    def test_replace_speed(self):
        """测试速度表达替换."""
        assert "限速80km/h" in _replace_speed("限速八十公里每小时")
        assert "限速120km/h" in _replace_speed("限速一百二十公里每小时")

    def test_replace_kilometer(self):
        """测试公里标替换."""
        assert "K134+800" in _replace_kilometer("一百三十四公里八百米")
        assert "K134+800" in _replace_kilometer("k一百三十四公里八百米")

    def test_replace_train_number(self):
        """测试车次号替换."""
        assert "K1023次" in _replace_train_number("K一千零二十三次")
        assert "G534次" in _replace_train_number("G五百三十四次")

    def test_replace_numbered_term_switch(self):
        """测试道岔编号替换."""
        assert "18号道岔" in _replace_numbered_term("十八号道岔")

    def test_replace_numbered_term_signal(self):
        """测试信号机编号替换."""
        assert "1号信号机" in _replace_numbered_term("一号信号机")

    def test_replace_numbered_term_track(self):
        """测试股道编号替换（不含号）."""
        assert "3道" in _replace_numbered_term("三道")

    def test_replace_numbered_term_track_with_hao(self):
        """测试股道含号时保留号（避免误伤道岔）."""
        assert "18号道" in _replace_numbered_term("十八号道")

    def test_replace_numbered_term_button(self):
        """测试按钮编号替换."""
        assert "2号按钮" in _replace_numbered_term("二号按钮")

    def test_normalize_train_prefix(self):
        """测试车次前缀大写."""
        assert "G534次" in _normalize_train_prefix("g534次")
        assert "K1023次" in _normalize_train_prefix("k1023次")

    def test_normalize_punctuation(self):
        """测试标点半规范化."""
        assert " " not in _normalize_punctuation("G 1023 次")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
