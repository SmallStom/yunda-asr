"""术语共现规则库.

定义铁路术语的合法前后搭配词，用于上下文校验。
"""

from typing import Dict, List, Set


class CollocationRule:
    """单条共现规则."""

    def __init__(
        self,
        term: str,
        before: List[str],
        after: List[str],
        score_threshold: float = 0.3,
    ):
        self.term = term
        self.before = set(before)
        self.after = set(after)
        self.score_threshold = score_threshold

    def check(self, prev_words: List[str], next_words: List[str]) -> float:
        """检查术语在上下文中的共现得分.

        返回 0-1 之间的得分，1 表示完全合法，0 表示完全不合法。
        """
        score = 0.0
        total = 0

        # 检查前词
        for w in prev_words:
            total += 1
            if w in self.before:
                score += 1.0

        # 检查后词
        for w in next_words:
            total += 1
            if w in self.after:
                score += 1.0

        if total == 0:
            return 0.5  # 无上下文时，中性得分

        return score / total


# 预定义的核心术语共现规则
COLLOCATION_RULES: Dict[str, CollocationRule] = {
    "道岔": CollocationRule(
        term="道岔",
        before=["号", "故障", "开通", "解锁", "加锁", "单锁", "扳动", "操纵", "排列"],
        after=["定位", "反位", "开通", "加锁", "解锁", "单锁", "故障", "编号", "无异常", "好了"],
        score_threshold=0.3,
    ),
    "信号机": CollocationRule(
        term="信号机",
        before=["号", "进站", "出站", "调车", "通过", "引导", "开放", "关闭", "点灯", "灭灯"],
        after=["点灯", "灭灯", "灯光熄灭", "开放", "关闭", "好了", "显示", "故障"],
        score_threshold=0.3,
    ),
    "进路": CollocationRule(
        term="进路",
        before=["准备", "排列", "取消", "接车", "发车", "调车", "影响", "解锁"],
        after=["准备好了", "解锁", "取消", "排列", "准备好", "好了", "办理"],
        score_threshold=0.3,
    ),
    "闭塞": CollocationRule(
        term="闭塞",
        before=["办理", "区间", "分区", "自动", "半自动"],
        after=["分区", "区间", "办理", "好了", "故障"],
        score_threshold=0.3,
    ),
    "联锁": CollocationRule(
        term="联锁",
        before=["", "故障", "试验", "检查", "恢复"],
        after=["试验", "故障", "检查", "恢复", "良好"],
        score_threshold=0.3,
    ),
    "轨道电路": CollocationRule(
        term="轨道电路",
        before=["号", "故障", "检查", "红光带"],
        after=["故障", "红光带", "占用", "空闲", "检查"],
        score_threshold=0.3,
    ),
    "发车": CollocationRule(
        term="发车",
        before=["准备", "同意", "准许", "命令", "办理"],
        after=["好了", "准备", "进路", "信号", "命令"],
        score_threshold=0.3,
    ),
    "接车": CollocationRule(
        term="接车",
        before=["准备", "同意", "准许", "引导", "办理"],
        after=["好了", "准备", "进路", "信号", "命令"],
        score_threshold=0.3,
    ),
    "限速": CollocationRule(
        term="限速",
        before=["命令", "区间", "地段", "注意"],
        after=["运行", "通过", "注意", "km/h"],
        score_threshold=0.3,
    ),
    "加锁": CollocationRule(
        term="加锁",
        before=["道岔", "定位", "反位", "单锁", "破封"],
        after=["好了", "解锁", "单锁", "完毕"],
        score_threshold=0.3,
    ),
    "解锁": CollocationRule(
        term="解锁",
        before=["道岔", "进路", "加锁", "单锁", "总人解"],
        after=["好了", "完毕", "进路", "道岔"],
        score_threshold=0.3,
    ),
    "按钮": CollocationRule(
        term="按钮",
        before=["号", "故障通知", "总人解", "按压", "点击"],
        after=["好了", "按下", "故障", "通知"],
        score_threshold=0.3,
    ),
    "咽喉区": CollocationRule(
        term="咽喉区",
        before=["上行", "下行", "站内", "区间"],
        after=["无异常", "空闲", "占用", "检查"],
        score_threshold=0.3,
    ),
    "占线簿": CollocationRule(
        term="占线簿",
        before=["填写", "抹消", "登记"],
        after=["", "登记", "填写"],
        score_threshold=0.3,
    ),
    "仍无表示": CollocationRule(
        term="仍无表示",
        before=["控制台", "道岔", "信号机", "轨道电路"],
        after=["", "检查", "抢修", "故障"],
        score_threshold=0.3,
    ),
    "密贴": CollocationRule(
        term="密贴",
        before=["尖轨", "基本轨", "轨"],
        after=["", "良好", "无异状", "检查"],
        score_threshold=0.3,
    ),
    "尖轨": CollocationRule(
        term="尖轨",
        before=["", "道岔", "故障"],
        after=["与基本轨密贴", "密贴", "基本轨"],
        score_threshold=0.3,
    ),
    "基本轨": CollocationRule(
        term="基本轨",
        before=["尖轨与", "尖轨", "轨"],
        after=["密贴", "良好", "检查"],
        score_threshold=0.3,
    ),
    "扳动": CollocationRule(
        term="扳动",
        before=["将", "道岔", "来回"],
        after=["三次", "道岔", "完毕"],
        score_threshold=0.3,
    ),
}


def get_collocation_rule(term: str) -> CollocationRule | None:
    """获取术语的共现规则."""
    return COLLOCATION_RULES.get(term)


def get_all_monitored_terms() -> Set[str]:
    """获取所有受监控的术语集合."""
    return set(COLLOCATION_RULES.keys())
