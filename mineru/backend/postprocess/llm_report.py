"""仅记录本次增强执行情况，不扩展文档协议。"""

from dataclasses import dataclass, field


@dataclass
class LLMFeatureReport:
    """记录一个增强功能的候选数及已完成结果，未完成项保留为 pending。"""

    total: int = 0
    succeeded: int = 0
    failed: int = 0

    @property
    def pending(self) -> int:
        """计算仍未完成的候选数，供超时回退报告使用。"""
        return self.total - self.succeeded - self.failed


@dataclass
class LLMAidedReport:
    """分别保存标题和单元格增强结果，允许超时后读取已完成进度。"""

    title_leveling: LLMFeatureReport = field(default_factory=LLMFeatureReport)
    cross_page_table_cell_merge: LLMFeatureReport = field(default_factory=LLMFeatureReport)


__all__ = ["LLMAidedReport", "LLMFeatureReport"]
