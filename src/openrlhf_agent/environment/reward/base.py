"""Reward strategy abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from openrlhf_agent.core import ParsedAssistantAction


class RewardStrategy(ABC):
    """Interface all reward strategies must implement."""

    @abstractmethod
    def reward_from_action(
        self,
        action: ParsedAssistantAction,
        label: Optional[str],
    ) -> float:
        """Compute reward from a parsed action and optional label."""


__all__ = ["RewardStrategy"]
