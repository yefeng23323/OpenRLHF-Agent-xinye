"""Base abstraction for result reward strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from openrlhf_agent.utils.types import Action


class ResultRewardStrategy(ABC):
    """Scores the final user-visible reply."""

    @abstractmethod
    def score(
        self,
        *,
        action: Action,
        label: Optional[str],
    ) -> float:
        """Return the reward for the assistant's final answer."""
