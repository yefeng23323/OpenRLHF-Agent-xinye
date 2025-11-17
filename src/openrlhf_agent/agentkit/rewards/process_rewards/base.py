"""Base abstraction for process reward strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from openrlhf_agent.utils.types import Action


class ProcessRewardStrategy(ABC):
    """Scores intermediate planning/tool steps."""

    @abstractmethod
    def score(
        self,
        *,
        action: Action,
        label: Optional[str],
    ) -> float:
        """Return the reward associated with the latest tool usage."""
