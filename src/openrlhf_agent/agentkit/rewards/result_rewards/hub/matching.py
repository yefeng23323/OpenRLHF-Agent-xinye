"""Reward strategy that matches predictions directly against labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from openrlhf_agent.utils.types import Action, RewardSample
from openrlhf_agent.agentkit.rewards.result_rewards.base import ResultRewardStrategy


@dataclass
class MatchingReward(ResultRewardStrategy):
    """Default reward strategy that compares predictions and labels."""

    correct_score: float = 1.0
    miss_score: float = 0.0

    def score_response(self, response: str, label: Optional[Any]) -> float:
        """Score a plain-text response against the target label."""

        if label is None:
            return self.miss_score

        target = str(label).strip()
        prediction = response.strip()
        return self.correct_score if prediction == target else self.miss_score

    async def score(
        self,
        *,
        action: Action,
        label: Optional[Any],
        sample: Optional[RewardSample] = None,
    ) -> float:
        """Derive a reward from the parsed assistant action."""

        if label is None:
            return self.miss_score

        response = self.extract_final_response(action)
        if not response:
            return self.miss_score

        return self.score_response(response, label)
