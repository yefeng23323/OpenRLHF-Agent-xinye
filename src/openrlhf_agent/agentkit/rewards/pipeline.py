"""Reward pipeline orchestrating process/result strategies."""

from __future__ import annotations

from typing import Optional

from openrlhf_agent.utils.types import Action

from .base import ProcessRewardStrategy, ResultRewardStrategy
from .matching import MatchingReward


class RewardPipeline:
    """Combines optional process/result reward components."""

    def __init__(
        self,
        *,
        result_reward: Optional[ResultRewardStrategy] = None,
        process_reward: Optional[ProcessRewardStrategy] = None,
    ) -> None:
        self._result_reward = result_reward if result_reward is not None else MatchingReward()
        self._process_reward = process_reward

    def score(
        self,
        *,
        action: Action,
        label: Optional[str],
    ) -> float:
        """Compute a scalar reward for the latest action."""

        reward = 0.0
        used_tools = bool(action.tool_calls)
        final_plain_text = bool((action.content or "").strip()) and not used_tools and not action.refusal

        if self._process_reward and used_tools:
            reward += self._process_reward.score_process(action=action, label=label)

        if self._result_reward and final_plain_text:
            reward += self._result_reward.score_result(action=action, label=label)

        return reward


__all__ = ["RewardPipeline"]
