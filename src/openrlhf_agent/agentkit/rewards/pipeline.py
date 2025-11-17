"""Reward pipeline orchestrating process/result strategies."""

from __future__ import annotations

from typing import Optional

from openrlhf_agent.utils.types import Action

from .base import ProcessRewardStrategy, ResultRewardStrategy


class RewardPipeline:
    """Combines optional process/result reward components."""

    def __init__(
        self,
        *,
        result_reward: Optional[ResultRewardStrategy] = None,
        process_reward: Optional[ProcessRewardStrategy] = None,
    ) -> None:
        self._result_reward = result_reward
        self._process_reward = process_reward

        assert (
            self._result_reward is not None or self._process_reward is not None
        ), "RewardPipeline requires at least one reward strategy"

    def score(
        self,
        *,
        action: Action,
        label: Optional[str],
        done: bool,
    ) -> float:
        """Compute a scalar reward for the latest action."""

        reward = 0.0

        if self._process_reward:
            reward += self._process_reward.score(action=action, label=label)

        if self._result_reward and done:
            reward += self._result_reward.score(action=action, label=label)

        return reward
