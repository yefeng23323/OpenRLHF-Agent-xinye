"""Reward strategy helpers."""

from openrlhf_agent.environment.reward.base import RewardStrategy
from openrlhf_agent.environment.reward.matching import MatchingReward
from openrlhf_agent.environment.reward.factory import make_reward, register_reward

_REWARD_ = [
    "MatchingReward",
]

__all__ = [
    "RewardStrategy",
    "make_reward",
    "register_reward",
    *_REWARD_,
]
