"""Reward strategy factory helpers."""

from __future__ import annotations

from typing import Dict, Optional, Type

from openrlhf_agent.environment.reward.base import RewardStrategy
from openrlhf_agent.environment.reward.matching import MatchingReward


_DEFAULT_STRATEGY = "matching"
_REWARD_REGISTRY: Dict[str, Type[RewardStrategy]] = {
    "matching": MatchingReward,
}


def register_reward(name: str, strategy_cls: Type[RewardStrategy]) -> None:
    """Register a reward strategy class under a name."""

    normalized = name.lower()
    _REWARD_REGISTRY[normalized] = strategy_cls


def make_reward(
    name: Optional[str] = None,
    *,
    config: Optional[dict] = None,
    strategy: Optional[RewardStrategy] = None,
) -> RewardStrategy:
    """Instantiate a reward strategy by name or return the provided strategy."""

    if strategy is not None:
        return strategy

    resolved_name = (name or _DEFAULT_STRATEGY).lower()
    try:
        reward_cls = _REWARD_REGISTRY[resolved_name]
    except KeyError as exc:
        raise ValueError(f"Unknown reward strategy '{name}'.") from exc

    config = dict(config or {})
    return reward_cls(**config)


__all__ = ["register_reward", "make_reward"]
