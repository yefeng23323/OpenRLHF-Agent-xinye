"""Reward strategy factory helpers."""

from __future__ import annotations

from typing import Dict, Optional, Type

from .base import ResultRewardStrategy
from .matching import MatchingReward


_DEFAULT_STRATEGY = "matching"
_RESULT_REGISTRY: Dict[str, Type[ResultRewardStrategy]] = {
    "matching": MatchingReward,
}


def register_result_reward(name: str, strategy_cls: Type[ResultRewardStrategy]) -> None:
    """Register a result reward strategy class under a readable name."""

    normalized = name.lower()
    _RESULT_REGISTRY[normalized] = strategy_cls


def make_result_reward(
    name: Optional[str] = None,
    *,
    config: Optional[dict] = None,
) -> ResultRewardStrategy:
    """Instantiate a result reward strategy via the registry."""

    resolved = (name or _DEFAULT_STRATEGY).lower()
    try:
        reward_cls = _RESULT_REGISTRY[resolved]
    except KeyError as exc:
        raise ValueError(f"Unknown reward strategy '{name}'.") from exc

    payload = dict(config or {})
    return reward_cls(**payload)


__all__ = ["register_result_reward", "make_result_reward"]
