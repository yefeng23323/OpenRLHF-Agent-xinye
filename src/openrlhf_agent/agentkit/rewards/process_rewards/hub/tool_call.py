"""Lightweight tool process reward: per-tool scoring rules."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from openrlhf_agent.agentkit.rewards.process_rewards.base import ProcessRewardStrategy
from openrlhf_agent.utils.types import Action


@dataclass
class ToolPolicy:
    """Per-tool scoring config."""

    reward_per_call: float
    max_calls: Optional[int]
    overuse_penalty: float

    @staticmethod
    def new(policy):
        if isinstance(policy, ToolPolicy):
            return policy

        if isinstance(policy, Mapping):
            return ToolPolicy(
                reward_per_call=policy.get("reward_per_call", 0.1),
                max_calls=policy.get("max_calls", None),
                overuse_penalty=policy.get("overuse_penalty", -0.05),
            )

        raise TypeError("tool_policies values must be ToolPolicy or mapping")


@dataclass
class ToolCallReward(ProcessRewardStrategy):
    """Per-tool scoring with simple caps and penalties."""

    min_reward: Optional[float] = None
    max_reward: Optional[float] = None
    parse_error_penalty: float = -0.2
    penalty_for_refused: float = -0.1
    tool_policies: Mapping[str, ToolPolicy] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalize user-supplied policies; allow dicts or ToolPolicy.
        self.tool_policies = {
            key.lower(): ToolPolicy.new(policy)
            for key, policy in (self.tool_policies or {}).items()
        }

    def _clamp(self, reward: float) -> float:
        if self.max_reward is not None:
            reward = min(reward, self.max_reward)
        if self.min_reward is not None:
            reward = max(reward, self.min_reward)
        return reward

    def _collect_call_stats(self, action: Action) -> tuple[Counter[str], int]:
        counts: Counter[str] = Counter()
        refused = 0

        for call in action.tool_calls or []:
            if call is None or call.refusal:
                refused += 1
                continue

            name = (call.name or "").strip()
            if not name:
                refused += 1
                continue

            counts[name.lower()] += 1

        return counts, refused

    def _score_counts(self, counts: Counter[str], refused: int) -> float:
        reward = refused * self.penalty_for_refused

        for name, count in counts.items():
            policy = self.tool_policies.get(name, None)
            if not policy:  # ignore calls to tools without a policy
                continue

            allowed = max(0, policy.max_calls) if policy.max_calls is not None else count
            if count <= allowed:
                reward += min(count, allowed) * policy.reward_per_call
            else:
                reward += (count - allowed) * policy.overuse_penalty

        return reward

    async def score(
        self,
        *,
        action: Action,
        label: Optional[Any],
    ) -> float:
        """Score tool usage with per-tool rules; fast-path, no extra objects."""

        if action.refusal:
            return self._clamp(self.parse_error_penalty)

        counts, refused = self._collect_call_stats(action)
        reward = self._score_counts(counts, refused)
        return self._clamp(reward)
