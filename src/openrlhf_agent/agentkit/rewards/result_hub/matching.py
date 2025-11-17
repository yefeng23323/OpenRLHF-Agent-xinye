"""Reward strategy that matches predictions directly against labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from openrlhf_agent.utils.types import Action
from openrlhf_agent.agentkit.rewards.base import ResultRewardStrategy


@dataclass
class MatchingReward(ResultRewardStrategy):
    """Default reward strategy that compares predictions and labels."""

    correct_score: float = 1.0
    miss_score: float = 0.0

    def score_response(self, response: str, label: Optional[str]) -> float:
        """Score a plain-text response against the target label."""

        if label is None:
            return self.miss_score

        target = label.strip()
        prediction = response.strip()
        return self.correct_score if prediction == target else self.miss_score

    def score(
        self,
        *,
        action: Action,
        label: Optional[str],
    ) -> float:
        """Derive a reward from the parsed assistant action."""

        if label is None:
            return 0.0

        # Reward response
        final_text = (action.content or "").strip()
        if final_text and not action.tool_calls:
            return self.score_response(final_text, label)

        # Reward final tool
        for tool_call in action.tool_calls or []:
            if tool_call is None or (tool_call.name or "").strip() != "final":
                continue

            arguments = tool_call.arguments or {}
            if not isinstance(arguments, dict):
                continue

            answer = str(arguments.get("answer", "")).strip()
            if answer:
                return self.score_response(answer, label)

        return 0.0


__all__ = ["MatchingReward"]
