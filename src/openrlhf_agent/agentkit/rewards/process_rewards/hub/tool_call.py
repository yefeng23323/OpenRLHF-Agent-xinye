"""Reward strategy that encourages valid tool usage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from openrlhf_agent.agentkit.rewards.process_rewards.base import ProcessRewardStrategy
from openrlhf_agent.utils.types import Action, ToolCall


@dataclass
class ToolCallReward(ProcessRewardStrategy):
    """Assigns reward when the model issues tool calls."""

    reward_per_call: float = 0.1
    no_tool_score: float = 0.0
    parse_error_score: float = -0.1
    max_reward: Optional[float] = None

    def _has_parse_error(self, action: Action) -> bool:
        if action.refusal:
            return True
        return any(call and call.refusal for call in action.tool_calls or [])

    def _filtered_calls(self, action: Action) -> List[ToolCall]:
        calls: List[ToolCall] = []
        for tool_call in action.tool_calls or []:
            if tool_call is None or tool_call.refusal:
                continue

            name = (tool_call.name or "").strip()
            if not name:
                continue

            # if not self.include_final_tool and name == self.final_tool_name:
            #     continue

            calls.append(tool_call)

        return calls

    async def score(
        self,
        *,
        action: Action,
        label: Optional[Any],
    ) -> float:
        """Reward tool calls while penalizing parse errors."""

        if self._has_parse_error(action):
            return self.parse_error_score

        valid_calls = self._filtered_calls(action)
        if not valid_calls:
            return self.no_tool_score

        reward = len(valid_calls) * self.reward_per_call
        if self.max_reward is not None:
            reward = min(reward, self.max_reward)

        return reward
