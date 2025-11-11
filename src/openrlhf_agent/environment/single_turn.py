"""Environment that ends after one plain-text assistant reply."""

from __future__ import annotations

from typing import List, Optional, Tuple

from openrlhf_agent.core import Action
from openrlhf_agent.environment.base import Environment
from openrlhf_agent.environment.reward import RewardStrategy, make_reward
from openrlhf_agent.environment.tools import ToolRegistry


DEFAULT_SINGLE_TURN_PROMPT = (
    "You are a helpful assistant.\n"
)


class SingleTurnEnvironment(Environment):
    """Minimal environment that accepts only one assistant reply."""

    def __init__(
        self,
        *,
        system_prompt: str = DEFAULT_SINGLE_TURN_PROMPT,
        result_reward: Optional[RewardStrategy] = None,
    ) -> None:
        super().__init__(
            max_steps=1,
            registry=ToolRegistry([]),
            result_reward=result_reward,
        )
        self._system_prompt = system_prompt

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def step(
        self,
        action: Action,
        label: Optional[str] = None,
        runtime: bool = False,
    ) -> Tuple[List[str], float, bool]:
        self._step_index += 1

        reply = (action.content or "").strip()
        valid_plain_text = bool(reply) and not action.refusal and not action.tool_calls  # plain text only

        reward = 0.0
        if not runtime and valid_plain_text:
            reward = self.result_reward(action, label)

        return [], reward, True


__all__ = ["SingleTurnEnvironment"]
