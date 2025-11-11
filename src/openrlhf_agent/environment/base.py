"""Shared interfaces for agent environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from openrlhf_agent.core import Action, ToolCall
from openrlhf_agent.environment.reward import RewardStrategy
from openrlhf_agent.environment.tools import ToolRegistry


class Environment(ABC):
    """Base interface describing the agent environment contract."""

    def __init__(
        self,
        *,
        max_steps: int = 999,
        registry: Optional[ToolRegistry] = None,
        result_reward: Optional[RewardStrategy] = None,
        process_reward: Optional[RewardStrategy] = None,
    ) -> None:
        self._step_index = 0
        self._max_steps = max_steps
        
        self.registry = registry or ToolRegistry([])

        self._result_reward = result_reward
        self._process_reward = process_reward

    def tools_manifest(self) -> List[Dict[str, Any]]:
        return self.registry.list_openai_tools()

    def execute_tool(self, call: ToolCall, context: Dict[str, Any]) -> str:
        """Execute one tool invocation."""

        tool = self.registry.get(call.name)
        arguments = call.arguments
        if not isinstance(arguments, dict):
            raise TypeError("Tool arguments must be a JSON object.")
        return tool.call(context=context, arguments=arguments)

    def reset_step(self) -> None:
        self._step_index = 0

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def step_index(self) -> int:
        return self._step_index

    def result_reward(self, action: Action, label: Optional[str]) -> float:
        """Reward tied to the final reply."""

        if self._result_reward is None:
            return 0.0
        return self._result_reward.reward_from_action(action, label)

    def process_reward(self, action: Action, label: Optional[str]) -> float:
        """Reward for intermediate tool steps."""

        if self._process_reward is None:
            return 0.0
        return self._process_reward.reward_from_action(action, label)
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the provider/system prompt used for the agent."""

    @abstractmethod
    def step(
        self,
        action: Action,
        label: Optional[str] = None,
        runtime: bool = False,
    ) -> Tuple[List[str], float, bool]:
        """Run one environment transition and return (observations, reward, done)."""
