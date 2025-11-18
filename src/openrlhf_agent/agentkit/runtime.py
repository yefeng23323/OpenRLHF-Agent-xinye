"""Streaming runtime loop for the tool-using agent."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from openrlhf_agent.utils.types import Message
from openrlhf_agent.backends import LLMEngine
from openrlhf_agent.agentkit.environments import Environment
from openrlhf_agent.agentkit.protocols import ChatProtocol
from openrlhf_agent.agentkit.session import AgentSession


class AgentRuntime:
    """Coordinates the language model with the environment at inference time."""

    def __init__(
        self,
        engine: LLMEngine,
        environment: Environment,
        protocol: ChatProtocol,
        *,
        max_new_tokens_per_step: int = 10240,
    ) -> None:
        self.engine = engine
        self.session = AgentSession(environment=environment, protocol=protocol)
        self.max_new_tokens_per_step = max_new_tokens_per_step

    def _bootstrap_prompt_ids(self, messages: Sequence[Dict[str, Any]]) -> List[int]:
        prompt = self.session.initialize(messages)
        return self.engine.tokenize(prompt)

    def _append_feedback_tokens(self, prompt_ids: List[int], feedback_text: str) -> None:
        if not feedback_text:
            return
        prompt_ids.extend(self.engine.tokenize(feedback_text))

    def run_steps(self, messages: Sequence[Dict[str, Any]]):
        """Yield chat messages emitted during the interaction loop."""

        prompt_ids = self._bootstrap_prompt_ids(messages)

        for _ in range(self.session.environment.max_steps):
            # Ask the model for the next action and track the emitted tokens.
            action_ids, action_text = self.engine.generate(
                prompt_ids,
                max_tokens=self.max_new_tokens_per_step,
            )
            prompt_ids.extend(action_ids)

            observation, _ = self.session.step_from_text(action_text)
            # Emit the assistant reply and any tool observations.
            for message in observation.feedback_messages:
                yield message.model_dump(exclude_none=True)

            if observation.done:
                return

            # Append tool outputs (plus the next assistant prefix) before continuing.
            self._append_feedback_tokens(prompt_ids, observation.feedback_text)

        yield Message(
            role="assistant",
            content="Max steps reached without final response.",
        ).model_dump(exclude_none=True)

    def run_final(self, messages: Sequence[Dict[str, Any]]) -> Optional[str]:
        """Convenience wrapper that returns the last assistant content."""

        final_text: Optional[str] = None
        for message in self.run_steps(messages):
            role = message.get("role") if isinstance(message, dict) else None
            if role != "assistant":
                continue
            tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
            if tool_calls:
                continue
            content = message.get("content") if isinstance(message, dict) else None
            final_text = content or final_text
        return final_text
