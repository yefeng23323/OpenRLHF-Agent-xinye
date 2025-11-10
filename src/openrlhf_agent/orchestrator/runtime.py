"""Streaming runtime loop for the tool-using agent."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from openrlhf_agent.chat_protocol import ChatProtocol
from openrlhf_agent.core import ChatMessage
from openrlhf_agent.environment import Environment
from openrlhf_agent.engine import LLMEngine

from openrlhf_agent.orchestrator.session import AgentSession


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
        self.session = AgentSession(environment, protocol)
        self.max_new_tokens_per_step = max_new_tokens_per_step

    def run_steps(self, messages: Sequence[Dict[str, Any]]):
        """Yield chat messages emitted during the interaction loop."""

        prompt_ids = self.engine.tokenize(self.session.initialize(messages))

        for _ in range(self.session.environment.max_steps):
            # Ask the model for the next action and track the emitted tokens.
            action_ids, action_text = self.engine.generate(
                prompt_ids,
                max_tokens=self.max_new_tokens_per_step,
            )
            prompt_ids.extend(action_ids)

            action_result = self.session.step_from_text(action_text, runtime=True)
            # Emit the assistant reply and any tool observations.
            for message in action_result.feedback_messages:
                yield message.model_dump(exclude_none=True)

            if action_result.terminated:
                return

            # Append tool outputs (plus the next assistant prefix) before continuing.
            feedback_ids = self.engine.tokenize(action_result.feedback_text)
            prompt_ids.extend(feedback_ids)

        yield ChatMessage(
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
