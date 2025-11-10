"""Session management for the tool-using agent."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from openrlhf_agent.chat_protocol import ChatProtocol
from openrlhf_agent.core import StepOutcome, Message, Action
from openrlhf_agent.environment import Environment
from openrlhf_agent.orchestrator.history import Conversation


class AgentSession:
    """Maintains chat history and bridges the protocol with the environment."""

    def __init__(self, environment: Environment, protocol: ChatProtocol) -> None:
        self.environment = environment
        self.protocol = protocol
        self.history = Conversation()

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _has_parse_error(action: Action) -> bool:
        if action.refusal:
            return True
        return any(call.refusal for call in action.tool_calls)

    def _ingest_payload(self, payload: Optional[Union[Sequence[Dict[str, Any]], str]]) -> None:
        if payload is None:
            return

        if isinstance(payload, str):
            text = payload.strip()
            if not text:
                return
            parsed_messages = self.protocol.parse_messages_from_completion_text(text)
            if parsed_messages and parsed_messages[0].role == "system":
                parsed_messages = parsed_messages[1:]
            if parsed_messages:
                self.history.extend(parsed_messages)
            return

        if payload:
            self.history.extend(payload)

    def _render_prompt(self) -> str:
        return self.history.render_prompt(
            self.protocol,
            tools_manifest=self.environment.tools_manifest(),
        )

    def _render_tool_feedback(self, tool_messages: List[Message]) -> str:
        if not tool_messages:
            return ""
        tool_payload = [message.model_dump(exclude_none=True) for message in tool_messages]
        return self.protocol.render_messages(
            messages=tool_payload,
            add_generation_prompt=True,
        )

    # ---------------------------------------------------------------- lifecycle

    def initialize(
        self, payload: Optional[Union[Sequence[Dict[str, Any]], str]] = None
    ) -> str:
        """Reset environment state and return the first prompt."""

        self.environment.reset_step()
        self.history.reset(system_prompt=self.environment.system_prompt)
        self._ingest_payload(payload)
        return self._render_prompt()

    # ------------------------------------------------------------------- stepping

    def step(
        self,
        action: Action,
        *,
        label: Optional[str] = None,
        runtime: bool = False,
        raw_text: Optional[str] = None,
    ) -> StepOutcome:
        """Apply a parsed assistant action to the environment."""

        # Action message
        assistant_message = Message(
            role="assistant",
            content=action.content,
            tool_calls=action.tool_calls,
        )
        parse_error = self._has_parse_error(action)
        if parse_error and not action.tool_calls and raw_text is not None:
            # Preserve the unparsed text so the user can see what went wrong.
            assistant_message.content = raw_text
        self.history.append(assistant_message)

        # Observation messages
        observations, reward, terminated, _ = self.environment.step(
            action, label=label, runtime=runtime
        )

        tool_messages = [self.history.add_tool(observation) for observation in observations]
        feedback_text = self._render_tool_feedback(tool_messages)

        return StepOutcome(
            step_index=self.environment.step_index,
            feedback_messages=[assistant_message, *tool_messages], # for runtime, with action
            feedback_text=feedback_text,  # for train, without action
            reward=reward,
            terminated=terminated,
        )

    def step_from_text(
        self,
        action_text: str,
        *,
        label: Optional[str] = None,
        runtime: bool = False,
    ) -> StepOutcome:
        """Parse a raw model response and forward to `step`."""

        parsed_action = self.protocol.parse_assistant_text(action_text)
        return self.step(
            parsed_action,
            label=label,
            runtime=runtime,
            raw_text=action_text,
        )
