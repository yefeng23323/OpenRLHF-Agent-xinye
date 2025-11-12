"""Session management for the tool-using agent."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

from openrlhf_agent.chat_protocol import ChatProtocol
from openrlhf_agent.core import StepOutcome, Message, Action
from openrlhf_agent.environment import Environment
from openrlhf_agent.orchestrator.conversation import Conversation


class AgentSession:
    """Maintains chat history and bridges the protocol with the environment."""

    def __init__(self, environment: Environment, protocol: ChatProtocol) -> None:
        self.environment = environment
        self.protocol = protocol
        self.history = Conversation()

    @staticmethod
    def _has_parse_error(action: Action) -> bool:
        if action.refusal:
            return True
        return action.tool_calls and any(call.refusal for call in action.tool_calls)

    def _prepare_history(self, payload: Optional[Union[Sequence[Dict[str, Any]], str]]) -> None:
        """Reset the chat history and optionally seed prior turns."""

        self.history.reset(system_prompt=self.environment.system_prompt)
        if payload is None:
            return

        if isinstance(payload, str):
            text = payload.strip()
            if not text:
                return
            parsed_messages = self.protocol.parse_messages_from_completion_text(text)
            # It will have 2 system
            # if parsed_messages and parsed_messages[0].role == "system":
            #     parsed_messages = parsed_messages[1:]
            if parsed_messages:
                self.history.extend(parsed_messages)
            return

        self.history.extend(payload)

    def initialize(
        self, payload: Optional[Union[Sequence[Dict[str, Any]], str]] = None
    ) -> str:
        """Reset environment state and return the first prompt."""

        self.environment.reset_step()
        self._prepare_history(payload)
        return self.protocol.render_messages(
            messages=self.history.messages,
            tools_manifest=self.environment.tools_manifest(),
            add_generation_prompt=True,
        )

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
            tool_calls=action.tool_calls or None,
            reasoning_content=action.reasoning_content,
        )
        parse_error = self._has_parse_error(action)
        if parse_error and not action.tool_calls and raw_text is not None:
            # Preserve the unparsed text so the user can see what went wrong.
            assistant_message.content = raw_text
            assistant_message.reasoning_content = None
        self.history.append(assistant_message)

        # Observation messages
        observations, reward, terminated = self.environment.step(
            action, label=label, runtime=runtime
        )

        tool_messages = [Message(role="tool", content=observation) for observation in observations]
        if tool_messages:
            tool_payload = [message.model_dump(exclude_none=True) for message in tool_messages]
            feedback_text = self.protocol.render_messages(
                messages=tool_payload,
                add_generation_prompt=True,
            )
        else:
            feedback_text = ""

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
