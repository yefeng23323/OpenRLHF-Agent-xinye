"""Default environment that supports function calls plus a hidden status tool."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from openrlhf_agent.utils.types import Action, ToolCall
from openrlhf_agent.agentkit.environments.base import Environment
from openrlhf_agent.agentkit.tools import CommentaryTool, ToolBase


SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful agent assistant.\n\n"
    "Keep the user informed as you work: give friendly, action-focused updates about what you're doing right now.\n\n"
    "Knowledge cutoff: 2023-06\n"
    "Current date: {date}\n\n"
    # "Rules:\n"
    # "- Use commentary(status=...) to share upbeat progress updates whenever your approach changes.\n"
    # "- To answer the user, provide plain text outside tool calls. That text ends the session.\n"
    # "- Tool calls must be JSON objects within <tool_call></tool_call> tags."
)


class FunctionCallEnvironment(Environment):
    """Function call environment with tool and plain-text finals."""

    def __init__(
        self,
        *,
        tools: Optional[Sequence[ToolBase]] = None,
        system_prompt: Optional[str] = None,
        max_steps: int = 64,
    ) -> None:
        resolved_prompt = system_prompt or SYSTEM_PROMPT_TEMPLATE.format(
            date=datetime.now().strftime("%Y-%m-%d")
        )
        super().__init__(
            tools=list(tools or [CommentaryTool()]),
            system_prompt=resolved_prompt,
            max_steps=max_steps,
        )

    # ----------------------------------------------------------------- helpers

    def _internal_message(
        self,
        *,
        code: str,
        message: str,
        hint: Optional[str] = None,
        tool: Optional[str] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "__internal": True,
            "visible_to_user": False,
            "ok": False,
            "error": {"code": code, "message": message},
            "policy": {
                "planning_requires_tools": True,
                "final_response_must_be_plain_text": True,
            },
            "allowed_tools": self.tool_names(),
        }
        if hint:
            payload["hint"] = hint
        if tool:
            payload["tool"] = tool
        if extras:
            payload.update(extras)
        return json.dumps(payload, ensure_ascii=False)

    # ------------------------------------------------------------------- steps

    def step(self, action: Action) -> Tuple[List[str], bool]:
        observations: List[str] = []
        terminated = False
        final_plain_text = False
        used_tools = False

        if action.refusal:
            observations.append(
                self._internal_message(
                    code="parse_error",
                    message=action.refusal,
                    hint="Wrap tool calls in <tool_call> tags or reply with plain text only.",
                )
            )
            observations, terminated = self._finalize_step(
                observations=observations,
                terminated=terminated,
            )
            return observations, terminated

        else:
            tool_calls = action.tool_calls or []
            if not tool_calls:
                final_plain_text = self._final_response_or_hint(action, observations)
                terminated = final_plain_text
            else:
                observations.extend(self._run_tool_calls(tool_calls))
                used_tools = True

        observations, terminated = self._finalize_step(
            observations=observations,
            terminated=terminated,
        )
        return observations, terminated

    # ----------------------------------------------------------------- internals

    def _finalize_step(
        self,
        *,
        observations: List[str],
        terminated: bool,
    ) -> Tuple[List[str], bool]:
        self._step_index += 1

        if self._step_index >= self.max_steps:
            terminated = True

        return observations, terminated

    def _final_response_or_hint(
        self,
        action: Action,
        observations: List[str],
    ) -> bool:
        response = (action.content or "").strip()
        if response:
            return True
        observations.append(
            self._internal_message(
                code="empty_final",
                message="Final response cannot be empty when no tool calls are provided.",
                hint="Share a quick commentary(status=...) call or reply with plain text to finish.",
            )
        )
        return False

    def _run_tool_calls(self, tool_calls: Sequence[ToolCall]) -> List[str]:
        outputs: List[str] = []
        allowed = set(self.tool_names())
        for index, tool_call in enumerate(tool_calls):
            if tool_call is None:
                continue
            message = self._handle_tool_call(tool_call, index=index, allowed_tools=allowed)
            if message is not None:
                outputs.append(message)
        return outputs

    def _handle_tool_call(
        self,
        tool_call: ToolCall,
        *,
        index: int,
        allowed_tools: Set[str],
    ) -> Optional[str]:
        if tool_call.refusal:
            return self._internal_message(
                code="tool_call_error",
                message=tool_call.refusal,
                hint="Fix the tool call JSON payload or share a commentary(status=...) call before retrying.",
                extras={"tool_call_id": tool_call.call_id, "action_index": index},
            )

        name = (tool_call.name or "").strip()
        if not name:
            return self._internal_message(
                code="missing_tool_name",
                message="Tool name is required.",
                hint="Provide a function name inside the tool call payload.",
                extras={"tool_call_id": tool_call.call_id, "action_index": index},
            )

        if name not in allowed_tools:
            return self._internal_message(
                code="invalid_tool",
                message=f"Tool '{name}' is not available.",
                hint="Choose one of the allowed tools.",
                extras={"tool_call_id": tool_call.call_id, "action_index": index},
            )

        arguments = tool_call.arguments or {}
        if not isinstance(arguments, dict):
            return self._internal_message(
                code="invalid_arguments",
                message="Tool arguments must be a JSON object.",
                tool=name,
                hint="Use key/value pairs when building tool arguments.",
                extras={
                    "tool_call_id": tool_call.call_id,
                    "action_index": index,
                    "arguments": arguments,
                },
            )

        try:
            outcome = self.execute_tool(call=tool_call, context={})
        except Exception as exc:  # pragma: no cover - defensive guard
            return self._internal_message(
                code="tool_runtime_error",
                message=f"Tool '{name}' raised an exception.",
                tool=name,
                hint="Revise the arguments or share a commentary(status=...) plan check before retrying.",
                extras={
                    "tool_call_id": tool_call.call_id,
                    "action_index": index,
                    "exception": str(exc),
                },
            )

        return str(outcome)
