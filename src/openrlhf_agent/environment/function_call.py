"""Default environment that supports function calls plus a hidden think tool."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from openrlhf_agent.core import Action, ToolCall
from openrlhf_agent.environment.base import Environment
from openrlhf_agent.environment.reward import RewardStrategy, make_reward
from openrlhf_agent.environment.tools import ThinkTool, ToolRegistry


SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful agent assistant.\n\n"
    "You may call tools to plan privately, but anything outside tool calls is visible to the user.\n\n"
    "Knowledge cutoff: 2023-06\n"
    "Current date: {date}\n\n"
    "Rules:\n"
    "- Use think(notes=...) when you need to plan internally; its output is hidden from the user.\n"
    "- To answer the user, provide plain text outside tool calls. That text ends the session.\n"
    "- Tool calls must be JSON objects within <tool_call></tool_call> tags."
)


class FunctionCallEnvironment(Environment):
    """Default environment with a private think tool and plain-text finals."""

    def __init__(
        self,
        *,
        max_steps: int = 32,
        result_reward: Optional[RewardStrategy] = None,
        process_reward: Optional[RewardStrategy] = None,
        result_reward_name: Optional[str] = None,
        process_reward_name: Optional[str] = None,
        reward_strategy: Optional[RewardStrategy] = None,
    ) -> None:
        resolved_result = result_reward or reward_strategy or make_reward(result_reward_name)
        resolved_process = process_reward or make_reward(process_reward_name)
        super().__init__(
            max_steps=max_steps,
            registry=ToolRegistry([ThinkTool()]),
            result_reward=resolved_result,
            process_reward=resolved_process,
        )

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(date=datetime.now().strftime("%Y-%m-%d"))

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
            "allowed_tools": self.registry.names(),
        }
        if hint:
            payload["hint"] = hint
        if tool:
            payload["tool"] = tool
        if extras:
            payload.update(extras)
        return json.dumps(payload, ensure_ascii=False)

    # ------------------------------------------------------------------- steps

    def step(
        self,
        action: Action,
        label: Optional[str] = None,
        runtime: bool = False,
    ) -> Tuple[List[str], float, bool]:
        observations: List[str] = []
        reward = 0.0
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
            return self._finalize_step(
                action=action,
                observations=observations,
                reward=reward,
                terminated=terminated,
                label=label,
                runtime=runtime,
            )

        else:
            tool_calls = action.tool_calls or []
            if not tool_calls:
                final_plain_text = self._final_response_or_hint(action, observations)
                terminated = final_plain_text
            else:
                observations.extend(self._run_tool_calls(tool_calls))
                used_tools = True

        if not runtime:
            if used_tools:
                reward += self.process_reward(action, label)  # score tool planning quality
            if final_plain_text:
                reward += self.result_reward(action, label)  # score the final answer

        return self._finalize_step(
            action=action,
            observations=observations,
            reward=reward,
            terminated=terminated,
            label=label,
        )

    # ----------------------------------------------------------------- internals

    def _finalize_step(
        self,
        *,
        action: Action,
        observations: List[str],
        reward: float,
        terminated: bool,
        label: Optional[str],
    ) -> Tuple[List[str], float, bool]:
        self._step_index += 1

        if self._step_index >= self.max_steps:
            terminated = True

        return observations, reward, terminated

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
                hint="Reply with plain text to finish or call think(...) first.",
            )
        )
        return False

    def _run_tool_calls(self, tool_calls: Sequence[ToolCall]) -> List[str]:
        outputs: List[str] = []
        allowed = set(self.registry.names())
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
                hint="Fix the tool call JSON payload and try again.",
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
            outcome = self.execute_tool(call=tool_call, context=None)
        except Exception as exc:  # pragma: no cover - defensive guard
            return self._internal_message(
                code="tool_runtime_error",
                message=f"Tool '{name}' raised an exception.",
                tool=name,
                hint="Revise the arguments or plan with think(...) before retrying.",
                extras={
                    "tool_call_id": tool_call.call_id,
                    "action_index": index,
                    "exception": str(exc),
                },
            )

        return str(outcome)


__all__ = ["FunctionCallEnvironment"]
