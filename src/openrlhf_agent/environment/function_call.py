"""Default environment that supports function calls plus a hidden think tool."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from openrlhf_agent.core import ParsedAssistantAction
from openrlhf_agent.environment.base import Environment
from openrlhf_agent.environment.reward import RewardStrategy, make_reward
from openrlhf_agent.environment.tools import FinalTool, ThinkTool, ToolRegistry


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
        reward_config: Optional[Dict[str, float]] = None,
        reward_strategy: Optional[RewardStrategy] = None,
        reward_name: Optional[str] = None,
    ) -> None:
        super().__init__(
            max_steps=max_steps,
            registry=ToolRegistry([ThinkTool()]),
        )

        self._reward = make_reward(
            reward_name,
            config=reward_config,
            strategy=reward_strategy,
        )

    # ----------------------------------------------------------------- tooling

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(date=datetime.now().strftime("%Y-%m-%d"))

    def reward_hook(self, action: ParsedAssistantAction, label: Optional[str]) -> float:
        return self._reward.reward_from_action(action, label)

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
        action: ParsedAssistantAction,
        label: Optional[str] = None,
        runtime: bool = False,
    ) -> Tuple[List[str], float, bool, Optional[str]]:
        observations: List[str] = []
        reward = 0.0
        terminated = False
        final_response: Optional[str] = None

        if action.refusal:
            observations.append(
                self._internal_message(
                    code="parse_error",
                    message=action.refusal,
                    hint="Wrap tool calls in <tool_call> tags or reply with plain text only.",
                )
            )
            self._step_index += 1
            return observations, reward, terminated, final_response

        if not action.tool_calls:
            response = (action.content or "").strip()
            if not response:
                observations.append(
                    self._internal_message(
                        code="empty_final",
                        message="Final response cannot be empty when no tool calls are provided.",
                        hint="Reply with plain text to finish or call think(...) first.",
                    )
                )
                self._step_index += 1
                return observations, reward, terminated, None

            final_response = response
            terminated = True
        else:
            allowed_tools = set(self.registry.names())

            for index, tool_call in enumerate(action.tool_calls):
                if tool_call is None:
                    continue

                if tool_call.refusal:
                    observations.append(
                        self._internal_message(
                            code="tool_call_error",
                            message=tool_call.refusal,
                            hint="Fix the tool call JSON payload and try again.",
                            extras={"tool_call_id": tool_call.id, "action_index": index},
                        )
                    )
                    continue

                name = (tool_call.name or "").strip()
                if not name:
                    observations.append(
                        self._internal_message(
                            code="missing_tool_name",
                            message="Tool name is required.",
                            hint="Provide a function name inside the tool call payload.",
                            extras={"tool_call_id": tool_call.id, "action_index": index},
                        )
                    )
                    continue

                if name not in allowed_tools:
                    observations.append(
                        self._internal_message(
                            code="invalid_tool",
                            message=f"Tool '{name}' is not available.",
                            hint="Choose one of the allowed tools.",
                            extras={"tool_call_id": tool_call.id, "action_index": index},
                        )
                    )
                    continue

                arguments = tool_call.arguments or {}
                if not isinstance(arguments, dict):
                    observations.append(
                        self._internal_message(
                            code="invalid_arguments",
                            message="Tool arguments must be a JSON object.",
                            tool=name,
                            hint="Use key/value pairs when building tool arguments.",
                            extras={
                                "tool_call_id": tool_call.id,
                                "action_index": index,
                                "arguments": arguments,
                            },
                        )
                    )
                    continue

                try:
                    outcome = self.execute_tool(call=tool_call, context=None)
                except Exception as exc:  # pragma: no cover - defensive guard
                    observations.append(
                        self._internal_message(
                            code="tool_runtime_error",
                            message=f"Tool '{name}' raised an exception.",
                            tool=name,
                            hint="Revise the arguments or plan with think(...) before retrying.",
                            extras={
                                "tool_call_id": tool_call.id,
                                "action_index": index,
                                "exception": str(exc),
                            },
                        )
                    )
                    continue

                # if name == ThinkTool.name:
                #     observations.append(
                #         self._internal_message(
                #             code="think_notes",
                #             message="Captured private notes via think().",
                #             tool=name,
                #             extras={"notes": outcome, "action_index": index},
                #         )
                #     )
                # else:
                observations.append(str(outcome))

        self._step_index += 1

        if not runtime:
            reward += self.reward_hook(action, label)

        if self._step_index >= self.max_steps:
            terminated = True

        return observations, reward, terminated, final_response


__all__ = ["FunctionCallEnvironment"]
