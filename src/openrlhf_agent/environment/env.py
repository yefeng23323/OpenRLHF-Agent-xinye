import re
import json

from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

from .base import Environment
from .tools.registry import ToolRegistry
from .tools.builtin.think import ThinkTool
from .tools.builtin.final import FinalTool

from openrlhf_agent.utils.types import ToolResult, ToolCall


SYSTEM_TEMPLATE = """
You are a helpful agent assistant.

You may draft or think internally, but anything outside tool calls is invisible to the user.
All user-visible communication must go through tool calls.

Knowledge cutoff: 2023-06
Current date: {date}

Rules:
- One tool call per step. And each step with exactly one <tool_call>…</tool_call>.
- Free text before/after the tool call is treated as internal and will not be shown.
- If information is missing or you need to plan, call think(...) before calling final(...).
- Use final(...) only to deliver the final user-facing answer and end the task.
""".strip()


def current_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def extract_verdict(text: str) -> str | None:
    """Return '[[A]]' or '[[B]]' if found; otherwise None."""
    found = re.findall(r"\[\[(A|B)\]\]", text)
    return f"[[{found[-1]}]]" if found else None


def compute_reward(
    response,
    target,  # [[A]] or [[B]]
    correct_score=1.0,
    verdict_correct_score=0.1,
    format_score=0.0,
):
    if response == target.strip():
        return correct_score
    elif extract_verdict(response) == target.strip():
        return verdict_correct_score
    else:
        return format_score


class DefaultEnvironment(Environment):
    def __init__(self, *, max_steps: int = 66):
        self.registry = ToolRegistry(tools=[ThinkTool(), FinalTool()])
        self._init_observation = None
        self.max_steps = max_steps

    # ---------------- helpers ----------------

    def _internal_obs(
        self,
        code: str,
        message: str,
        *,
        hint: Optional[str] = None,
        tool: Optional[str] = None,
        extras: Optional[Dict[str, Any]] = None
    ) -> str:
        """Return an INTERNAL (non user-visible) observation."""
        payload: Dict[str, Any] = {
            "__internal": True,
            "ok": False,
            "visible_to_user": False,
            "error": {"code": code, "message": message},
            "policy": {
                "user_visible_only_via_tools": True,
                "must_end_with_single_tool_call": True,
            },
            "next_action_suggestion": {
                "name": "think",
                "arguments_schema": {"notes": "string"},
                "why": "Close the step with a single tool call; plan briefly before proceeding."
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

    # ---------------- Boilerplate ----------------

    @property
    def system_prompt(self):
        return SYSTEM_TEMPLATE.format(date=current_date())

    def tools_manifest(self) -> List[Dict[str, Any]]:
        return self.registry.list_openai_tools()

    def execute_tool(self, name: str, args: Dict[str, Any], context: Dict[str, Any] = {}) -> ToolResult:
        context = context or {}
        tool = self.registry.get(name)
        out = tool.call(context=context, **(args or {}))
        return ToolResult(
            tool_call_id=context.get("tool_call_id", ""),
            content=str(out),
            is_error=False,
        )

    # --- RL hooks -------------------------------------------------------------

    def reset(self, observation: str | None = None) -> None:
        self._step_idx = 0
        self._init_observation = observation

    def reward_hook(self, tool_name: str, tool_args: Dict[str, Any], label: str | None) -> float: # type: ignore
        if label is None:
            return 0.0
        if tool_name == "think":
            return 0.0
        if tool_name == "final":
            pred = (tool_args.get("answer") or "").strip()
            return compute_reward(pred, label)
        return 0.0
    
    # --- Main step ------------------------------------------------------------

    def step(
        self,
        actions: Optional[List[ToolCall]] = None,
        label: Optional[str] = None,
        runtime: bool = False
    ) -> Tuple[List[str], float, bool]:
        # No tool calls captured -> internal (non user-visible) hint
        if actions is None:
            obs = self._internal_obs(
                code="no_tool_call",
                message="No tool call captured this step. Close the step with exactly one tool call.",
                hint='Prefer think(notes=...) if you need to plan before final.'
            )
            return [obs], 0.0, False

        allowed_tools = self.registry.names()

        next_observations: List[str] = []
        total_reward: float = 0.0
        terminated: bool = False
        for idx, action in enumerate(actions):
            # If already terminated by a prior 'final', ignore subsequent actions
            if terminated:
                next_observations.append(self._internal_obs(
                    code="ignored_after_final",
                    message=f"Action #{idx} ignored because 'final' has already been called.",
                    extras={"action_index": idx, "allowed_tools": allowed_tools}
                ))
                continue

            # Validate action object and name
            if action is None:
                next_observations.append(self._internal_obs(
                    code="invalid_action_format",
                    message=f"Action #{idx} is missing a valid tool name.",
                    hint='Use <tool_call>{"name": "...", "arguments": {...}}</tool_call>.',
                    extras={"action_index": idx}
                ))
                continue

            name = action.name
            args = action.arguments

            # Validate tool name
            if name not in allowed_tools:
                next_observations.append(self._internal_obs(
                    code="invalid_action",
                    message=f"Tool '{name}' is not allowed.",
                    tool=name,
                    hint="Use only tools listed in 'allowed_tools'.",
                    extras={"action_index": idx}
                ))
                continue

            # Validate required arguments against tool schema
            tool = self.registry.get(name)
            required_params = [p["name"] for p in getattr(tool, "parameters", []) if p.get("required")]
            missing = [p for p in required_params if (p not in args) or (args[p] is None) or (isinstance(args[p], str) and args[p].strip() == "")]
            if missing:
                next_observations.append(self._internal_obs(
                    code="invalid_arguments",
                    message=f"Missing required arguments: {missing}",
                    tool=name,
                    hint="Fill all required fields and try again.",
                    extras={"required": required_params, "received_keys": list(args.keys()), "action_index": idx}
                ))
                continue

            # Execute tool (runtime errors are internal)
            try:
                tool_result = self.execute_tool(name=name, args=args)
                obs_text = tool_result.content  # visible text (from tools) goes straight to UI layer
            except Exception as e:
                next_observations.append(self._internal_obs(
                    code="tool_runtime_error",
                    message=f"Tool '{name}' raised an exception.",
                    tool=name,
                    hint="Revise inputs or plan with think(...) before retrying.",
                    extras={"exception": str(e), "action_index": idx}
                ))
                continue

            # Record observation for this action
            next_observations.append(obs_text)

            # Reward per action (usually only 'final' yields score)
            if not runtime:
                # try:
                total_reward += self.reward_hook(name, args, label)
                # except Exception as e:
                #     # Reward errors should not break the loop; surface internally
                #     next_observations.append(self._internal_obs(
                #         code="reward_hook_error",
                #         message=f"reward_hook failed for tool '{name}'.",
                #         tool=name,
                #         extras={"exception": str(e), "action_index": idx}
                #     ))
            
            if name == "final":
                terminated = True

        self._step_idx += 1
        if self._step_idx >= self.max_steps:
            terminated = True

        return next_observations, total_reward, terminated
