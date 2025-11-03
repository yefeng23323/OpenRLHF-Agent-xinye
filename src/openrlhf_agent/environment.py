import json
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .types import ToolCall, ToolResult


class ToolBase(ABC):
    """Minimal function-style tool definition."""

    name: str
    description: str
    parameters: List[Dict[str, Any]]

    def openai_tool(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    def call(self, context: Dict[str, Any], **kwargs) -> str:
        raise NotImplementedError


class ToolRegistry:
    def __init__(self, tools: Sequence[ToolBase]):
        self._tools: Dict[str, ToolBase] = {tool.name: tool for tool in tools}

    def register(self, tool: ToolBase) -> None:
        self._tools[tool.name] = tool

    def names(self) -> Iterable[str]:
        return list(self._tools.keys())

    def list_openai_tools(self) -> List[Dict[str, Any]]:
        return [tool.openai_tool() for tool in self._tools.values()]

    def get(self, name: str) -> ToolBase:
        return self._tools[name]


class ThinkTool(ToolBase):
    name = "think"
    description = "Write down private notes before taking a visible action."
    parameters = [
        {
            "name": "notes",
            "type": "string",
            "description": "Short plan or reasoning that stays internal.",
            "required": True,
        }
    ]

    def call(self, context: Dict[str, Any], **kwargs) -> str:
        return kwargs.get("notes", "")


class FinalTool(ToolBase):
    name = "final"
    description = "Deliver the final user-facing answer and terminate."
    parameters = [
        {
            "name": "answer",
            "type": "string",
            "description": "Message to show to the user.",
            "required": True,
        }
    ]

    def call(self, context: Dict[str, Any], **kwargs) -> str:
        return kwargs.get("answer", "")


class Environment(ABC):
    """Abstract interface for agent environments."""

    @property
    @abstractmethod
    def max_steps(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def tools_manifest(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def execute_tool(self, name: str, args: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        raise NotImplementedError

    @abstractmethod
    def reward_hook(self, tool_name: str, tool_args: Dict[str, Any], label: Optional[str]) -> float:
        raise NotImplementedError

    @abstractmethod
    def reset(self, observation: Optional[str] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        actions: Optional[List[Optional[ToolCall]]],
        label: Optional[str] = None,
        runtime: bool = False,
    ) -> Tuple[List[str], float, bool]:
        raise NotImplementedError


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


def extract_verdict(text: str) -> Optional[str]:
    """Return '[[A]]' or '[[B]]' if found; otherwise None."""
    found = re.findall(r"\[\[(A|B)\]\]", text)
    return f"[[{found[-1]}]]" if found else None


def compute_reward(
    response: str,
    target: str,
    *,
    correct_score: float = 1.0,
    verdict_correct_score: float = 0.1,
    format_score: float = 0.0,
) -> float:
    if response == target.strip():
        return correct_score
    if extract_verdict(response) == target.strip():
        return verdict_correct_score
    return format_score


class ReActEnvironment(Environment):
    """Default environment with think/final tool semantics."""

    def __init__(self, *, max_steps: int = 66):
        self.registry = ToolRegistry([ThinkTool(), FinalTool()])
        self._init_observation: Optional[str] = None
        self._max_steps = max_steps
        self._step_idx = 0

    # ------------------------------------------------------------------ proxy

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def system_prompt(self) -> str:
        current_date = datetime.now().strftime("%Y-%m-%d")
        return SYSTEM_TEMPLATE.format(date=current_date)

    def tools_manifest(self) -> List[Dict[str, Any]]:
        return self.registry.list_openai_tools()

    def execute_tool(self, name: str, args: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        tool = self.registry.get(name)
        out = tool.call(context=context, **(args or {}))
        return ToolResult(
            tool_call_id=context.get("tool_call_id", ""),
            content=str(out),
            is_error=False,
        )

    # -------------------------------------------------------------- lifecycle

    def reset(self, observation: Optional[str] = None) -> None:
        self._step_idx = 0
        self._init_observation = observation

    def reward_hook(self, tool_name: str, tool_args: Dict[str, Any], label: Optional[str]) -> float:
        if label is None:
            return 0.0
        if tool_name == "think":
            return 0.0
        if tool_name == "final":
            pred = (tool_args.get("answer") or "").strip()
            return compute_reward(pred, label)
        return 0.0

    # --------------------------------------------------------------- stepping

    def _internal_obs(
        self,
        code: str,
        message: str,
        *,
        hint: Optional[str] = None,
        tool: Optional[str] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> str:
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
                "why": "Close the step with a single tool call; plan briefly before proceeding.",
            },
            "allowed_tools": list(self.registry.names()),
        }
        if hint:
            payload["hint"] = hint
        if tool:
            payload["tool"] = tool
        if extras:
            payload.update(extras)
        return json.dumps(payload, ensure_ascii=False)

    def step(
        self,
        actions: Optional[List[Optional[ToolCall]]],
        label: Optional[str] = None,
        runtime: bool = False,
    ) -> Tuple[List[str], float, bool]:
        if actions is None:
            obs = self._internal_obs(
                code="no_tool_call",
                message="No tool call captured this step. Close the step with exactly one tool call.",
                hint="Prefer think(notes=...) if you need to plan before final.",
            )
            return [obs], 0.0, False

        allowed_tools = set(self.registry.names())
        next_observations: List[str] = []
        total_reward = 0.0
        terminated = False

        for idx, action in enumerate(actions):
            if terminated:
                next_observations.append(
                    self._internal_obs(
                        code="ignored_after_final",
                        message=f"Action #{idx} ignored because 'final' has already been called.",
                        extras={"action_index": idx, "allowed_tools": list(allowed_tools)},
                    )
                )
                continue

            if action is None:
                next_observations.append(
                    self._internal_obs(
                        code="invalid_action_format",
                        message=f"Action #{idx} is missing a valid tool name.",
                        hint='Use <tool_call>{"name": "...", "arguments": {...}}</tool_call>.',
                        extras={"action_index": idx},
                    )
                )
                continue

            name = action.name
            args = action.arguments

            if name not in allowed_tools:
                next_observations.append(
                    self._internal_obs(
                        code="invalid_action",
                        message=f"Tool '{name}' is not allowed.",
                        tool=name,
                        hint="Use only tools listed in 'allowed_tools'.",
                        extras={"action_index": idx},
                    )
                )
                continue

            tool = self.registry.get(name)
            required = [p["name"] for p in tool.parameters if p.get("required")]
            missing = [
                param
                for param in required
                if (param not in args)
                or (args[param] is None)
                or (isinstance(args[param], str) and args[param].strip() == "")
            ]
            if missing:
                next_observations.append(
                    self._internal_obs(
                        code="invalid_arguments",
                        message=f"Missing required arguments: {missing}",
                        tool=name,
                        hint="Fill all required fields and try again.",
                        extras={
                            "required": required,
                            "received_keys": list(args.keys()),
                            "action_index": idx,
                        },
                    )
                )
                continue

            try:
                context = {"tool_call_id": action.id}
                tool_result = self.execute_tool(name=name, args=args, context=context)
                obs_text = tool_result.content
            except Exception as exc:  # pragma: no cover - defensive
                next_observations.append(
                    self._internal_obs(
                        code="tool_runtime_error",
                        message=f"Tool '{name}' raised an exception.",
                        tool=name,
                        hint="Revise inputs or plan with think(...) before retrying.",
                        extras={"exception": str(exc), "action_index": idx},
                    )
                )
                continue

            next_observations.append(obs_text)

            if not runtime:
                total_reward += self.reward_hook(name, args, label)

            if name == "final":
                terminated = True

        self._step_idx += 1
        if self._step_idx >= self.max_steps:
            terminated = True

        return next_observations, total_reward, terminated


def make_environment(name: Optional[str] = None, **kwargs: Any) -> Environment:
    if name in (None, "", "default"):
        return ReActEnvironment(**kwargs)
    raise ValueError(f"Unknown environment '{name}'.")


__all__ = ["Environment", "ReActEnvironment", "make_environment"]
