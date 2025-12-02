"""High-level builders for assembling agent components."""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

from openrlhf_agent.agentkit.environments import (
    Environment,
    SingleTurnEnvironment,
    FunctionCallEnvironment
)
from openrlhf_agent.agentkit.protocols import (
    ChatProtocol,
    Qwen3InstructProtocol,
    Qwen3ThinkingProtocol,
)
from openrlhf_agent.agentkit.rewards.process_rewards import (
    ProcessRewardStrategy,
    ToolCallReward,
)
from openrlhf_agent.agentkit.rewards.result_rewards import (
    ResultRewardStrategy,
    MatchingReward,
    MathMatchingReward,
    GRMJudgeReward,
)


_DEFAULT_ENVIRONMENT = "single_turn"
_ENVIRONMENT_REGISTRY: Dict[str, Type[Environment]] = {
    "single_turn": SingleTurnEnvironment,
    "function_call": FunctionCallEnvironment,
}

_DEFAULT_PROTOCOL = "qwen3_instruct"
_PROTOCOL_REGISTRY: Dict[str, Type[ChatProtocol]] = {
    "qwen3_instruct": Qwen3InstructProtocol,
    "qwen3_thinking": Qwen3ThinkingProtocol,
}

_DEFAULT_PROCESS_REWARD = "tool_call"
_PROCESS_REWARD_REGISTRY: Dict[str, Type[ProcessRewardStrategy]] = {
    "tool_call": ToolCallReward,
}

_DEFAULT_RESULT_REWARD = "matching"
_RESULT_REWARD_REGISTRY: Dict[str, Type[ResultRewardStrategy]] = {
    "matching": MatchingReward,
    "math_matching": MathMatchingReward,
    "grm": GRMJudgeReward,
}


def build_process_reward(
    name: Optional[str] = None,
    *,
    config: Optional[dict] = None,
) -> ProcessRewardStrategy:
    """Instantiate a process reward strategy via the registry."""

    resolved = (name or _DEFAULT_PROCESS_REWARD).lower()
    try:
        reward_cls = _PROCESS_REWARD_REGISTRY[resolved]
    except KeyError as exc:
        raise ValueError(f"Unknown reward strategy '{name}'.") from exc

    payload = dict(config or {})
    return reward_cls(**payload)


def build_result_reward(
    name: Optional[str] = None,
    *,
    config: Optional[dict] = None,
) -> ResultRewardStrategy:
    """Instantiate a result reward strategy via the registry."""

    resolved = (name or _DEFAULT_RESULT_REWARD).lower()
    try:
        reward_cls = _RESULT_REWARD_REGISTRY[resolved]
    except KeyError as exc:
        raise ValueError(f"Unknown reward strategy '{name}'.") from exc

    payload = dict(config or {})
    return reward_cls(**payload)


def build_environment(name: Optional[str] = None, **kwargs: Any) -> Environment:
    """Create an environment by name."""

    resolved = (name or _DEFAULT_ENVIRONMENT).lower()
    try:
        env_cls = _ENVIRONMENT_REGISTRY[resolved]
    except KeyError as exc:
        raise ValueError(f"Unknown environment '{name}'.") from exc
    return env_cls(**kwargs)


def build_protocol(name: Optional[str] = None) -> ChatProtocol:
    """Return a chat protocol instance by name."""

    resolved_name = (name or _DEFAULT_PROTOCOL).lower()
    try:
        protocol_cls = _PROTOCOL_REGISTRY[resolved_name]
    except KeyError as exc:
        raise ValueError(f"Unknown chat protocol '{name}'.") from exc
    return protocol_cls()
