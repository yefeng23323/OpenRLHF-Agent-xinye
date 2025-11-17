"""High-level builders for assembling agent components."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type

from openrlhf_agent.backends import LLMEngine, OpenAIEngine
from openrlhf_agent.agentkit.environments import Environment, SingleTurnEnvironment, FunctionCallEnvironment
from openrlhf_agent.agentkit.protocols import (
    ChatProtocol,
    Qwen3InstructProtocol,
    Qwen3ThinkingProtocol,
)
from openrlhf_agent.agentkit.rewards.base import ProcessRewardStrategy, ResultRewardStrategy
from openrlhf_agent.agentkit.rewards.pipeline import RewardPipeline
from openrlhf_agent.agentkit.rewards.result_hub import MatchingReward
from openrlhf_agent.agentkit.runtime import AgentRuntime
from openrlhf_agent.agentkit.session import AgentSession


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

_DEFAULT_RESULT_REWARD = "matching"
_RESULT_REWARD_REGISTRY: Dict[str, Type[ResultRewardStrategy]] = {
    "matching": MatchingReward,
}


def register_result_reward(name: str, strategy_cls: Type[ResultRewardStrategy]) -> None:
    """Register a result reward strategy class under a readable name."""

    normalized = name.lower()
    _RESULT_REWARD_REGISTRY[normalized] = strategy_cls


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
    """Create an environment by name; returns the default when unspecified."""

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




def build_session(
    *,
    environment: Optional[Environment] = None,
    protocol: Optional[ChatProtocol] = None,
    environment_name: Optional[str] = None,
    protocol_name: Optional[str] = None,
    environment_kwargs: Optional[dict] = None,
    reward_pipeline: Optional[RewardPipeline] = None,
    result_reward: Optional[ResultRewardStrategy] = None,
    process_reward: Optional[ProcessRewardStrategy] = None,
) -> AgentSession:
    """Construct a ready-to-use agent session."""

    env = environment or build_environment(name=environment_name, **(environment_kwargs or {}))
    proto = protocol or build_protocol(protocol_name)
    pipeline = reward_pipeline or RewardPipeline(
        result_reward=result_reward,
        process_reward=process_reward,
    )
    return AgentSession(env, proto, reward_pipeline=pipeline)


def build_runtime(
    *,
    engine: Optional[LLMEngine] = None,
    environment: Optional[Environment] = None,
    protocol: Optional[ChatProtocol] = None,
    environment_name: Optional[str] = None,
    protocol_name: Optional[str] = None,
    environment_kwargs: Optional[dict] = None,
    engine_factory: Optional[Callable[..., LLMEngine]] = None,
    engine_kwargs: Optional[dict] = None,
    max_new_tokens_per_step: int = 10240,
) -> AgentRuntime:
    """Build a streaming runtime with optional overrides."""

    env = environment or build_environment(name=environment_name, **(environment_kwargs or {}))
    proto = protocol or build_protocol(protocol_name)

    if engine is None:
        factory = engine_factory or OpenAIEngine
        engine = factory(**(engine_kwargs or {}))

    return AgentRuntime(
        engine=engine,
        environment=env,
        protocol=proto,
        max_new_tokens_per_step=max_new_tokens_per_step,
    )


__all__ = [
    "build_environment",
    "build_protocol",
    "build_session",
    "build_runtime",
    "register_result_reward",
    "build_result_reward",
]
