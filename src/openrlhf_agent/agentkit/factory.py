"""High-level builders for assembling agent components."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type

from openrlhf_agent.backends import LLMEngine, OpenAIEngine
from openrlhf_agent.agentkit.environments import Environment, FunctionCallEnvironment, SingleTurnEnvironment
from openrlhf_agent.agentkit.protocols import (
    ChatProtocol,
    Qwen3InstructProtocol,
    Qwen3ThinkingProtocol,
)
from openrlhf_agent.agentkit.rewards import (
    ProcessRewardStrategy,
    ResultRewardStrategy,
    RewardPipeline,
)
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
]
