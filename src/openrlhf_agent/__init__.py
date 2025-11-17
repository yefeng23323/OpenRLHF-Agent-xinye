"""
OpenRLHF Agent runtime and supporting building blocks.

Top-level layout:
- `agentkit/` hosts environments, protocols, rewards/tools, the session buffer, runtime loop, and helper factories.
- `backends/` wraps provider-specific language model engines.
- `utils/types/` stores the shared Action/Message/ToolCall dataclasses used across modules.
"""

__version__ = "0.0.1"

from openrlhf_agent.agentkit import build_environment, build_protocol
from openrlhf_agent.agentkit.environments import Environment
from openrlhf_agent.agentkit.environments.hub import FunctionCallEnvironment, SingleTurnEnvironment
from openrlhf_agent.agentkit.protocols import ChatProtocol
from openrlhf_agent.agentkit.runtime import AgentRuntime
from openrlhf_agent.agentkit.session import AgentSession
from openrlhf_agent.backends import LLMEngine, OpenAIEngine

__all__ = [
    "__version__",
    "AgentRuntime",
    "AgentSession",
    "Environment",
    "FunctionCallEnvironment",
    "SingleTurnEnvironment",
    "LLMEngine",
    "OpenAIEngine",
    "ChatProtocol",
    "build_environment",
    "build_protocol",
]
