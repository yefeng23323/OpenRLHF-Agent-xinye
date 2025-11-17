"""
OpenRLHF Agent runtime and supporting building blocks.

Top-level layout:
- `agentkit/` hosts environments, protocols, rewards/tools, the session buffer, runtime loop, and helper factories.
- `backends/` wraps provider-specific language model engines.
"""

__version__ = "0.0.1"

from openrlhf_agent.agentkit import build_environment, build_protocol
from openrlhf_agent.agentkit.runtime import AgentRuntime
from openrlhf_agent.agentkit.session import AgentSession
from openrlhf_agent.backends import OpenAIEngine

__all__ = [
    "__version__",
    "AgentRuntime",
    "AgentSession",
    "OpenAIEngine",
    "build_environment",
    "build_protocol",
]
