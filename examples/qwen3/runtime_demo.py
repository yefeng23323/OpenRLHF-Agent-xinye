import asyncio

from openrlhf_agent.backends import OpenAIEngine
from openrlhf_agent.agentkit.runtime import AgentRuntime
from openrlhf_agent.agentkit.environments import FunctionCallEnvironment
from openrlhf_agent.agentkit.protocols import Qwen3ThinkingProtocol


async def main() -> None:
    engine = OpenAIEngine(
        model="qwen3",
        base_url="http://localhost:8009/v1",
        api_key="empty",
    )
    env = FunctionCallEnvironment()
    protocol = Qwen3ThinkingProtocol()

    rt = AgentRuntime(engine, env, protocol)
    messages = [{"role": "user", "content": "Tell me a joke about programming."}]
    async for step in rt.run_steps(messages):
        print(step)


if __name__ == "__main__":
    asyncio.run(main())
