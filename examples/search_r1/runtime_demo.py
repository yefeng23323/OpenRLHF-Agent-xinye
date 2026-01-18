import asyncio
from datetime import datetime
from openrlhf_agent.backends import OpenAIEngine
from openrlhf_agent.agentkit.runtime import AgentRuntime
from openrlhf_agent.agentkit.environments import FunctionCallEnvironment
from openrlhf_agent.agentkit.protocols import Qwen3ThinkingProtocol
from openrlhf_agent.agentkit.tools import LocalSearchTool


CUSTOM_SYSTEM_PROMPT = """
You are a helpful assistant.

Your Knowledge cutoff: 2023-06
Current date: {date}

Work systematically and explain steps clearly. Format the final response as:
Answer: \\boxed{{$Answer}}
""".strip()


async def main() -> None:
    agent_runtime = AgentRuntime(
        protocol=Qwen3ThinkingProtocol(), # qwen3-thinking
        engine=OpenAIEngine(
            model="qwen3", 
            base_url="http://localhost:8009/v1",
            api_key="empty"
        ),
        environment=FunctionCallEnvironment(
            system_prompt=CUSTOM_SYSTEM_PROMPT.format(date=datetime.now().strftime("%Y-%m-%d")),
            tools=[
                LocalSearchTool(base_url="http://localhost:8000/retrieve"),
            ],
        ),
    )
    messages = [{"role": "user", "content": "Please use the commentary tool to share your thoughts, and use local_search to find what Python is."}]
    async for message in agent_runtime.run_steps(messages):
        print(message)


if __name__ == "__main__":
    asyncio.run(main())
