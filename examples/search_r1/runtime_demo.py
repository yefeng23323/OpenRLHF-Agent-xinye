import asyncio
from datetime import datetime
from openrlhf_agent.backends import OpenAIEngine
from openrlhf_agent.agentkit.runtime import AgentRuntime
from openrlhf_agent.agentkit.environments import FunctionCallEnvironment
from openrlhf_agent.agentkit.protocols import Qwen3ThinkingProtocol
from openrlhf_agent.agentkit.tools import CommentaryTool, LocalSearchTool


CUSTOM_SYSTEM_PROMPT = """
Solve the question step by step. When you receive new information, think through it in detail inside <think>...</think> before continuing.
After reasoning, decide whether any tools are needed. Use tools only to verify specific parts of your reasoning or to fetch missing information—not to produce the final answer. For brief progress updates, use the commentary tool sparingly.

Work systematically and explain steps clearly. Format the final response as:
Answer: \\boxed{{$Answer}}

Knowledge cutoff: 2023-06
Current date: {date}
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
                CommentaryTool(),
                LocalSearchTool(base_url="http://localhost:8000/retrieve"),
            ],
        ),
    )
    messages = [{"role": "user", "content": "Please use the commentary function to share your thoughts, and also help me search what Python is?"}]
    async for message in agent_runtime.run_steps(messages):
        print(message)


if __name__ == "__main__":
    asyncio.run(main())
