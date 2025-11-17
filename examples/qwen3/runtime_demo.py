from openrlhf_agent import AgentRuntime, OpenAIEngine, build_environment, build_protocol

if __name__ == "__main__":
    engine = OpenAIEngine(
        model="qwen3",
        base_url="http://0.0.0.0:8009/v1",
        api_key="empty",
    )
    env = build_environment(name="function_call")
    protocol = build_protocol(name="qwen3_thinking")

    rt = AgentRuntime(engine, env, protocol)
    messages = [{"role": "user", "content": "Tell me a joke about programming."}]
    for step in rt.run_steps(messages):
        print(step)
