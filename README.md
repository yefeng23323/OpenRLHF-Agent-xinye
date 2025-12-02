# OpenRLHF-Agent

> Consistent training and inference stack for building tool-using chat agents on OpenRLHF and vLLM.

OpenRLHF-Agent is a slim runtime for tool-using chat agents. It keeps environment orchestration, chat protocols, and model I/O identical across RL training and production inference, so the code you prototype with OpenRLHF can ship unchanged behind a vLLM/OpenAI-compatible endpoint.

## ✨ Highlights

- **Training = inference**: the same `AgentSession` flow drives resets, tool calls, and transcript rendering in both phases.
- **Small surface area**: minimal primitives (`AgentRuntime`, `Environment`, `ChatProtocol`, `LLMEngine`, shared types) that are easy to audit.
- **Tool-first**: built-ins like `commentary` show ReAct-style loops; finals stay plain-text assistant replies.
- **Proven demos**: Qwen-3 samples cover inference serving, RL data collection, and REINFORCE++ training.
- **OpenRLHF-ready**: drop `AgentRuntime` into `train_reinforce_agent.sh` or Ray jobs without extra glue.

## 🧭 Why this matters

Teams need agents that plan actions, call tools, and stay consistent between experiments and production. Use this stack to:

- Iterate on multi-step reasoning with reward shaping and safety hooks.
- Deploy the same prompt + tool logic to live inference endpoints.
- Extend agents with memory, search, or enterprise tools while keeping one runtime abstraction.

## 🧱 Architecture

```
AgentRuntime
 ├─ AgentSession  (shared rollouts for training + inference)
 ├─ ChatProtocol  (prompt rendering, tool parsing)
 ├─ Environment   (state, rewards, injected tools)
 └─ LLMEngine     (token streaming via OpenAI/vLLM/custom)
```

See `docs/ARCHITECTURE.md` for a deeper dive into how these modules interact.

## 🚀 Quick start

### 1. Install

```bash
git clone https://github.com/OpenRLHF/OpenRLHF-Agent.git
cd OpenRLHF-Agent
pip install -e .
# optional extras
pip install -e .[dev]        # linting & tests
pip install -e .[openrlhf]   # pulls OpenRLHF core packages
```

Runtime dependencies are also listed in `requirements.txt` if you prefer `pip install -r requirements.txt`.

### 2. Run inference

Optionally launch vLLM for local serving (`pip install vllm`), then start a Qwen-3 endpoint such as:

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --port 8009 \
  --served-model-name qwen \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.8
```

Then run the reference inference demo:

```bash
python examples/qwen3/runtime_demo.py
```

The script wires together:

- `OpenAIEngine` pointing at a vLLM/OpenAI-compatible endpoint.
- `FunctionCallEnvironment` with the `commentary` tool, feedback hooks, and plain-text finals.
- `build_protocol("qwen3_thinking")` for prompt rendering and `<tool_call>` parsing.

You will see tool traces and the final answer printed to the console.

### 3. Train with OpenRLHF

`examples/qwen3/agent_func.py` exposes the `AgentInstance` / `AgentExecutor` hooks required by OpenRLHF. Run `examples/qwen3/train_reinforce_agent.sh` (set `DATASET_PATH`) or integrate the functions into your own Ray jobs to collect trajectories and train policies.

### 4. Customize

Start from the built-in abstractions—tools, environments, protocols, rewards—and extend them as needed:

#### 4.1. Add a tool

1. Subclass `ToolBase` from `src/openrlhf_agent/agentkit/tools/base.py`.
2. Implement `async def call(self, context, **kwargs)` to return visible output or structured JSON.
3. Pass the tool into your environment (`FunctionCallEnvironment(tools=[...])`) or register it dynamically via `env.register_tool(...)`.

#### 4.2. Shape rewards

- Compose a `RewardPipeline` with result/process strategies from `src/openrlhf_agent/agentkit/rewards/` (e.g., `MatchingReward`, `MathMatchingReward` for symbolic math equivalence).
- Pass the pipeline into `AgentSession(..., reward_pipeline=...)` so each `step_from_text` call can emit scalar rewards during RL training.

#### 4.3. Ship a new chat protocol

- Subclass `ChatProtocol` in `src/openrlhf_agent/agentkit/protocols/base.py`.
- Implement render + parse helpers for your provider format.
- Expose it via `build_protocol` and pass it into `AgentRuntime`.

## 📄 License

Apache License 2.0.
