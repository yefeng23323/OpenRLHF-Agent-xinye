# OpenRLHF-Agent

> Consistent training and inference stack for building tool-using chat agents on OpenRLHF and vLLM.

OpenRLHF-Agent provides a shared runtime that covers environment orchestration, chat protocols, and model I/O for both RL training and production inference. Teams can prototype an agent policy with OpenRLHF, then ship the same codepath behind a chatbot powered by vLLM or any OpenAI-compatible endpoint.

## ✨ Highlights

- **Training and inference stay aligned**: the identical `AgentSession` flow drives resets, tool calls, and transcript rendering across phases.
- **Lean agent primitives**: a minimal set of modules (`AgentRuntime`, `Environment`, `ChatProtocol`, `LLMEngine`, and shared core models) keeps the runtime easy to audit and extend.
- **Tool-centric design**: bundled `commentary` helper demonstrates ReAct-style loops while final answers ship as plain assistant text.
- **Production-ready examples**: Qwen-3 samples cover inference serving, RL data collection, and REINFORCE++ training.
- **Optimized for OpenRLHF**: plug `AgentRuntime` into `train_reinforce_agent.sh` or Ray jobs without extra glue code.

## 🧭 Why this matters

Chat assistants are shifting from passive Q&A toward autonomous task execution. Leading providers now expose agent modes that plan actions, invoke tools, and maintain long-lived context. OpenRLHF-Agent focuses on the engineering glue needed to keep those behaviors consistent between experimentation and deployment. Use it to:

- Iterate on multi-step reasoning policies with reward shaping and safety hooks before you ship.
- Connect the same prompt strategy to live inference endpoints without rewriting tool logic.
- Extend agents with memory stores, search APIs, or enterprise tools while staying within a single runtime abstraction.

## 🧱 Architecture

```
AgentRuntime
 ├─ AgentSession  (shared rollouts for training + inference)
 ├─ ChatProtocol  (prompt rendering, tool parsing)
 ├─ Environment   (state, rewards, injected tools)
 └─ LLMEngine     (token streaming via OpenAI/vLLM/custom)
```

### Where the pieces live

- `src/openrlhf_agent/utils/types.py`: domain models (`Action`, `ToolCall`, `Observation`, `Conversation`) shared across the stack.
- `src/openrlhf_agent/agentkit/session.py`: keeps chat history in sync with tool manifests and converts environment outputs back into prompts.
- `src/openrlhf_agent/agentkit/runtime.py`: streaming runtime loop that binds the `LLMEngine` to an `AgentSession`.
- `src/openrlhf_agent/agentkit/environments/`: base class plus `hub/function_call.py` (tool calling + plain-text finals) and `hub/single_turn.py`.
- `src/openrlhf_agent/agentkit/tools/`: `ToolBase` and bundled helpers such as `CommentaryTool`, `FinalTool`, and `ThinkTool`.
- `src/openrlhf_agent/agentkit/rewards/`: result/process reward strategies and the `RewardPipeline` glue used during RL training.
- `src/openrlhf_agent/agentkit/protocols/`: provider codecs and templates (Qwen-3 instruct and thinking modes live in `hub/`).
- `src/openrlhf_agent/backends/`: `LLMEngine` contract and the OpenAI/vLLM-compatible HTTP client (`hub/openai.py`).
- `examples/qwen3/`: runnable demos for inference serving, OpenRLHF data collection, and REINFORCE++ training loops.

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

### 2. (Optional) launch vLLM

Install vLLM (e.g. `pip install vllm>=0.10.2`) and start the Qwen-3 endpoint via `examples/qwen3/run_vllm.sh`, or run it manually:

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --port 8009 \
  --served-model-name qwen \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.8
```

### 3. Run the inference demo

```bash
python examples/qwen3/runtime_demo.py
```

The script wires together:

- `OpenAIEngine` pointing at a vLLM/OpenAI-compatible endpoint.
- `FunctionCallEnvironment` with the `commentary` tool, feedback hooks, and plain-text finals.
- `build_protocol("qwen3_thinking")` for prompt rendering and `<tool_call>` parsing.

You will see tool traces and the final answer printed to the console.

### 4. Plug into OpenRLHF training

`examples/qwen3/agent_func.py` exposes the `AgentInstance` / `AgentExecutor` hooks required by OpenRLHF. Run `examples/qwen3/train_reinforce_agent.sh` (set `DATASET_PATH`) or integrate the functions into your own Ray jobs to collect trajectories and train policies.

Need a reward model hosted behind a GRM-compatible endpoint? Use `examples/qwen3/agent_func_grm.py`, export the GRM connection info, and point OpenRLHF at this module instead:

```bash
export GRM_BASE_URL="https://your-grm-endpoint/v1"
export GRM_API_KEY="sk-..."
export GRM_MODEL="grm-judge"
# edit examples/qwen3/train_reinforce_agent.sh so that
#   AGNET_FUNC_PATH=".../examples/qwen3/agent_func_grm.py"
# then launch your preferred OpenRLHF training entrypoint
```

## 🛠️ Customize the stack

### Add a tool

1. Subclass `ToolBase` from `src/openrlhf_agent/agentkit/tools/base.py`.
2. Implement `call(self, context, **kwargs)` to return visible output or structured JSON.
3. Pass the tool into your environment (`FunctionCallEnvironment(tools=[...])`) or register it dynamically via `env.register_tool(...)`.

### Tailor the environment

- Subclass `Environment` from `src/openrlhf_agent/agentkit/environments/base.py` to decide how tool calls are validated, executed, and terminated.
- Customize prompts, `max_steps`, and default tools when instantiating `FunctionCallEnvironment` (see `hub/function_call.py` for reference).
- Return structured observation strings to emit internal guardrail hints or UI-visible tool outputs.

### Shape rewards

- Compose a `RewardPipeline` with result/process strategies from `src/openrlhf_agent/agentkit/rewards/` (e.g., `MatchingReward`).
- Pass the pipeline into `AgentSession(..., reward_pipeline=...)` so each `step_from_text` call can emit scalar rewards during RL training.

### Ship a new chat protocol

- Subclass `ChatProtocol` in `src/openrlhf_agent/agentkit/protocols/base.py`.
- Implement render + parse helpers for your provider format.
- Expose it via `build_protocol` and pass it into `AgentRuntime`.

### Support another engine

- Subclass `LLMEngine` in `src/openrlhf_agent/backends/base.py`.
- Implement `generate` and `tokenize` for your provider.
- Instantiate the engine and supply it to `AgentRuntime`.

## 🌅 Project vision

OpenRLHF-Agent is the open-source bridge between RLHF-style training loops and production-grade agent deployments. By aligning tool schemas, prompts, and environment contracts, it lowers the barrier for teams that want to:

- Train agents with reward-driven planning, self-monitoring, and safety checks.
- Deploy those agents behind proactive chat products without reimplementing logic.
- Experiment with emerging agent patterns (long-term memory, hierarchical planners, multi-agent collaboration) while keeping a maintainable codebase.

## 🤝 Contributing

1. Fork and clone the repo.
2. Install dev dependencies: `pip install -e .[dev]`.
3. Run `ruff`, `mypy`, and `pytest` (or the demos) before submitting.
4. Confirm the Qwen-3 training and inference demos still run.
5. Open a PR summarizing behaviour changes and test coverage.

## 📄 License

Apache License 2.0.
