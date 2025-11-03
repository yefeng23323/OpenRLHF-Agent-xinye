# OpenRLHF-Agent

> Minimal runtime primitives that keep OpenRLHF training and inference in sync.

OpenRLHF-Agent focuses on a tiny, readable surface. One runtime (`AgentRuntime`) powers both vLLM-style inference and OpenRLHF reinforcement learning so you do not need parallel implementations.

## ✨ Highlights

- **One codepath** for OpenRLHF + inference deployments.
- **Small module set**: a single file each for runtime, environment, template, LLM wrapper, and shared types.
- **Tool-driven UX**: ships with `think` / `final` helpers for fast experimentation.
- **Example-first**: Qwen-3 demo and RL harness ready to run.

## 🧱 How it fits together

```
AgentRuntime
 ├─ Template (renders prompts, parses <tool_call>)
 ├─ Environment (tools, rewards, stop rules)
 └─ LLMEngine (model client; default OpenAI-compatible)
```

Key modules (all under `src/openrlhf_agent/`):

- `agent.py` – runtime/session orchestration.
- `environment.py` – default environment + tool helpers.
- `template.py` – Qwen-3 chat template implementation.
- `model.py` – OpenAI-compatible engine base.
- `types.py` – lightweight dataclasses for tool calls/results.
- `examples/qwen3/` – inference demo & OpenRLHF training glue.

## 🚀 Quick start

### 1. Install

```bash
git clone https://github.com/OpenRLHF/OpenRLHF-Agent.git
cd OpenRLHF-Agent
pip install -e .
# optional extras
pip install -e .[dev]        # linting & tests
pip install -e .[openrlhf]   # pulls OpenRLHF core
```

Runtime dependencies are also listed in `requirements.txt` if you prefer `pip install -r requirements.txt`.

### 2. (Optional) launch vLLM

`examples/qwen3/run_vllm.sh` starts a Qwen-3 endpoint:

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --port 8009 \
  --served-model-name qwen
```

### 3. Run the inference demo

```bash
python examples/qwen3/runtime_demo.py
```

The script wires together:

- `OpenAIEngine` (points at a vLLM/OpenAI endpoint)
- `make_environment()` (default think/final tools + reward logic)
- `make_template("qwen3")`

### 4. Plug into OpenRLHF training

`examples/qwen3/agent_func.py` exposes the required `AgentInstance` / `AgentExecutor` hooks for OpenRLHF. Call via `train_reinforce_agent.sh` after exporting your dataset path.

## 🛠️ Customize it

### Add a tool

```python
from openrlhf_agent.environment import ToolBase, DefaultEnvironment

class SearchTool(ToolBase):
    name = "search"
    description = "Query documentation."
    parameters = [{"name": "query", "type": "string", "required": True}]

    def call(self, context, **kwargs):
        return lookup(kwargs["query"])

env = DefaultEnvironment()
env.registry.register(SearchTool())
```

### Tailor the environment

Subclass `Environment` or wrap `DefaultEnvironment` to:

- override `reward_hook`
- adjust `max_steps`
- customize `_internal_obs` messaging for guardrails

### Swap prompt formatting

Create a new `Template` implementation and expose it through `make_template`. Only three methods are required: `render_system`, `render_turn`, and `extract_tool_calls_from_text`.

### Talk to a different model backend

Subclass `LLMEngine` (in `model.py`) and override `generate` + `tokenize`. Pass your engine into `AgentRuntime`.

## 📂 Project layout

```
OpenRLHF-Agent/
├── src/openrlhf_agent/
│   ├── __init__.py
│   ├── agent.py
│   ├── environment.py
│   ├── model.py
│   ├── template.py
│   └── types.py
├── examples/qwen3/
├── pyproject.toml
├── requirements.txt
├── setup.py
└── README.md
```

## 🤝 Contributing

1. Fork & clone the repo.
2. `pip install -e .[dev]`
3. Run `ruff`, `mypy`, and `pytest` (or the demos) before opening a PR.
4. Describe behaviour changes and test coverage in your PR description.

## 📄 License

Apache License 2.0.
