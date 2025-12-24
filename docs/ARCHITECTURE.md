# OpenRLHF Agent Architecture

OpenRLHF-Agent uses the same primitives for RL rollouts and production inference: `Environment`, `ChatProtocol`, `AgentSession`, `AgentRuntime`, and provider `LLMEngine`s. Below is the short map of where things live and how tokens flow.

## Module layout

- `utils/types/`: shared dataclasses (`Message`, `ToolCall`, `Conversation`, `Action`, `Observation`, `RewardSample`).
- `agentkit/session.py`: keeps chat history, renders prompts, and applies parsed actions.
- `agentkit/runtime.py`: streams tokens through an `LLMEngine`, calling `AgentSession` each turn.
- `agentkit/environments/`: base contract plus `hub/function_call.py` (tool calling, default `CommentaryTool`) and `hub/single_turn.py`.
- `agentkit/tools/`: `ToolBase` and built-ins (`CommentaryTool`, `ThinkTool`, `FinalTool`).
- `agentkit/protocols/`: prompt/render/parse codecs (`hub/qwen3_instruct.py`, `hub/qwen3_thinking.py`).
- `agentkit/rewards/`: `RewardPipeline`, process reward (`process_rewards/hub/tool_call.py`), result rewards (`result_rewards/hub/matching.py` for string/math matching, `hub/grm.py`).
- `backends/`: `LLMEngine` interface and OpenAI/vLLM HTTP client (`hub/openai.py`).
- `examples/qwen3/`, `examples/single_turn/`: runnable demos for streaming and RL hooks.

## Runtime data flow

1. **Assembly**: pick an `Environment`, a `ChatProtocol`, an `LLMEngine`, optionally a `RewardPipeline`, then build `AgentSession` (or wrap it with `AgentRuntime` for inference).
2. **Initialization**: `AgentSession.initialize` resets steps, seeds the system prompt and prior turns, and renders the first prompt with the tool manifest; `AgentRuntime` tokenizes it.
3. **Stepping**: `LLMEngine.generate` produces text → `AgentSession.step_from_text` parses into an `Action` → `environment.step` runs tools/marks `done` → tool outputs are rendered back into the prompt; rewards are scored if attached.
4. **Streaming/termination**: `AgentRuntime.run_steps` yields assistant/tool messages each turn and stops when `done` or `max_steps` is hit (otherwise emits a max-steps warning).

## Extending the system

- **Rewards**: implement `ResultRewardStrategy` or `ProcessRewardStrategy` and plug them into `RewardPipeline`.
- **Tools**: subclass `ToolBase` and pass instances into environments or `env.register_tool(...)`.
- **Environments**: extend `Environment` and override `step`; instantiate the class directly for `AgentSession` or `AgentRuntime`.
- **Protocols**: subclass `ChatProtocol`, implement render/parse, and instantiate it directly.
- **Backends**: implement `LLMEngine` (`tokenize`, `generate`) and pass it to `AgentRuntime`.
