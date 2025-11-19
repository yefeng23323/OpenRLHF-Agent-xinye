# OpenRLHF Agent Architecture

OpenRLHF-Agent keeps reinforcement learning rollouts and production inference on the same set of primitives: `Environment`, `ChatProtocol`, `AgentSession`, `AgentRuntime`, and provider-specific `LLMEngine`s. This document highlights how those components are wired inside `src/openrlhf_agent`.

## Module layout

- `utils/types.py`: shared dataclasses and Pydantic models (`Message`, `Action`, `ToolCall`, `Observation`, and `Conversation`).
- `agentkit/session.py`: stateful bridge between chat protocols and environments. Tracks the conversation transcript, renders prompts with the current tool manifest, and exposes `initialize`, `step`, and `step_from_text`.
- `agentkit/runtime.py`: inference-oriented driver that streams tokens via an `LLMEngine`, feeds them through `AgentSession`, and yields messages for the caller.
- `agentkit/environments/`: base contract plus the default `hub/function_call.py` (tool execution with JSON hints) and `hub/single_turn.py`.
- `agentkit/tools/`: `ToolBase` plus built-in helpers (`CommentaryTool`, `ThinkTool`, `FinalTool`) that can be passed into environments.
- `agentkit/protocols/`: codecs and chat templates. `hub/qwen3_instruct.py` and `hub/qwen3_thinking.py` render prompts and parse `<tool_call>` payloads.
- `agentkit/rewards/`: reward strategy base classes, result/process registries, and the `RewardPipeline` that composes them.
- `agentkit/factory.py`: `build_environment`, `build_protocol`, and `build_result_reward` helpers to resolve registry entries by name.
- `backends/`: `LLMEngine` interface plus the OpenAI/vLLM-compatible HTTP backend living in `hub/openai.py`.
- `examples/qwen3/`: end-to-end demos (runtime streaming, OpenRLHF agent wrappers, and REINFORCE++ scripts).

## Runtime data flow

1. **Assembly**  
   - Pick an `Environment` (`FunctionCallEnvironment` or custom).  
   - Choose a `ChatProtocol` implementation via `build_protocol`.  
   - Instantiate an `LLMEngine` such as `OpenAIEngine`.  
   - (Optional) create a `RewardPipeline` when training.  
   - Build `AgentSession(environment=..., protocol=..., reward_pipeline=...)` for training flows or wrap it with `AgentRuntime(engine, environment, protocol)` for inference.

2. **Initialization**  
   - `AgentRuntime.run_steps` (or trainers) call `await AgentSession.initialize(...)`.  
   - The session resets environment step counters, seeds the `Conversation` with the environment system prompt, optionally extends it with prior turns, and renders the first prompt using the protocol template plus the tool manifest returned by `environment.tools_manifest()`.  
   - `AgentRuntime` tokenizes this prompt via `await LLMEngine.tokenize` to bootstrap streaming.

3. **Stepping**  
   - For each turn, `await LLMEngine.generate` produces token IDs and decoded assistant text.  
   - `await AgentSession.step_from_text` forwards the text to the protocol parser, yielding an `Action` (plain response, tool calls, or refusals).  
   - The session persists the assistant message in the `Conversation` and invokes `await environment.step(action)` to execute tool calls and emit observation strings plus a `done` flag.  
   - Tool outputs are rendered back into prompt chunks with `protocol.render_messages(..., add_generation_prompt=True)` so the next LLM call sees both the assistant reply and tool feedback.  
   - When a `RewardPipeline` is attached and a label is provided, `await AgentSession.step` scores the action via optional process and result strategies.

4. **Streaming + termination**  
   - `AgentRuntime` yields each assistant/tool message via `async for` (`run_steps`) or returns the final assistant text (`run_final`).  
   - Iteration stops when the environment signals `done` or `max_steps` is hit, in which case a final assistant warning is produced.

## Extending the system

- **Rewards**: implement `ResultRewardStrategy` or `ProcessRewardStrategy` under `agentkit/rewards/` (`async def score`), then compose them with `RewardPipeline` before passing it to `AgentSession`.
- **Tools**: subclass `ToolBase` in `agentkit/tools/` (`async def call`) and supply instances to your environment constructor or call `env.register_tool(...)`.
- **Environments**: extend `Environment` and override `async def step` to enforce custom guardrails, tool schemas, or prompt policies. Register it in `agentkit/factory._ENVIRONMENT_REGISTRY` if you want `build_environment("name")` to discover it.
- **Protocols**: create a `ChatProtocol` subclass inside `agentkit/protocols/`, implement render/parse helpers, and register it so `build_protocol("name")` resolves it.
- **Backends**: implement `LLMEngine` under `backends/` (or `backends/hub/`) with `async def tokenize` / `async def generate` to integrate a new inference provider, then pass it to `AgentRuntime`.
