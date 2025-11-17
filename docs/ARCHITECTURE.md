# OpenRLHF Agent Architecture

## Module layout
- `utils/types`: shared domain models such as `Action`, `Message`, `ToolCall`, and `Observation`. Every layer can depend on them safely.
- `agentkit/`: home for chat protocols, tools, rewards, environments, the session/runtime objects, and factory helpers.
  - `agentkit/tools` defines tool schemas and exposes runtime registration helpers.
  - `agentkit/rewards` hosts independent reward strategies that the `RewardPipeline` can combine (default result reward is `MatchingReward`).
  - `agentkit/environments` handles tool orchestration and system prompts; the session decides which rewards to trigger based on each `Action`.
  - `agentkit/session` is the training/eval entry point. It calls the environment, stitches prompts, and runs the reward pipeline.
  - `agentkit/runtime` is a light wrapper around the session that drives the rollout loop with the LLM backend.
  - `agentkit/protocols` keeps concrete chat codecs such as Qwen3 instruct/thinking in one place.
- `backends/`: defines the `LLMEngine` interface and ships an OpenAI-compatible engine. New providers can live under `backends/hub`.
- `agentkit/factory.py`: high-level helpers such as `build_environment` and `build_protocol` to cut down boilerplate when wiring components.

## Data flow
1. Instantiate `AgentSession` for training or `AgentRuntime` for inference by wiring `Environment + ChatProtocol + LLMEngine + RewardPipeline`.
2. `AgentSession.initialize` resets the environment, clears the `Conversation`, and renders the first prompt with the current tool manifest.
3. Each reasoning step:
   - The runtime or trainer asks the LLM backend for the next action text.
   - The protocol parses the text into an `Action` (tool calls, reasoning traces, refusal flags, etc.).
   - The environment executes tool calls and returns observations, a `done` flag, and execution metadata (for example `used_tools` or `final_submitted`).
   - The session converts observations back into prompt chunks and, in training mode, feeds the `Action` into the `RewardPipeline`.
4. The runtime appends fresh observation prompts into the streaming token loop and surfaces the final assistant reply when `done` is true.

## Extending the system
- Reward: implement `ResultRewardStrategy` or `ProcessRewardStrategy` under `agentkit/rewards`, then wire it into your `RewardPipeline`.
- Tool: subclass `ToolBase` inside `agentkit/tools` and register it via `ToolRegistry`.
- Environment: add a class under `agentkit/environments`, then register it so `build_environment` can resolve it.
- Protocol: implement a `ChatProtocol` subclass under `agentkit/protocols` and register it for `build_protocol`.
- Backend: implement `LLMEngine` under `backends/hub` and inject it into `AgentRuntime` (or your own runtime builder) via the constructor.
