# OpenRLHF Agent 架构概览

## 模块分层
- `utils/types`: 约定 Action、Message、ToolCall、StepOutcome 等领域模型，任何层都可以安全依赖。
- `agentkit/`: 统一收纳协议、工具、奖励、环境、Session/Runtime 以及工厂方法。
  - `agentkit/tools` 仅描述工具 schema 与运行期注册。
  - `agentkit/rewards` 脱离环境，以 `RewardPipeline` 组合 process/result 策略（默认匹配策略为 `MatchingReward`）。
  - `agentkit/environments` 只关心工具交互/系统提示，reward 由 Session 根据 `Action` 决定。
  - `agentkit/session` 是训练/评估的主入口，负责调用环境、拼接提示词并触发奖励流水线。
  - `agentkit/runtime`（rollout loop）是 session 的轻量包装，负责与 LLM backend 交互。
  - `agentkit/protocols` 集中管理 Qwen3 instruct/thinking 等 chat codec。
- `backends/`: 定义 `LLMEngine` 接口并提供 OpenAI 兼容实现；未来可以在 `providers/` 下新增其它后端。
- `agentkit/factory.py`: 高阶装配器，提供 `build_environment/build_session/build_runtime` 等函数，用于减少业务接入的样板代码。

## 数据流
1. 上层通过 `agentkit.build_session`（训练）或 `agentkit.build_runtime`（推理）拉起组件，也可以手动组合 `Environment + ChatProtocol + LLMEngine + RewardPipeline`。
2. `AgentSession.initialize` 会重置环境、清空 `Conversation`，并利用协议渲染含工具清单的 prompt。
3. 每一步推理：
   - Runtime 或训练脚本调用 LLM backend 生成新的 action 文本。
   - Protocol 解析文本为 `Action`（含 tool calls/思维链）。
   - Environment 根据工具调用返回 observation 列表 + 是否终止 + 执行信息（例如 `used_tools`、`final_submitted`）。
   - Session 将 observation 重新渲染为提示词片段，并在训练模式下通过 `RewardPipeline` 决定奖励分数。
4. Runtime 负责在 streaming 环路里把 observation prompt 追加回 token 序列，并在终止时输出最终 assistant 内容。

- 新奖励策略：在 `agentkit/rewards` 内继承 `ResultRewardStrategy` 或 `ProcessRewardStrategy`，并交给 `RewardPipeline` 或 `register_result_reward` 管理。
- 新工具：在 `agentkit/tools` 中继承 `ToolBase`，可借助 `ToolRegistry` 做运行期注册。
- 新场景：在 `agentkit/environments` 创建环境实现，并通过 `agentkit.factory` 的注册表让 `build_environment` 识别它。
- 新协议：在 `agentkit/protocols` 实现 `ChatProtocol` 子类，并注册到工厂以便 `build_protocol` 调用。
- 新模型后端：在 `backends/providers` 中实现 `LLMEngine`，可交由 `agentkit.build_runtime` 的 `engine_factory` 注入。
