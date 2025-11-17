"""OpenRLHF integration example that scores rollouts with the GRM judge."""

import logging
import os
from typing import Any, Dict

import torch

from openrlhf_agent.agentkit.factory import (
    build_environment,
    build_protocol,
    build_result_reward,
)
from openrlhf_agent.agentkit.rewards import RewardPipeline
from openrlhf_agent.agentkit.session import AgentSession

from openrlhf.utils.agent import AgentExecutorBase, AgentInstanceBase


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AgentInstance(AgentInstanceBase):
    """Exact same agent wiring as examples/qwen3/agent_func.py, but GRM-scored."""

    def __init__(self, *_, **__):
        environment = build_environment(name="function_call")
        protocol = build_protocol(name="qwen3_thinking")
        pipeline = RewardPipeline(result_reward=build_result_reward(
            name="grm",
            config={
                "model": "{model_name}",
                "base_url": "{base_url}",
                "api_key": "{api_key}",
                "correct_score": 1.0,
                "format_score": 0.0,
                "error_score": -0.1,
            },
        ))
        self.session = AgentSession(
            environment=environment,
            protocol=protocol,
            reward_pipeline=pipeline,
        )

    async def reset(self, states: dict, **_) -> Dict[str, Any]:
        prompt = self.session.initialize(states.get("observation"))
        return {"observation": prompt}

    async def step(self, states: dict, **_) -> Dict[str, Any]:
        action_text: str = states.get("action_text", "")
        label = states.get("label")

        observation, reward = self.session.step_from_text(action_text, label=label)

        reward = float(reward) if reward is not None else 0.0
        if reward < -1:
            reward = -1.0

        done = observation.done
        return {
            "rewards": torch.tensor(reward),
            "scores": torch.tensor(reward),
            "environment_feedback": "" if done else observation.feedback_text,
            "done": done,
            "sampling_params": states.get("sampling_params"),
            "extra_logs": {
                "dummy_scores": torch.tensor(reward),
                "turn_count": torch.tensor(observation.step_index),
            },
        }


class AgentExecutor(AgentExecutorBase):
    """Adapter consumed by OpenRLHF's rollout workers."""

    def __init__(self, max_steps, max_length, llm_engine, hf_tokenizer, result_queue):
        super().__init__(AgentInstance, max_steps, max_length, llm_engine, hf_tokenizer, result_queue)
