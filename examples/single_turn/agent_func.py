import logging
import torch

from typing import Any, Dict
from openrlhf_agent.agentkit.rewards import MatchingReward, RewardPipeline
from openrlhf_agent import AgentSession, build_environment, build_protocol
from openrlhf.utils.agent import AgentExecutorBase, AgentInstanceBase

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AgentInstance(AgentInstanceBase):
    def __init__(self, *args, **kwargs):
        environment = build_environment(name="single_turn")
        protocol = build_protocol(name="qwen3_instruct")
        pipeline = RewardPipeline(
            result_reward=MatchingReward(correct_score=1.0, miss_score=0.0)
        )
        self.session = AgentSession(environment=environment, protocol=protocol, reward_pipeline=pipeline)

    async def reset(self, states: dict, **kwargs):
        prompt = self.session.initialize(states.get("observation"))
        return {"observation": prompt}

    async def step(self, states: dict, **kwargs) -> Dict[str, Any]:
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
            "sampling_params": states.get("sampling_params", None),
            "extra_logs": {
                "dummy_scores": torch.tensor(reward),
                "turn_count": torch.tensor(observation.step_index),
            },
        }


class AgentExecutor(AgentExecutorBase):
    def __init__(self, max_steps, max_length, llm_engine, hf_tokenizer, result_queue):
        super().__init__(AgentInstance, max_steps, max_length, llm_engine, hf_tokenizer, result_queue)
