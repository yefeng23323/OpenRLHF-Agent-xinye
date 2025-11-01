import time
import torch
import logging

from typing import Any, Dict, List, Optional
from openrlhf_agent.environment import make_environment
from openrlhf_agent.template import make_template
from openrlhf.utils.agent import AgentExecutorBase, AgentInstanceBase

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AgentInstance(AgentInstanceBase):
    async def __init__(self, *args, **kwargs):
        self.step_idx = 0
        self.environment = make_environment(name="default")
        self.template = make_template("qwen3")
        # self.max_steps = self.environment.max_steps

    async def reset(self, states: dict, **kwargs):
        self.step_idx = 0

        self.environment.reset(states.get("observation"))  # TODO: message | question

        system_prompt = self.template.render_system(
            self.environment.system_prompt,
            tools_manifest=self.environment.tools_manifest(),
        )
        observation = system_prompt + states["observation"]
        return {"observation": observation}  # Return original text observation

    def _render_obs_block(self, obs_list: List[str]) -> str:
        """Wrap each observation via template.render_tool_response and join."""
        return "\n".join(self.template.render_tool_response(o) for o in obs_list)

    async def step(self, states: dict, **kwargs) -> Dict[str, Any]:
        # print(f"step_idx: {self.step_idx}, max_steps: {self.max_steps}")

        observation_text: str = states.get("observation_text", "")
        action_text: str = states.get("action_text", "")
        label: Optional[str] = states.get("label")

        # apply action and receive next observation, reward and whether the episode has ended
        tool_call_parse_error, actions = self.template.extract_tool_calls_from_text(action_text)
        if tool_call_parse_error or not actions:
            obs_list, reward, done = self.environment.step(None, label)
            next_observation = self.template.render_turn(
                role="user",
                text=self._render_obs_block(obs_list),
                add_generation_prompt=True,
            )
        
        else:
            obs_list, reward, done = self.environment.step(actions, label)
            next_observation = self.template.render_turn(
                role="user",
                text=self._render_obs_block(obs_list),
                add_generation_prompt=True,
            )

        if reward < -1:
            reward = -1.0

        # logger.info
        # print({
        #     "INFO": "##INFO##",
        #     "action": action_text,
        #     "observation": next_observation,
        #     "reward": reward,
        #     "done": done,
        #     "step_idx": self.step_idx,
        # })

        # Check if episode is done
        self.step_idx += 1
        # if self.step_idx >= self.max_steps:
        #     done = True

        return {
            "rewards": torch.tensor(reward),  # Rewards for advantage calculation
            "scores": torch.tensor(reward),  # Scores for dynamic filtering (0-1 reward)
            "environment_feedback": "" if done else next_observation,  # Environment feedback text
            "done": done,  # Boolean indicating if the episode is complete
            "sampling_params": states.get("sampling_params", None),  # Parameters for vLLM sampling in next step
            "extra_logs": {
                "dummy_scores": torch.tensor(reward),
                "turn_count": torch.tensor(self.step_idx),
            },  # Additional logging information
        }


class AgentExecutor(AgentExecutorBase):
    def __init__(self, max_steps, max_length, llm_engine, hf_tokenizer, result_queue):
        super().__init__(AgentInstance, max_steps, max_length, llm_engine, hf_tokenizer, result_queue)

    async def execute(self, prompt, label, sampling_params):
        # You could override the execute function of AgentExecutorBase to add custom agent running logic
        return await super().execute(prompt, label, sampling_params)