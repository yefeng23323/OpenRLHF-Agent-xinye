import json
from typing import Any, Dict, List, Optional, Tuple

from openrlhf_agent.runtime.engine import LLMEngine
from openrlhf_agent.environment import Environment
from openrlhf_agent.template import Template
from openrlhf_agent.utils.types import ToolCall


class AgentRuntime:
    def __init__(
        self,
        engine: LLMEngine,
        environment: Environment,
        template: Template,
        *,
        max_new_tokens_per_step: int = 10240,
    ):
        self.engine = engine
        self.environment = environment
        self.template = template
        self.max_new_tokens_per_step = max_new_tokens_per_step
    
    # ---------------- helpers ----------------

    @staticmethod
    def _is_internal_obs(text: str) -> bool:
        """
        Internal observations are JSON payloads with __internal=true (by our env design).
        Tool outputs (think/final) are plain text; they should NOT be JSON.
        """
        try:
            data = json.loads(text)
            return isinstance(data, dict) and (data.get("__internal") is True or data.get("visible_to_user") is False)
        except Exception:
            return False

    def _render_obs_for_model(self, observations: List[str]) -> str:
        """
        What we feed back to the model:
        - include both internal & visible observations (rich signal to self-correct)
        - each wrapped by template.render_tool_response
        """
        blocks = [self.template.render_tool_response(o) for o in observations]
        return "\n".join(blocks)

    def run_steps(self, messages: List[Dict[str, str]]):
        """
        Streaming generator. Yields dicts for UI/logging:
          - {"role": "tool_call", "name": ..., "arguments": {...}}
          - {"role": "tool_response", "content": "..."}      # visible only
          - {"role": "assistant", "content": "..."}          # final answer
          - {"role": "error", "content": "..."}              # parse errors, timeouts, etc.
        """
        self.environment.reset()

        # 1) Build initial prompt with tools manifest
        prompt = self.template.render_messages(
            [{"role": "system", "content": self.environment.system_prompt}, *messages],
            tools_manifest=self.environment.tools_manifest(),
            add_generation_prompt=True,
        )
        prompt_ids = self.engine.tokenize(prompt)

        # 2) Turn loop
        for _ in range(self.environment.max_steps):
            # 2.1 generate actions (tool calls block)
            action_ids, action_text = self.engine.generate(
                prompt_ids, max_tokens=self.max_new_tokens_per_step
            )
            parse_err, actions = self.template.extract_tool_calls_from_text(action_text)

            # 2.2 handle parse error or missing calls => internal hint from env
            if parse_err or not actions:
                # Optional: surface raw text for debugging
                yield {"role": "_inner", "content": action_text}

                internal_obs_list, _, _ = self.environment.step(None, runtime=True)  # List[str]
                # feed internal obs back to the model (not shown to user)
                next_user_turn = self.template.render_turn(
                    role="user",
                    text=self._render_obs_for_model(internal_obs_list),
                    add_generation_prompt=True,
                )
                observation_ids = self.engine.tokenize(next_user_turn)
                prompt_ids += action_ids + observation_ids
                continue
        
            # 2.3 log attempted tool calls for UI/debug
            for a in actions:
                if a is None:
                    continue
                yield {"role": "tool_call", "name": a.name, "arguments": a.arguments}
            
            # 2.4 execute actions in env
            obs_list, _, terminated = self.environment.step(actions, runtime=True)  # List[str], float, bool

            # 2.5 stream visible tool responses to UI
            for obs in obs_list:
                if not self._is_internal_obs(obs):
                    yield {"role": "tool_response", "content": obs}
            
            # 2.6 push ALL observations (internal + visible) back to the model
            user_turn = self.template.render_turn(
                role="user",
                text=self._render_obs_for_model(obs_list),
                add_generation_prompt=True,
            )
            observation_ids = self.engine.tokenize(user_turn)
            prompt_ids += action_ids + observation_ids

            # 2.7 termination: pick the last visible obs as final assistant output
            if terminated:
                final_text = ""
                for o in reversed(obs_list):
                    if not self._is_internal_obs(o):
                        final_text = o
                        break
                yield {"role": "assistant", "content": final_text}
                return  # done

        # 3) safety exit
        yield {"role": "error", "content": "Max steps reached without final."}

    def run_final(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Convenience wrapper: returns the final assistant content or None.
        """
        for step in self.run_steps(messages):
            # You can log or print(step) here if needed
            if step.get("role") == "assistant":
                return step.get("content")
        return None
