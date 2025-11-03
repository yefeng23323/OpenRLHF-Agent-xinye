import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .environment import Environment
from .model import LLMEngine
from .template import Template
from .types import ToolCall


@dataclass
class AgentStepResult:
    observations: List[str]
    reward: float
    terminated: bool
    rendered_observation: str
    actions: Optional[List[Optional[ToolCall]]]
    parse_error: bool = False


class AgentSession:
    """Shared logic for applying tool calls in both training and inference."""

    def __init__(self, environment: Environment, template: Template) -> None:
        self.environment = environment
        self.template = template

    def reset(self, initial_observation: Optional[str] = None) -> None:
        self.environment.reset(initial_observation)

    def build_system_prompt(self) -> str:
        return self.template.render_system(
            text=self.environment.system_prompt,
            tools_manifest=self.environment.tools_manifest(),
        )

    def render_observations(self, observations: Sequence[str]) -> str:
        blocks = [self.template.render_tool_response(obs) for obs in observations]
        return "\n".join(blocks)

    def build_user_turn(self, observations: Sequence[str]) -> str:
        return self.template.render_turn(
            role="user",
            text=self.render_observations(observations),
            add_generation_prompt=True,
        )

    def parse_actions(self, text: str) -> Tuple[bool, Optional[List[Optional[ToolCall]]]]:
        parse_error, actions = self.template.extract_tool_calls_from_text(text)
        return parse_error, actions

    def step(
        self,
        actions: Optional[Sequence[ToolCall | None]],
        *,
        label: Optional[str] = None,
        runtime: bool = False,
        parse_error: bool = False,
    ) -> AgentStepResult:
        observations, reward, terminated = self.environment.step(actions, label, runtime=runtime)
        return AgentStepResult(
            observations=observations,
            reward=reward,
            terminated=terminated,
            rendered_observation=self.build_user_turn(observations),
            actions=actions,
            parse_error=parse_error,
        )

    def step_from_text(
        self,
        action_text: str,
        *,
        label: Optional[str] = None,
        runtime: bool = False,
    ) -> AgentStepResult:
        parse_error, actions = self.parse_actions(action_text)
        if parse_error or not actions:
            return self.step(None, label=label, runtime=runtime, parse_error=True)
        return self.step(actions, label=label, runtime=runtime)


class AgentRuntime:
    """Lightweight runtime that streams tool calls and observations."""

    def __init__(
        self,
        engine: LLMEngine,
        environment: Environment,
        template: Template,
        *,
        max_new_tokens_per_step: int = 10240,
    ):
        self.engine = engine
        self.session = AgentSession(environment, template)
        self.environment = self.session.environment
        self.template = self.session.template
        self.max_new_tokens_per_step = max_new_tokens_per_step

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

    def run_steps(self, messages: List[Dict[str, str]]):
        """
        Streaming generator that yields OpenAI-compliant chat messages:
          - Assistant tool request: {"role": "assistant", "content": "", "tool_calls": [...]}
          - Tool response:         {"role": "tool", "tool_call_id": "...", "content": "..."}
          - Final answer:          {"role": "assistant", "content": "..."}
          - Error:                 {"role": "assistant", "content": "..."}  # e.g. parse failure
        """
        self.session.reset()

        prompt = self.template.render_messages(
            messages=[{"role": "system", "content": self.environment.system_prompt}, *messages],
            tools_manifest=self.environment.tools_manifest(),
            add_generation_prompt=True,
        )
        prompt_ids = self.engine.tokenize(prompt)

        for _ in range(self.environment.max_steps):
            action_ids, action_text = self.engine.generate(
                prompt_ids, max_tokens=self.max_new_tokens_per_step
            )

            step_result = self.session.step_from_text(action_text, runtime=True)

            if step_result.parse_error:
                # surface the raw model text so downstream logs can inspect it
                yield {"role": "assistant", "content": action_text}
                observation_ids = self.engine.tokenize(step_result.rendered_observation)
                prompt_ids += action_ids + observation_ids
                continue

            actions = step_result.actions or []
            tool_calls: List[Dict[str, Any]] = []
            for action in actions:
                if action is None:
                    continue
                tool_calls.append(
                    {
                        "id": action.id,
                        "type": "function",
                        "function": {
                            "name": action.name,
                            "arguments": json.dumps(action.arguments or {}, ensure_ascii=False),
                        },
                    }
                )

            if tool_calls:
                yield {"role": "assistant", "content": "", "tool_calls": tool_calls}

            for idx, obs in enumerate(step_result.observations):
                action = actions[idx] if idx < len(actions) else None
                if action is None:
                    continue
                if not self._is_internal_obs(obs):
                    yield {
                        "role": "tool",
                        "tool_call_id": action.id,
                        "content": obs,
                    }

            observation_ids = self.engine.tokenize(step_result.rendered_observation)
            prompt_ids += action_ids + observation_ids

            if step_result.terminated:
                final_text = ""
                for o in reversed(step_result.observations):
                    if not self._is_internal_obs(o):
                        final_text = o
                        break
                yield {"role": "assistant", "content": final_text}
                return

        yield {"role": "assistant", "content": "Max steps reached without final."}

    def run_final(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Convenience wrapper: returns the final assistant content or None.
        """
        final_text: Optional[str] = None
        for step in self.run_steps(messages):
            if step.get("role") != "assistant":
                continue
            if "tool_calls" in step:
                continue
            content = step.get("content")
            if content is not None:
                final_text = content
        return final_text


__all__ = ["AgentRuntime", "AgentSession", "AgentStepResult"]
