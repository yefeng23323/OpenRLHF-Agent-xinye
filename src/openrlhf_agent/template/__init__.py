from .base import Template


def make_template(name: str = None) -> Template:
    if name == "qwen3":
        from .qwen3 import Qwen3Template
        return Qwen3Template()

    raise ValueError(f"Unknown template: {name}")
